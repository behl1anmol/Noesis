"""Issue #25 — an automatic path back to a full, anchored walk (ADR-56/57).

PR #24 stopped discovery from reading a filesystem fault as deletion evidence.
Correct, but the guards it added could latch: a permanently unreadable
directory re-pended every stored path under it on every run, so the
"awaiting reindex" backlog never drained; and because every automated trigger
is *scoped*, nothing ever produced the full walk that drains the H1 dirty set.

Two mechanisms fix that, and the split between them is the whole safety
argument:

* **quarantine** (ADR-56) bounds the RETRY — after N consecutive failures a
  directory's hidden paths stop being re-queued. It never touches the DELETION
  policy: ADR-51/54 still decide that, so a quarantined subtree's content is
  kept and stays searchable. `test_quarantine_never_licenses_a_deletion` is the
  over-correction guard for exactly that, and it passes in both directions by
  construction.
* **promotion** (ADR-57) restores the full walk, which is what drains the dirty
  set and re-probes the tree.

The staleness hole that kept the re-pend alive in the PR #24 round-5 review — a
file edited while its directory was unreadable has already lost its watcher
event — is closed by re-admission instead:
`test_recovery_run_rehashes_the_subtree_and_catches_a_hidden_edit`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.app import AppContext
from noesis.core import jobs, state
from noesis.core import watcher as watcher_mod
from noesis.core.config import IndexingSettings
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import execute_run, is_attributable_prefix, prepare_run
from noesis.core.vectorstore import VectorStore

from tests.test_gitfast import make_env


def _fail_scandir_on(monkeypatch, *dirnames: str) -> None:
    """Make `os.scandir` fail for these directory names, as a dead mount or a
    root-owned directory does — the failure os.walk routes to `onerror`."""
    real_scandir = os.scandir
    targets = set(dirnames)

    def flaky(path=".", *args, **kwargs):
        if Path(path).name in targets:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)


def _fail_scandir_at_root(monkeypatch, root: Path) -> None:
    """Fail the walk ROOT itself — the unattributable case, where no prefix
    describes what went unseen and every stored path is unverified."""
    resolved = root.resolve()
    real_scandir = os.scandir

    def broken(path=".", *args, **kwargs):
        if Path(path) == resolved:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", broken)


def make_ctx(tmp_path, **indexing) -> AppContext:
    conn, store, embedder = make_env(tmp_path)
    return AppContext(
        conn=conn,
        store=store,
        embedder=embedder,
        indexing=IndexingSettings(**indexing),
    )


async def _run(conn, store, embedder, root, *, quarantine_after=5, **kwargs):
    project_id, run_id = prepare_run(conn, embedder, str(root))
    return await execute_run(
        conn,
        store,
        embedder,
        str(root),
        project_id,
        run_id,
        git_fast_path=False,
        unwalkable_quarantine_runs=quarantine_after,
        **kwargs,
    )


# --- unit: prefix attribution -------------------------------------------------


def test_is_attributable_prefix_rejects_what_no_prefix_describes() -> None:
    """The promotion policy and the re-admission union both branch on this, so
    the three "describes no subtree" shapes have to stay distinguishable."""
    assert is_attributable_prefix("pkg")
    assert is_attributable_prefix("a/b/c")
    assert not is_attributable_prefix("<root>")
    assert not is_attributable_prefix("")
    assert not is_attributable_prefix(os.path.join(os.sep, "mnt", "dead"))
    assert not is_attributable_prefix("../outside")


# --- the ledger ---------------------------------------------------------------


async def test_repeated_failure_quarantines_and_drains_the_backlog(
    tmp_path, monkeypatch
) -> None:
    """The defect issue #25 is titled after: under a permanently unreadable
    directory every run re-derived the same `unverified` set and re-pended it,
    so the backlog never drained."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "a.py").write_text("a = 1\n")
    (root / "pkg" / "b.py").write_text("b = 1\n")
    conn, store, embedder = make_env(tmp_path)

    first = await _run(conn, store, embedder, root)
    pid = first.project_id
    assert set(state.get_file_states(conn, pid)) == {"top.py", "pkg/a.py", "pkg/b.py"}

    _fail_scandir_on(monkeypatch, "pkg")
    # Runs 1 and 2 of 3: still retrying, so the paths keep coming back.
    for _ in range(2):
        result = await _run(conn, store, embedder, root, quarantine_after=3)
        assert set(result.failed_paths) == {"pkg/a.py", "pkg/b.py"}
        assert result.unverified_suppressed == 0

    # Run 3 crosses the threshold: the paths stop being re-queued.
    third = await _run(conn, store, embedder, root, quarantine_after=3)
    assert third.failed_paths == ()
    assert third.unverified_suppressed == 2

    rows = state.list_unwalkable_dirs(conn, pid)
    assert [r["dir_path"] for r in rows] == ["pkg"]
    assert rows[0]["consecutive_runs"] == 3
    assert rows[0]["quarantined_at"] is not None
    assert state.count_unwalkable_dirs(conn, pid) == (1, 1)


async def test_quarantine_never_licenses_a_deletion(tmp_path, monkeypatch) -> None:
    """THE over-correction guard. Quarantine bounds the retry; ADR-51/54 keep
    owning deletion. A quarantined subtree's files must stay in state and stay
    searchable no matter how many runs its directory fails — otherwise this
    change would have re-introduced exactly the purge-on-doubt PR #24 removed.

    Passes in both directions by construction, which is the point: it pins the
    invariant the new code must not break, not the new code's behaviour."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "a.py").write_text("a = 1\n")
    conn, store, embedder = make_env(tmp_path)

    first = await _run(conn, store, embedder, root)
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    for _ in range(6):  # well past a threshold of 2
        result = await _run(conn, store, embedder, root, quarantine_after=2)
        assert result.files_deleted == 0

    assert "pkg/a.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("pkg/a.py", 0) > 0


async def test_a_transient_failure_never_quarantines(tmp_path, monkeypatch) -> None:
    """One bad run is the case the re-pend exists for — a 9p blip, not a latch.
    The count must reset on the walk that succeeds, or a flaky mount would
    accumulate its way to a quarantine it never deserved."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("a = 1\n")
    conn, store, embedder = make_env(tmp_path)
    first = await _run(conn, store, embedder, root)
    pid = first.project_id

    for _ in range(4):
        _fail_scandir_on(monkeypatch, "pkg")
        bad = await _run(conn, store, embedder, root, quarantine_after=2)
        monkeypatch.undo()
        assert bad.failed_paths == ("pkg/a.py",)
        assert bad.unverified_suppressed == 0
        good = await _run(conn, store, embedder, root, quarantine_after=2)
        assert good.unverified_suppressed == 0
        assert state.list_unwalkable_dirs(conn, pid) == []


async def test_quarantine_is_scoped_to_its_own_subtree(tmp_path, monkeypatch) -> None:
    """Containment on the separator, like ADR-54's own suppression: `pkg2/` was
    walked fine and must keep being retried while `pkg/` is quarantined."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg2").mkdir()
    (root / "pkg" / "a.py").write_text("a = 1\n")
    (root / "pkg2" / "b.py").write_text("b = 1\n")
    conn, store, embedder = make_env(tmp_path)
    await _run(conn, store, embedder, root)

    # `pkg` is long-latched; `pkg2` fails for the first time on the last run.
    _fail_scandir_on(monkeypatch, "pkg")
    for _ in range(2):
        await _run(conn, store, embedder, root, quarantine_after=2)
    monkeypatch.undo()
    _fail_scandir_on(monkeypatch, "pkg", "pkg2")
    result = await _run(conn, store, embedder, root, quarantine_after=2)

    assert result.failed_paths == ("pkg2/b.py",)
    assert result.unverified_suppressed == 1


async def test_unattributable_failure_stands_down_only_when_all_are_quarantined(
    tmp_path, monkeypatch
) -> None:
    """When the walk ROOT fails, no prefix describes what went unseen, so the
    choice is all-or-nothing over the whole index. A single fresh failure means
    this run's doubt is new — and new doubt is what the retry is for."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)
    await _run(conn, store, embedder, root)

    _fail_scandir_at_root(monkeypatch, root)
    first = await _run(conn, store, embedder, root, quarantine_after=2)
    assert set(first.failed_paths) == {"a.py", "b.py"}  # whole index re-pended
    assert first.unverified_suppressed == 0

    second = await _run(conn, store, embedder, root, quarantine_after=2)
    assert second.failed_paths == ()
    assert second.unverified_suppressed == 2
    # Still no deletion: an unreadable root is never deletion evidence.
    assert second.files_deleted == 0


async def test_quarantine_can_be_disabled(tmp_path, monkeypatch) -> None:
    """`0` restores the pre-ADR-56 behaviour exactly — the paths keep coming
    back forever. An operator who prefers the visible non-draining backlog
    keeps it."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("a = 1\n")
    conn, store, embedder = make_env(tmp_path)
    first = await _run(conn, store, embedder, root)
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    for _ in range(5):
        result = await _run(conn, store, embedder, root, quarantine_after=0)
        assert result.failed_paths == ("pkg/a.py",)
        assert result.unverified_suppressed == 0
    rows = state.list_unwalkable_dirs(conn, pid)
    assert rows[0]["consecutive_runs"] == 5
    assert rows[0]["quarantined_at"] is None


# --- recovery -----------------------------------------------------------------


async def test_recovery_run_rehashes_the_subtree_and_catches_a_hidden_edit(
    tmp_path, monkeypatch
) -> None:
    """The round-5 staleness objection, closed without the re-pend.

    A file edited while its directory was unreadable has already lost its
    watcher event, so it is in no scoped run's candidate set. The run that
    finds the directory walkable again must therefore re-admit what it holds —
    or the stale content sits in the index until a human triggers a full run.
    """
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("original = 1\n")
    (root / "top.py").write_text("t = 1\n")
    conn, store, embedder = make_env(tmp_path)
    first = await _run(conn, store, embedder, root)
    pid = first.project_id
    original_hash = state.get_file_states(conn, pid)["pkg/a.py"]

    # Quarantine it, then edit a file behind its back.
    _fail_scandir_on(monkeypatch, "pkg")
    for _ in range(2):
        await _run(conn, store, embedder, root, quarantine_after=2)
    (root / "pkg" / "a.py").write_text("edited = 2\n")
    monkeypatch.undo()

    # A SCOPED run whose candidate set does not mention the edited file: only
    # the re-admission can bring it back into the hash set.
    recovery = await _run(
        conn, store, embedder, root, quarantine_after=2, paths=["top.py"]
    )

    # The edit is picked up, and picked up as ordinary indexing work — not as a
    # failure, and not by deleting anything.
    assert state.get_file_states(conn, pid)["pkg/a.py"] != original_hash
    assert recovery.files_indexed >= 1
    assert recovery.files_deleted == 0
    assert recovery.failed_paths == ()
    assert state.list_unwalkable_dirs(conn, pid) == []
    assert state.count_unwalkable_dirs(conn, pid) == (0, 0)


async def test_a_deleted_directory_clears_its_row_and_deletes_normally(
    tmp_path, monkeypatch
) -> None:
    """Recovery is "the walk reached it", not "it is readable again". A
    directory that was genuinely removed raises nothing — os.walk simply does
    not list it — so its row clears and its files become provably absent."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("a = 1\n")
    (root / "top.py").write_text("t = 1\n")
    conn, store, embedder = make_env(tmp_path)
    first = await _run(conn, store, embedder, root)
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    await _run(conn, store, embedder, root, quarantine_after=2)
    monkeypatch.undo()
    assert state.list_unwalkable_dirs(conn, pid) != []

    (root / "pkg" / "a.py").unlink()
    (root / "pkg").rmdir()
    result = await _run(conn, store, embedder, root, quarantine_after=2)

    assert result.files_deleted == 1
    assert set(state.get_file_states(conn, pid)) == {"top.py"}
    assert state.list_unwalkable_dirs(conn, pid) == []


async def test_ledger_survives_a_crash_after_discovery(tmp_path, monkeypatch) -> None:
    """The read-early/write-late ordering. If the reconcile ran before the file
    loop, a crash mid-run would delete a recovered directory's row while its
    re-admission never landed — the ledger would read healthy and nothing would
    ever re-hash those files."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("a = 1\n")
    conn, store, embedder = make_env(tmp_path)
    first = await _run(conn, store, embedder, root)
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    await _run(conn, store, embedder, root, quarantine_after=2)
    monkeypatch.undo()
    assert [r["dir_path"] for r in state.list_unwalkable_dirs(conn, pid)] == ["pkg"]

    # A recovering run that dies partway through must leave the row alone.
    def boom(*args, **kwargs):
        raise RuntimeError("crash after discovery")

    monkeypatch.setattr(state, "get_file_chunk_counts", boom)
    with pytest.raises(RuntimeError):
        await _run(conn, store, embedder, root, quarantine_after=2)
    monkeypatch.undo()

    assert [r["dir_path"] for r in state.list_unwalkable_dirs(conn, pid)] == ["pkg"]


# --- promotion ----------------------------------------------------------------


def test_promotion_fires_on_the_scoped_run_count(tmp_path) -> None:
    ctx = make_ctx(tmp_path, promote_after_scoped_runs=3, promote_candidate_fraction=0)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)

    full, _ = state.try_start_run(ctx.conn, pid, scoped=False)
    state.finish_run(ctx.conn, full, "done")
    assert jobs._promotion_reason(ctx, pid, ["a.py"]) is None

    for _ in range(3):
        run, _ = state.try_start_run(ctx.conn, pid, scoped=True)
        state.finish_run(ctx.conn, run, "done")
    reason = jobs._promotion_reason(ctx, pid, ["a.py"])
    assert reason is not None and "scoped run" in reason


def test_promotion_fires_when_the_candidate_set_stops_being_narrow(tmp_path) -> None:
    ctx = make_ctx(
        tmp_path, promote_after_scoped_runs=0, promote_candidate_fraction=0.25
    )
    root = tmp_path / "proj"
    root.mkdir()
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    for i in range(20):
        state.upsert_file(ctx.conn, pid, f"f{i}.py", f"hash{i}")

    assert jobs._promotion_reason(ctx, pid, ["f0.py"]) is None
    # The H1 dirty set counts toward it: that accumulation is what makes a
    # scoped run stop being cheap (issue #25's second half).
    state.add_dirty_paths(ctx.conn, pid, [f"f{i}.py" for i in range(5)])
    reason = jobs._promotion_reason(ctx, pid, ["f0.py"])
    assert reason is not None and "candidate set" in reason


def test_an_unattributable_ledger_row_forces_a_full_walk(tmp_path) -> None:
    """A vanished root leaves every stored path unverified, and no scoped run
    can be wide enough to help. The re-admission union deliberately skips these
    (it would silently become a whole-index hash), so promotion is what covers
    them."""
    ctx = make_ctx(tmp_path, promote_after_scoped_runs=0, promote_candidate_fraction=0)
    root = tmp_path / "proj"
    root.mkdir()
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)

    state.reconcile_unwalkable_dirs(
        ctx.conn, pid, reported=[("pkg", "EACCES")], cleared=[], quarantine_after=5
    )
    assert jobs._promotion_reason(ctx, pid, ["a.py"]) is None

    state.reconcile_unwalkable_dirs(
        ctx.conn, pid, reported=[("<root>", "ENOENT")], cleared=[], quarantine_after=5
    )
    reason = jobs._promotion_reason(ctx, pid, ["a.py"])
    assert reason is not None and "unattributable" in reason


async def test_promotion_is_disableable_and_never_fires_for_a_full_caller(
    tmp_path,
) -> None:
    ctx = make_ctx(tmp_path, promote_after_scoped_runs=0, promote_candidate_fraction=0)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    for _ in range(50):
        run, _ = state.try_start_run(ctx.conn, pid, scoped=True)
        state.finish_run(ctx.conn, run, "done")
    assert jobs._promotion_reason(ctx, pid, ["a.py"]) is None

    # A caller that already asked for a full walk is never routed through the
    # policy at all, so `scoped` is recorded honestly for the counter.
    result = jobs.launch_index_run(ctx, str(root))
    assert result["status"] == "accepted"
    await ctx.jobs[result["run_id"]]
    row = state.get_latest_run(ctx.conn, pid)
    assert row["scoped"] == 0


async def test_a_promoted_run_drains_the_dirty_set_and_clears_pending(
    tmp_path,
) -> None:
    """The second half of issue #25: `dirty_paths` only ever grew, because
    every automated trigger is scoped and only a full run drains it. A promoted
    run must take the drain branch AND clear pending project-wide — the two go
    together, since `paths` feeds both."""
    ctx = make_ctx(tmp_path, promote_after_scoped_runs=1, promote_candidate_fraction=0)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")

    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    state.add_dirty_paths(ctx.conn, pid, ["a.py", "b.py"])
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])
    scoped, _ = state.try_start_run(ctx.conn, pid, scoped=True)
    state.finish_run(ctx.conn, scoped, "done")
    assert state.get_dirty_paths(ctx.conn, pid) == frozenset({"a.py", "b.py"})

    launched = jobs.launch_index_run(ctx, str(root), paths=["a.py"])
    await ctx.jobs[launched["run_id"]]

    assert state.get_latest_run(ctx.conn, pid)["scoped"] == 0
    assert state.get_dirty_paths(ctx.conn, pid) == frozenset()
    assert state.list_pending_changes(ctx.conn, pid) == []


# --- watcher catch-up ---------------------------------------------------------


async def test_watcher_start_launches_one_full_run_when_a_dir_is_unwalkable(
    tmp_path, monkeypatch
) -> None:
    """Once quarantine drains the backlog a project can have no pending rows
    left, and every other automated trigger fires off a filesystem event — which
    a dead mount does not produce. Process start is the remaining re-probe."""
    ctx = make_ctx(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    state.set_project_flags(ctx.conn, pid, watch_enabled=True, auto_reindex=True)
    state.reconcile_unwalkable_dirs(
        ctx.conn, pid, reported=[("pkg", "EACCES")], cleared=[], quarantine_after=5
    )

    launches: list[object] = []
    monkeypatch.setattr(
        jobs,
        "launch_index_run",
        lambda ctx_, rp, **kw: (
            launches.append(kw.get("paths")),
            {"project_id": pid, "run_id": "r", "status": "accepted"},
        )[1],
    )
    manager = watcher_mod.WatcherManager(ctx)
    monkeypatch.setattr(manager, "_schedule", lambda *a, **k: None)
    manager.start()
    try:
        assert launches == [None]  # exactly one launch, and it is a full walk
    finally:
        await manager.stop()


async def test_watcher_start_still_scopes_a_healthy_project(
    tmp_path, monkeypatch
) -> None:
    """Over-correction guard: no ledger rows means the ordinary scoped catch-up
    is unchanged."""
    ctx = make_ctx(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    state.set_project_flags(ctx.conn, pid, watch_enabled=True, auto_reindex=True)
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])

    launches: list[object] = []
    monkeypatch.setattr(
        jobs,
        "launch_index_run",
        lambda ctx_, rp, **kw: (
            launches.append(kw.get("paths")),
            {"project_id": pid, "run_id": "r", "status": "accepted"},
        )[1],
    )
    manager = watcher_mod.WatcherManager(ctx)
    monkeypatch.setattr(manager, "_schedule", lambda *a, **k: None)
    manager.start()
    try:
        assert launches == [["a.py"]]
    finally:
        await manager.stop()


# --- state bookkeeping --------------------------------------------------------


def test_reconcile_is_idempotent_and_delete_project_leaves_no_orphans(
    tmp_path,
) -> None:
    conn, _, _ = make_env(tmp_path)
    pid = state.register_project(conn, tmp_path / "proj", "m1")

    state.reconcile_unwalkable_dirs(
        conn, pid, reported=[], cleared=[], quarantine_after=3
    )
    assert state.list_unwalkable_dirs(conn, pid) == []
    # Clearing a path that was never recorded is a no-op, not an error.
    state.reconcile_unwalkable_dirs(
        conn, pid, reported=[], cleared=["never-seen"], quarantine_after=3
    )
    assert state.list_unwalkable_dirs(conn, pid) == []

    state.reconcile_unwalkable_dirs(
        conn, pid, reported=[("pkg", "EACCES")], cleared=[], quarantine_after=3
    )
    first_seen = state.list_unwalkable_dirs(conn, pid)[0]["first_seen_at"]
    state.reconcile_unwalkable_dirs(
        conn, pid, reported=[("pkg", "EIO")], cleared=[], quarantine_after=3
    )
    row = state.list_unwalkable_dirs(conn, pid)[0]
    assert row["consecutive_runs"] == 2
    assert row["first_seen_at"] == first_seen  # the clock starts once
    assert row["last_error"] == "EIO"  # the newest reason wins

    state.delete_project(conn, pid)
    assert conn.execute("SELECT COUNT(*) FROM unwalkable_dirs").fetchone()[0] == 0


def test_scoped_runs_since_full_ignores_failed_full_runs(tmp_path) -> None:
    """A failed full run did not drain anything — `clear_dirty_paths` and the
    anchor write both need a clean run — so treating it as a reset would let a
    project that fails every full walk stop promoting entirely."""
    conn, _, _ = make_env(tmp_path)
    pid = state.register_project(conn, tmp_path / "proj", "m1")

    for _ in range(3):
        run, _ = state.try_start_run(conn, pid, scoped=True)
        state.finish_run(conn, run, "done")
    assert state.scoped_runs_since_full(conn, pid) == 3

    failed_full, _ = state.try_start_run(conn, pid, scoped=False)
    state.finish_run(conn, failed_full, "failed", error="boom")
    assert state.scoped_runs_since_full(conn, pid) == 3

    good_full, _ = state.try_start_run(conn, pid, scoped=False)
    state.finish_run(conn, good_full, "done")
    assert state.scoped_runs_since_full(conn, pid) == 0


# --- status surfaces ----------------------------------------------------------


async def test_index_status_reports_coverage_health(tmp_path) -> None:
    """An agent searching a project is entitled to know part of it could not be
    read before trusting an answer, so this rides the shared REST/MCP shape."""
    ctx = make_ctx(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)

    healthy = await jobs.index_status(ctx, pid)
    assert healthy["unwalkable_dirs"] == 0
    assert healthy["quarantined_dirs"] == 0

    state.reconcile_unwalkable_dirs(
        ctx.conn,
        pid,
        reported=[("pkg", "EACCES"), ("other", "EIO")],
        cleared=[],
        quarantine_after=1,
    )
    degraded = await jobs.index_status(ctx, pid)
    assert degraded["unwalkable_dirs"] == 2
    assert degraded["quarantined_dirs"] == 2


def test_project_detail_counts_the_files_each_directory_hides(tmp_path) -> None:
    """The figure that tells an operator whether this is a stray subdirectory
    or most of the project. Counted from stored state, because the whole
    problem is that the directory cannot be read."""
    from noesis.core import dashboard

    ctx = make_ctx(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
    for path in ("pkg/a.py", "pkg/b.py", "pkg2/c.py", "top.py"):
        state.upsert_file(ctx.conn, pid, path, "h")
    state.reconcile_unwalkable_dirs(
        ctx.conn,
        pid,
        reported=[("pkg", "EACCES"), ("<root>", "ENOENT")],
        cleared=[],
        quarantine_after=99,
    )

    detail = dashboard.project_detail(ctx, pid)
    hidden = {d["dir_path"]: d["files_hidden"] for d in detail["unwalkable_dirs"]}
    # `pkg` hides its own two — never `pkg2/c.py`, which shares a prefix but
    # not a path segment.
    assert hidden["pkg"] == 2
    # `<root>` describes no subtree, so everything is behind it.
    assert hidden["<root>"] == 4
    assert detail["unwalkable_count"] == 2


def test_indexing_settings_reject_an_out_of_range_fraction(tmp_path) -> None:
    from noesis.core.config import load_settings

    path = tmp_path / "config.toml"
    path.write_text("[indexing]\npromote_candidate_fraction = 1.5\n")
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_settings(path)

    path.write_text("[indexing]\nunwalkable_quarantine_runs = -1\n")
    with pytest.raises(ValueError, match=">= 0"):
        load_settings(path)

    path.write_text(
        "[indexing]\npromote_after_scoped_runs = 0\n"
        "promote_candidate_fraction = 0\nunwalkable_quarantine_runs = 0\n"
    )
    cfg = load_settings(path).indexing
    assert (cfg.promote_after_scoped_runs, cfg.unwalkable_quarantine_runs) == (0, 0)
