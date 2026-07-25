"""Regression tests for the 2026-07-20 bug hunt.

PIPE-1 (high): a transient fs error during discovery must never purge live
files as "deleted". Discovery now reports per-file and per-directory errors
(DiscoveryErrors); partition gives discovery-errored files the H7 carry-
forward; the indexer skips deletions when a subtree went unwalked and blocks
anchor advance on any discovery error.

PIPE-2 (medium): scoped (watcher) runs never advance the anchor, so they
never persisted dirty_paths — content indexed only by scoped runs was
invisible to the H1 re-admission and a revert-while-unwatched stayed stale
forever. Scoped runs now union their indexed paths into the dirty set.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import pytest

from noesis.core import discovery, state
from noesis.core.discovery import DiscoveryErrors, discover_files
from noesis.core.hashdiff import partition
from noesis.core.indexer import execute_run, prepare_run

from tests.test_gitfast import (
    anchor_of,
    build_git_repo,
    git,
    git_head,
    make_env,
    requires_git,
)


# --- PIPE-1 unit: discover_files error collection -----------------------------


def test_discover_files_collects_transient_file_errors(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    real = discovery._is_binary

    def flaky(path):
        if path.name == "a.py":
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(discovery, "_is_binary", flaky)
    errors = DiscoveryErrors()
    files = discover_files(tmp_path, errors=errors)

    # a.py failed screening: excluded from results but surfaced, not silent.
    assert files == ["b.py"]
    assert [p for p, _ in errors.files] == ["a.py"]
    assert "Permission denied" in dict(errors.files)["a.py"]
    assert errors.dirs == []
    # errors=None keeps the historical silent skip byte-identical.
    assert discover_files(tmp_path) == ["b.py"]


def test_discover_files_gone_file_is_not_an_error(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")

    def gone(path):
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(discovery, "_is_binary", gone)
    errors = DiscoveryErrors()
    # Vanished mid-walk is a genuine deletion (H7 discrimination): no error.
    assert discover_files(tmp_path, errors=errors) == []
    assert errors.files == [] and errors.dirs == []


def test_discover_files_records_dir_scan_errors(tmp_path, monkeypatch):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("x = 1\n")
    (tmp_path / "top.py").write_text("y = 2\n")
    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == "pkg":
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)
    errors = DiscoveryErrors()
    files = discover_files(tmp_path, errors=errors)

    assert files == ["top.py"]
    assert [p for p, _ in errors.dirs] == ["pkg"]
    assert errors.files == []


def test_discover_files_root_scan_error_reports_root(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    resolved = tmp_path.resolve()
    real_scandir = os.scandir

    def broken(path=".", *args, **kwargs):
        if Path(path) == resolved:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", broken)
    errors = DiscoveryErrors()
    assert discover_files(tmp_path, errors=errors) == []
    assert [p for p, _ in errors.dirs] == ["<root>"]


# --- PIPE-1 unit: partition treats discovery errors like H7 -------------------


def test_partition_discovery_errored_carries_forward(tmp_path):
    (tmp_path / "b.py").write_text("y = 2\n")
    stored = {"a.py": "STORED_HASH", "b.py": "OLD_HASH"}

    res = partition(
        tmp_path, ["b.py"], stored, discovery_errored=[("a.py", "EACCES")]
    )

    # a.py never entered `discovered`, yet it must not read as deleted.
    assert "a.py" not in res.deleted
    assert "a.py" in res.unchanged
    assert res.hashes["a.py"] == "STORED_HASH"
    assert ("a.py", "EACCES") in res.errored
    # b.py hashed normally and differs from stored → changed.
    assert "b.py" in res.changed


def test_partition_discovery_errored_unknown_file_reported_only(tmp_path):
    # A file unknown to stored state: nothing to preserve, but the failure
    # is still surfaced (same as the H7 hash-time branch).
    res = partition(tmp_path, [], {}, discovery_errored=[("new.py", "EIO")])
    assert res.errored == (("new.py", "EIO"),)
    assert res.new == () and res.deleted == ()
    assert "new.py" not in res.hashes


# --- PIPE-1 integration: transient discovery error is not a deletion ----------


@requires_git
async def test_file_discovery_error_not_a_deletion_and_blocks_anchor(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    build_git_repo(root)
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(conn, store, embedder, str(root), project_id, run_id)

    result = await run()
    pid = result.project_id
    c0 = git_head(root)
    assert anchor_of(conn, pid) == c0

    (root / "beta.py").write_text("def beta():\n    return 200\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change beta")
    c1 = git_head(root)

    real = discovery._is_binary

    def flaky(path):
        if path.name == "alpha.py":
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(discovery, "_is_binary", flaky)
    result = await run()
    monkeypatch.undo()

    # alpha.py errored in discovery: pre-fix it fell out of the walk silently
    # and was purged as deleted. Now its state and chunks survive, the run is
    # honest about the failure, and the anchor stays put.
    assert "alpha.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("alpha.py", 0) > 0
    assert result.files_deleted == 0
    assert result.files_failed == 1
    assert "alpha.py" in result.failed_paths
    assert anchor_of(conn, pid) == c0
    run_row = state.get_latest_run(conn, pid)
    assert run_row["status"] == "done"

    # Healed: the next clean run advances the anchor normally.
    result = await run()
    assert result.files_failed == 0
    assert anchor_of(conn, pid) == c1


@requires_git
async def test_dir_walk_error_skips_deletions_and_blocks_anchor(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    build_git_repo(
        root,
        {
            "top.py": "def top():\n    return 1\n",
            "pkg/mod.py": "def mod():\n    return 2\n",
        },
    )
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(conn, store, embedder, str(root), project_id, run_id)

    result = await run()
    pid = result.project_id
    c0 = git_head(root)
    assert anchor_of(conn, pid) == c0

    (root / "top.py").write_text("def top():\n    return 100\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change top")
    c1 = git_head(root)

    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == "pkg":
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)
    result = await run()
    monkeypatch.undo()

    # pkg/ went unwalked: its file must survive (deletion evidence is
    # untrustworthy this run), the dir error must count as a failure, and
    # the anchor must not advance past it.
    assert "pkg/mod.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0
    assert result.files_deleted == 0
    assert result.files_failed >= 1
    assert anchor_of(conn, pid) == c0
    assert state.get_latest_run(conn, pid)["status"] == "done"

    # Clean run: pkg/mod.py is rediscovered unchanged, anchor advances.
    result = await run()
    assert result.files_failed == 0
    assert result.files_deleted == 0
    assert anchor_of(conn, pid) == c1


async def test_root_scan_failure_causes_no_mass_delete(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert first.files_indexed == 2

    resolved = root.resolve()
    real_scandir = os.scandir

    def broken(path=".", *args, **kwargs):
        if Path(path) == resolved:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", broken)
    second = await run()
    monkeypatch.undo()

    # Pre-fix worst case: root scandir error → discovery empty → every stored
    # file "deleted" → full purge, run "done". Now: nothing purged, run failed.
    assert set(state.get_file_states(conn, pid)) == {"a.py", "b.py"}
    counts = store.per_file_point_counts(pid)
    assert counts.get("a.py", 0) > 0 and counts.get("b.py", 0) > 0
    assert second.files_indexed == 0
    run_row = state.get_latest_run(conn, pid)
    assert run_row["status"] == "failed"


async def test_genuine_deletion_still_purges(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert first.files_indexed == 2

    (root / "b.py").unlink()
    second = await run()

    # A real deletion (no discovery errors) still purges chunks and state.
    assert second.files_deleted == 1
    assert "b.py" not in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("b.py", 0) == 0


# --- PIPE-2: scoped runs persist the dirty paths they indexed -----------------


@requires_git
async def test_scoped_run_persists_dirty_paths_so_revert_is_reindexed(
    tmp_path,
) -> None:
    committed = "def alpha():\n    return 1\n"
    dirty = "def alpha():\n    return 999\n"
    root = tmp_path / "repo"
    build_git_repo(root, {"alpha.py": committed, "beta.py": "def beta():\n    return 2\n"})
    conn, store, embedder = make_env(tmp_path)

    async def run(paths=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, paths=paths
        )

    # Run 1: full walk, anchor recorded, nothing dirty.
    pid = (await run()).project_id
    committed_hash = hashlib.sha256(committed.encode()).hexdigest()
    assert state.get_file_states(conn, pid)["alpha.py"] == committed_hash

    # Run 2: the watcher indexes the uncommitted edit via a SCOPED run —
    # pre-fix, no set_last_indexed_commit fires (scoped runs never advance
    # the anchor), so the dirty set was never persisted.
    (root / "alpha.py").write_text(dirty)
    await run(paths=["alpha.py"])
    dirty_hash = hashlib.sha256(dirty.encode()).hexdigest()
    assert state.get_file_states(conn, pid)["alpha.py"] == dirty_hash
    assert "alpha.py" in state.get_dirty_paths(conn, pid)

    # Revert to HEAD content while the watcher isn't observing: neither the
    # commit diff nor `git status` names alpha.py now.
    (root / "alpha.py").write_text(committed)

    # Run 3: full fast-path run. Pre-fix the stale dirty hash carried forward
    # forever; with the fix the persisted dirty set re-admits alpha.py, the
    # revert is detected, and it rotates out of the dirty set once clean.
    result = await run()
    assert result.fast_path_used
    assert state.get_file_states(conn, pid)["alpha.py"] == committed_hash
    assert "alpha.py" not in state.get_dirty_paths(conn, pid)


def test_add_dirty_paths_unions_without_touching_anchor(tmp_path) -> None:
    conn = state.connect(tmp_path / "s.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, tmp_path, "m")
    state.set_last_indexed_commit(conn, pid, "c0", dirty_paths=["a.py"])

    state.add_dirty_paths(conn, pid, ["b.py", "a.py"])
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py", "b.py"})
    # Union-only, and the anchor is untouched.
    assert state.get_project(conn, pid)["last_indexed_commit"] == "c0"

    # Empty union and unknown project are safe no-ops.
    state.add_dirty_paths(conn, pid, [])
    state.add_dirty_paths(conn, "no-such-project", ["x.py"])
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py", "b.py"})
    conn.close()


# --- PR #24 review (Codex): discovery errors must not leak into the two
# other purge paths -----------------------------------------------------------


async def test_dir_walk_error_suppresses_orphan_pruning(tmp_path, monkeypatch) -> None:
    """A failed directory walk must not let drift cleanup prune live points.

    `orphan_paths` is computed from the same incomplete `discovered_set` the
    deletion guard already distrusts. A file live on disk whose state row is
    missing (a crash between `upsert_chunks` and `upsert_file`) is exactly the
    shape that lands there, so an unwalked directory made drift cleanup delete
    searchable points for a live file — the purge-on-doubt case PIPE-1 exists
    to close.
    """
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "mod.py").write_text("def mod():\n    return 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0

    # Simulate the crash window: points are in Qdrant, the state row is not.
    # This both makes pkg/mod.py an orphan candidate and trips the drift gate
    # (expected chunk total now disagrees with the live point count).
    state.delete_files(conn, pid, ["pkg/mod.py"])
    assert "pkg/mod.py" not in state.get_file_states(conn, pid)

    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == "pkg":
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)
    second = await run()
    monkeypatch.undo()

    # The file is live on disk; only the walk failed. Its points must survive.
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0
    assert second.files_deleted == 0

    # And the clean run heals it: rediscovered, unknown to state, re-indexed.
    third = await run()
    assert "pkg/mod.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0
    assert third.files_failed == 0


async def test_genuine_orphan_still_pruned_on_a_clean_run(tmp_path) -> None:
    """The guard above must not disable orphan pruning outright: with no
    discovery error, points for a path absent from both state and disk are
    still pruned (ADR-49), or the drift gate fires forever."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "ghost.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert store.per_file_point_counts(pid).get("ghost.py", 0) > 0

    # Neither tracked in state nor present on disk — a true orphan.
    state.delete_files(conn, pid, ["ghost.py"])
    (root / "ghost.py").unlink()

    await run()
    assert store.per_file_point_counts(pid).get("ghost.py", 0) == 0


async def test_scoped_dir_error_requeues_the_scoped_file_not_just_the_dir(
    tmp_path, monkeypatch
) -> None:
    """A scoped retry matches its candidate set exactly, with no prefix
    expansion, so re-pending only the errored directory would make the retry
    carry every stored child forward as unchanged and clear the pending row —
    stranding the edit until a manual full run. The suppressed deletions are
    re-queued as the file paths they are.
    """
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "mod.py").write_text("def mod():\n    return 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(paths=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            paths=paths,
        )

    first = await run()
    pid = first.project_id
    original_hash = state.get_file_states(conn, pid)["pkg/mod.py"]

    # The watcher saw this edit and scoped a run to it...
    (root / "pkg" / "mod.py").write_text("def mod():\n    return 999\n")

    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == "pkg":
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    # ...but the walk of pkg/ failed, so the run never read the file.
    monkeypatch.setattr(os, "scandir", flaky)
    second = await run(paths=["pkg/mod.py"])
    monkeypatch.undo()

    # Nothing purged, and the file itself is queued for retry — not only the
    # directory, whose exact-match candidate set would skip its children.
    assert second.files_deleted == 0
    assert "pkg/mod.py" in second.failed_paths
    assert state.get_file_states(conn, pid)["pkg/mod.py"] == original_hash

    # The retry the caller performs from failed_paths picks the edit up.
    await run(paths=list(second.failed_paths))
    assert state.get_file_states(conn, pid)["pkg/mod.py"] != original_hash


# --- PR #24 review round 2 ----------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs only")
def test_discovery_skips_non_regular_files(tmp_path) -> None:
    """A FIFO must never be opened by discovery.

    `full.stat()` succeeds on a FIFO and reports st_size 0, so it passed the
    size gate and `_is_binary` then called open() on it — which blocks until
    some process opens the write end. Discovery hung inside its worker
    thread with nothing raising, so neither the OSError handler nor the new
    errors collector could see it: the run simply never finished.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "real.py").write_text("x = 1\n")
    os.mkfifo(root / "pipe.py")

    result: dict[str, list[str]] = {}
    worker = threading.Thread(
        target=lambda: result.update(files=discover_files(root)), daemon=True
    )
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "discovery blocked on the FIFO"
    assert result["files"] == ["real.py"]


def test_add_dirty_paths_rolls_back_on_failure(tmp_path, monkeypatch) -> None:
    """The union is a read-modify-write, now inside BEGIN IMMEDIATE. A
    failure between the two halves must leave the stored set untouched and
    no transaction open — otherwise a lost update silently reopens the H1
    gap PIPE-2 closes."""
    conn = state.connect(tmp_path / "s.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, tmp_path, "m")
    state.set_last_indexed_commit(conn, pid, "c0", dirty_paths=["a.py"])

    def boom(*_a, **_k):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(state.json, "dumps", boom)
    try:
        state.add_dirty_paths(conn, pid, ["b.py"])
    except RuntimeError:
        pass
    else:  # pragma: no cover — the monkeypatch must fire
        raise AssertionError("expected the injected failure")
    monkeypatch.undo()

    assert not conn.in_transaction
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py"})
    # And the function still works afterwards — no wedged connection.
    state.add_dirty_paths(conn, pid, ["b.py"])
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py", "b.py"})
    conn.close()


def test_add_dirty_paths_does_not_lose_a_concurrent_union(tmp_path, monkeypatch):
    """Two writers unioning into the same project row must both survive.

    The SELECT and the UPDATE used to be separate transactions, so two
    callers could both read the old set and the second write would drop the
    first one's paths — silently reopening the H1 gap PIPE-2 closes, since a
    path missing from the dirty set is never re-admitted by the next fast
    path. BEGIN IMMEDIATE makes the second caller wait for the first.
    """
    db = tmp_path / "s.sqlite"
    conn = state.connect(db)
    state.init_db(conn)
    pid = state.register_project(conn, tmp_path, "m")
    state.set_last_indexed_commit(conn, pid, "c0", dirty_paths=["a.py"])
    conn.close()

    real_dumps = state.json.dumps

    def slow_dumps(obj, *a, **k):
        # Widen the read-modify-write window so the interleaving is
        # deterministic instead of timing-dependent.
        time.sleep(0.25)
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(state.json, "dumps", slow_dumps)
    start = threading.Barrier(2)

    def worker(path: str) -> None:
        c = state.connect(db)
        try:
            start.wait(timeout=5)
            state.add_dirty_paths(c, pid, [path])
        finally:
            c.close()

    threads = [
        threading.Thread(target=worker, args=("b.py",)),
        threading.Thread(target=worker, args=("c.py",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    monkeypatch.undo()
    assert not any(t.is_alive() for t in threads)

    check = state.connect(db)
    try:
        assert state.get_dirty_paths(check, pid) == frozenset(
            {"a.py", "b.py", "c.py"}
        )
    finally:
        check.close()
