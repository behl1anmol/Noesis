"""Regression tests for the PR #24 round-6 review findings.

Three defects, all in what the operator is told rather than in what the index
does — the deletion suppression itself was correct in every case:

1. When discovery had *already* diagnosed an unreadable root, the empty-scan
   guard synthesized a second `<root>` row on top of it. `record_file_errors`
   is INSERT OR REPLACE keyed on (run_id, path), so the generic message
   overwrote the real errno, `files_failed` counted one failure twice, and the
   run error named two locations where one had failed.
2. With `force=True` the same shape logged "accepting as deletion (forced by
   caller)" while the deletion block correctly refused to delete anything. The
   same line reached through the file-level door credited the filters with
   entries that had errored — hashdiff carries those forward, so they are never
   deleted either.
3. `flush()` dropped its barrier on a full queue and returned immediately,
   reporting "flushed" with the backlog at its maximum — the one moment a
   caller most needs a real barrier.

Fixing 1 also narrows a case that was over-suppressed: an *attributable*
subtree failure on a walk that saw no files no longer widens to the whole run.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import pytest

from noesis.core import discovery, state
from noesis.core.discovery import DiscoveryConfig
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.telemetry import QueryTelemetry

from tests.test_gitfast import make_env


def _fail_scandir_on_root(monkeypatch, root: Path) -> None:
    """The walk root itself is unreadable — a dead mount, a renamed parent.
    os.walk routes the root's own failure through the same `onerror` hook, so
    discovery records it as `<root>`."""
    resolved = root.resolve()
    real_scandir = os.scandir

    def broken(path=".", *args, **kwargs):
        if Path(path) == resolved:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", broken)


def _fail_scandir_on(monkeypatch, dirname: str) -> None:
    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == dirname:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)


# --- finding 1: the diagnosis discovery already made ---------------------------


async def test_unreadable_root_keeps_the_errno_discovery_recorded(
    tmp_path, monkeypatch
) -> None:
    """The EACCES is the only thing in the run that tells an operator *why* the
    root could not be read. The guard's own message says nothing actionable, and
    INSERT OR REPLACE let it win."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id

    _fail_scandir_on_root(monkeypatch, root)
    second = await run()
    monkeypatch.undo()

    rows = state.list_file_errors(conn, second.run_id)
    assert [r["path"] for r in rows] == ["<root>"]
    assert "Permission denied" in rows[0]["error"]

    run_row = state.get_latest_run(conn, pid)
    # One location failed, so the count and the wording must both say one.
    assert run_row["files_failed"] == 1
    assert run_row["error"] == (
        "discovery failed for 1 location(s); no files could be verified"
    )
    # Unchanged, and the whole point of the guard: nothing was deleted.
    assert second.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py"}


async def test_empty_readable_root_still_synthesizes_the_sentinel(tmp_path) -> None:
    """Over-correction guard. A root that exists and is simply empty raises
    nothing at all — there is no error to defer to, so the emptiness itself has
    to be the signal and the sentinel must still be written."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    (root / "a.py").unlink()

    second = await run()

    rows = state.list_file_errors(conn, second.run_id)
    assert [r["path"] for r in rows] == ["<root>"]
    assert "treating as an unreadable root" in rows[0]["error"]
    assert second.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py"}


async def test_unwalkable_subtree_no_longer_voids_the_rest_of_the_walk(
    tmp_path, monkeypatch
) -> None:
    """Consequence of deferring to the recorded error: when the failure IS
    attributable, ADR-54 scoping applies even though the walk returned nothing.
    The synthetic `<root>` used to widen this to project-wide suppression, so a
    deletion outside the failed subtree stayed unprovable forever."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "other").mkdir()
    (root / "pkg" / "mod.py").write_text("z = 3\n")
    (root / "other" / "gone.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert set(state.get_file_states(conn, pid)) == {"pkg/mod.py", "other/gone.py"}

    # Every remaining file lives under the failing subtree, so the walk
    # enumerates no files at all — the shape that used to trip the guard.
    (root / "other" / "gone.py").unlink()
    _fail_scandir_on(monkeypatch, "pkg")
    second = await run()
    monkeypatch.undo()

    assert second.files_deleted == 1
    assert set(state.get_file_states(conn, pid)) == {"pkg/mod.py"}
    assert store.per_file_point_counts(pid).get("other/gone.py", 0) == 0
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0
    # Still a failed run with the subtree's own error, not a synthesized one.
    assert [r["path"] for r in state.list_file_errors(conn, second.run_id)] == ["pkg"]
    assert state.get_latest_run(conn, pid)["status"] == "failed"


# --- finding 2: a log line may only name a purge that can follow ---------------


async def test_force_does_not_log_a_purge_it_will_not_perform(
    tmp_path, monkeypatch, caplog
) -> None:
    """`force` skips the guard, so an unreadable root reached the "accepting as
    deletion" branch. The deletion block then suppressed everything — correctly,
    since `force` never relaxes a directory error — leaving only the log wrong."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(force=False):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            force=force,
        )

    first = await run()
    pid = first.project_id

    _fail_scandir_on_root(monkeypatch, root)
    with caplog.at_level(logging.WARNING, logger="noesis.core.indexer"):
        second = await run(force=True)
    monkeypatch.undo()

    # Matched on the bare verb, not on this round's phrasing: the invariant is
    # that NO line claims the scan was accepted, however it comes to be worded.
    assert not [rec for rec in caplog.records if "accepting" in rec.getMessage()]
    assert second.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py"}


async def test_unreadable_files_are_not_credited_to_the_filters(
    tmp_path, monkeypatch, caplog
) -> None:
    """The same overclaim reached through the file-level door. `entries_seen`
    counts every enumerated entry, including ones that then failed to be
    screened — so a tree whose files all errored logged "all filtered out —
    root is readable" and "accepting as deletion", while hashdiff carried every
    one of them forward and nothing was deleted."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id

    real = discovery._is_binary

    def flaky(path):
        if path.name == "a.py":
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(discovery, "_is_binary", flaky)
    with caplog.at_level(logging.WARNING, logger="noesis.core.indexer"):
        second = await run()
    monkeypatch.undo()

    accepted = [
        rec.getMessage()
        for rec in caplog.records
        if "accepting the empty scan" in rec.getMessage()
    ]
    assert len(accepted) == 1
    assert "(0 filtered out, 1 unreadable)" in accepted[0]
    # And the line's own claim about what follows holds.
    assert second.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py"}


async def test_a_real_purge_is_still_logged_with_its_escape(tmp_path, caplog) -> None:
    """Over-correction guard: the branch exists so an operator reading the log
    after the fact can see which escape emptied the project. Where a purge does
    follow, it must still say so."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(config=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            discovery_config=config,
        )

    first = await run()
    pid = first.project_id

    with caplog.at_level(logging.WARNING, logger="noesis.core.indexer"):
        second = await run(DiscoveryConfig(include_languages=frozenset({"rust"})))

    accepted = [
        rec.getMessage()
        for rec in caplog.records
        if "accepting the empty scan" in rec.getMessage()
    ]
    assert len(accepted) == 1
    assert "(1 filtered out, 0 unreadable)" in accepted[0]
    assert second.files_deleted == 1
    assert state.get_file_states(conn, pid) == {}


# --- finding 3: telemetry flush must not drop its barrier ----------------------


@pytest.fixture()
def writers():
    """Factory for writers these tests deliberately stall.

    Per-`AppContext` writers (ADR-59, issue #27) replaced the module globals
    this section used to drive, so a stalled writer is now confined to the test
    that made it — but it still owns a thread and a DB handle, so close them.
    """
    made: list[QueryTelemetry] = []

    def _make(**kwargs) -> QueryTelemetry:
        tel = QueryTelemetry(**kwargs)
        made.append(tel)
        return tel

    yield _make
    for tel in made:
        tel.close(timeout=5.0)


def _db(tmp_path) -> str:
    path = tmp_path / "state.sqlite"
    conn = state.connect(path)
    state.init_db(conn)
    conn.close()
    return str(path)


_FIELDS = dict(
    interface="rest",
    kind="search",
    project_id=None,
    channel=None,
    reranked=None,
    latency_ms=None,
    result_count=None,
)


def _fill_the_queue(tel: QueryTelemetry, db_path: str) -> int:
    queued = 0
    while tel._submit((db_path, dict(_FIELDS))):
        queued += 1
    return queued


async def test_flush_does_not_report_success_with_a_full_backlog(
    tmp_path, monkeypatch, writers
) -> None:
    """Dropping ROWS on overflow is the documented contract; dropping the
    BARRIER is a different promise. Before the fix this returned in 0.000s with
    the queue at its bound and nothing written."""
    db_path = _db(tmp_path)
    real_log_query = state.log_query

    def slow(conn, **fields):
        time.sleep(0.05)
        return real_log_query(conn, **fields)

    monkeypatch.setattr(state, "log_query", slow)
    # A small bound is a constructor argument now, not a monkeypatched module
    # constant — the writer under test is this test's alone (ADR-59).
    tel = writers(queue_max=8)

    queued = _fill_the_queue(tel, db_path)
    # The worker pulls the first row as soon as it starts, so the queue admits
    # its bound plus whatever the writer has already taken off it.
    assert queued >= 8

    await tel.flush(timeout=10.0)

    conn = state.connect(db_path)
    try:
        written = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    finally:
        conn.close()
    # Everything enqueued before the flush sits ahead of the barrier, so
    # reaching the barrier means all of it is on disk.
    assert written == queued


async def test_flush_gives_up_after_its_timeout_on_a_wedged_writer(
    tmp_path, monkeypatch, writers
) -> None:
    """The other half: a barrier that can never be queued must cost the caller
    its stated budget and no more. Returning instantly is the bug; blocking
    forever would be a worse one."""
    db_path = _db(tmp_path)
    released = threading.Event()
    real_log_query = state.log_query

    def wedged(conn, **fields):
        released.wait(timeout=30.0)
        return real_log_query(conn, **fields)

    monkeypatch.setattr(state, "log_query", wedged)
    tel = writers(queue_max=4)
    _fill_the_queue(tel, db_path)

    try:
        started = time.monotonic()
        await tel.flush(timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        released.set()

    assert 0.3 <= elapsed < 3.0


async def test_flush_still_returns_promptly_when_nothing_is_backed_up(
    tmp_path, writers
) -> None:
    """Over-correction guard: the retry loop must not add latency to the normal
    path, which is every call the dashboard tests make."""
    db_path = _db(tmp_path)
    tel = writers()
    assert tel._submit((db_path, dict(_FIELDS)))

    started = time.monotonic()
    await tel.flush(timeout=5.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    conn = state.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 1
    finally:
        conn.close()
