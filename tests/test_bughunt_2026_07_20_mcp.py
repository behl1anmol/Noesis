"""Bug-hunt 2026-07-20 (MCP stress test) — regression pins for MCP-2/MCP-3.

MCP-2: ``jobs.index_status`` ran wholesale inside ``asyncio.to_thread`` at
both call sites, so its quick ``state.*`` reads touched the shared sqlite
connection from a worker thread while the event loop writes the same conn
(telemetry, run bookkeeping) — a read could interleave inside another op's
open BEGIN IMMEDIATE transaction. Now ``index_status`` is async: state reads
stay on the loop; only the synchronous Qdrant count is offloaded.

MCP-3: ``telemetry.record_query`` wrote through the shared ctx.conn
(busy_timeout=5000) synchronously on the event loop. In the dual-transport
deployment (HTTP + stdio MCP sharing one DB file) another process's writer
lock stalled every in-flight request up to 5s per search. Now telemetry
owns a dedicated best-effort connection (busy_timeout=250) and the INSERT
runs in a worker thread; failures are still swallowed.

Harness mirrors tests/test_m8_dashboard.py: FakeEmbedder + in-memory Qdrant
+ a real state.sqlite file (the sqlite file is the point — locking needs it).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import jobs, state, telemetry
from noesis.core.embedder import FakeEmbedder
from noesis.core.telemetry import QueryTelemetry
from noesis.core.vectorstore import VectorStore


def _rows(db_path) -> int:
    """query_log count read through a connection nobody in the test owns."""
    fresh = sqlite3.connect(db_path)
    try:
        return fresh.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    finally:
        fresh.close()


@pytest.fixture()
def project_dir(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "auth.py").write_text("def validate_token(token):\n    return token\n")
    return src


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "state.sqlite"


@pytest.fixture()
def ctx(db_path):
    conn = state.connect(db_path)
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    context = AppContext(conn=conn, store=store, embedder=embedder)
    yield context
    # Each context owns its telemetry writer (ADR-59); production closes it in
    # close_runtime_context, and a hand-built context has no such hook — so
    # close it here rather than leaving a thread and a DB handle behind.
    context.telemetry.close(timeout=5.0)


# --- MCP-2: index_status thread placement ------------------------------------


async def test_index_status_state_reads_on_loop_qdrant_count_off_loop(
    ctx, project_dir, monkeypatch
):
    """State reads stay on the event-loop thread (shared conn is loop-owned);
    only the Qdrant round-trip runs in a worker thread."""
    pid = state.register_project(ctx.conn, project_dir, "fake")
    threads: dict[str, threading.Thread] = {}

    real_expected = state.expected_chunk_total
    real_latest = state.get_latest_run
    real_count = ctx.store.count_project_points

    def spy_expected(conn, project_id):
        threads["expected_chunk_total"] = threading.current_thread()
        return real_expected(conn, project_id)

    def spy_latest(conn, project_id):
        threads["get_latest_run"] = threading.current_thread()
        return real_latest(conn, project_id)

    def spy_count(project_id):
        threads["count_project_points"] = threading.current_thread()
        return real_count(project_id)

    monkeypatch.setattr(jobs.state, "expected_chunk_total", spy_expected)
    monkeypatch.setattr(jobs.state, "get_latest_run", spy_latest)
    monkeypatch.setattr(ctx.store, "count_project_points", spy_count)

    loop_thread = threading.current_thread()
    status = await jobs.index_status(ctx, pid)

    assert status["status"] == "never_indexed"  # shape untouched
    assert threads["expected_chunk_total"] is loop_thread
    assert threads["get_latest_run"] is loop_thread
    assert threads["count_project_points"] is not loop_thread


# --- MCP-3: telemetry off the loop, dedicated connection ----------------------


async def test_record_query_lands_via_dedicated_conn_off_loop(
    ctx, db_path, monkeypatch
):
    """The INSERT runs in a worker thread on a connection that is not the
    shared ctx.conn, and the row is visible to a fresh read connection."""
    seen: dict[str, object] = {}
    real_log = state.log_query

    def spy_log(conn, **kwargs):
        seen["thread"] = threading.current_thread()
        seen["conn"] = conn
        return real_log(conn, **kwargs)

    monkeypatch.setattr(telemetry.state, "log_query", spy_log)

    await ctx.telemetry.record_query(
        ctx.conn,
        interface="rest",
        kind="search",
        project_id=None,
        channel="hybrid",
        latency_ms=1.0,
        result_count=0,
    )
    # The write is asynchronous now (PR #24 review): the row is queued, not
    # written, when record_query returns. Barrier before reading it back.
    await ctx.telemetry.flush()

    assert seen["thread"] is not threading.current_thread()
    assert seen["conn"] is not ctx.conn
    fresh = sqlite3.connect(db_path)
    try:
        count = fresh.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    finally:
        fresh.close()
    assert count == 1


async def test_record_query_under_foreign_writer_lock_neither_raises_nor_blocks(
    ctx, db_path
):
    """A second connection holding the writer lock (the dual-transport
    scenario) must cost the caller at most the dedicated conn's short
    busy_timeout — never ctx.conn's 5s — and must not raise."""
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        t0 = time.perf_counter()
        await ctx.telemetry.record_query(
            ctx.conn, interface="mcp", kind="search", project_id=None
        )
        elapsed = time.perf_counter() - t0
    finally:
        blocker.rollback()
        blocker.close()
    assert elapsed < 1.0, f"telemetry blocked the caller for {elapsed:.2f}s"


def test_search_request_unaffected_by_foreign_writer_lock(db_path, ctx, project_dir):
    """Route-level pin (surface unchanged pre/post fix): with the writer lock
    held by another connection, POST /search still succeeds and returns well
    under ctx.conn's 5s busy_timeout. Pre-fix the telemetry INSERT on
    ctx.conn stalled the loop ~5s here."""
    with TestClient(create_app(ctx=ctx)) as tc:
        pid = state.register_project(ctx.conn, project_dir, "fake")
        blocker = sqlite3.connect(db_path)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            t0 = time.perf_counter()
            resp = tc.post("/search", json={"query": "token", "project_id": pid})
            elapsed = time.perf_counter() - t0
        finally:
            blocker.rollback()
            blocker.close()
        assert resp.status_code == 200
        assert elapsed < 2.0, f"search stalled {elapsed:.2f}s behind writer lock"


# --- PR #24 review: the awaited write still charged the query -----------------


async def test_record_query_does_not_wait_for_a_blocked_writer(ctx, db_path):
    """The caller must not pay the write's cost at all.

    Pre-review the INSERT was awaited inside the request path: off the loop,
    but still before the response. Under the exact contention this fix
    targets that cost the request up to the dedicated conn's 250ms
    busy_timeout, and the process-wide write lock serialized concurrent
    searches behind each other. Now the row is queued and the caller returns
    immediately, so the elapsed time is independent of the writer's fate.
    """
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        t0 = time.perf_counter()
        for _ in range(5):
            await ctx.telemetry.record_query(
                ctx.conn, interface="rest", kind="search", project_id=None
            )
        elapsed = time.perf_counter() - t0
    finally:
        blocker.rollback()
        blocker.close()
    # Five writes, each of which would have cost up to 250ms serialized
    # (~1.25s) when awaited. Queueing is microseconds; 100ms is a generous
    # ceiling that still fails loudly if the write moves back on-path.
    assert elapsed < 0.1, f"caller waited {elapsed:.3f}s for telemetry writes"


async def test_telemetry_close_drains_and_releases_the_db_handle(ctx, db_path):
    """close() must flush what is queued and leave no handle behind.

    The writer holds its own connection to the state DB. Without a close
    path it outlives ctx.conn for the process lifetime, which blocks
    temp-directory cleanup on Windows/WSL and survives a DB-file removal.
    """
    await ctx.telemetry.record_query(
        ctx.conn, interface="rest", kind="search", project_id=None
    )
    await asyncio.to_thread(ctx.telemetry.close)

    # Queued row was drained, not dropped, and the writer is stopped.
    assert not ctx.telemetry._worker.is_alive()
    assert _rows(db_path) == 1

    # A later record_query is DROPPED, not written on a resurrected worker.
    #
    # This assertion is inverted from the one it replaces (ADR-59, issue #27).
    # While the writer lived in module globals, close() could not be terminal:
    # a terminal flag there would have been a global every test building a
    # second context had to reset, so close() only retired the worker and the
    # next record_query started a fresh one — including one arriving after
    # teardown had decided to close, which opened a state.connect() handle the
    # closing context would never reap. Instance state makes terminality free,
    # because a second context simply builds a second writer.
    await ctx.telemetry.record_query(
        ctx.conn, interface="rest", kind="search", project_id=None
    )
    await ctx.telemetry.flush()
    assert _rows(db_path) == 1, "a closed writer accepted a row"

    # And the other half of "free": a NEW writer starts open, with no global
    # to reset between the two.
    fresh_writer = QueryTelemetry()
    try:
        await fresh_writer.record_query(
            ctx.conn, interface="rest", kind="search", project_id=None
        )
        await fresh_writer.flush()
    finally:
        await asyncio.to_thread(fresh_writer.close)
    assert _rows(db_path) == 2


async def test_index_status_tolerates_a_commit_during_the_qdrant_count(
    ctx, project_dir, monkeypatch
):
    """Moving the state reads onto the loop put a scheduling point between
    the two halves of the drift comparison, so a run committing chunks mid
    call made a healthy index report drift. Only a total that agrees with
    itself across the window counts as evidence."""
    pid = state.register_project(ctx.conn, project_dir, "fake")
    totals = iter([10, 20])
    monkeypatch.setattr(state, "expected_chunk_total", lambda *_a, **_k: next(totals))
    monkeypatch.setattr(ctx.store, "count_project_points", lambda *_a, **_k: 20)

    status = await jobs.index_status(ctx, pid)

    # The expected total moved from 10 to 20 across the count: that is an
    # in-flight run, not drift. The fresher figure is reported.
    assert status["drift"] is False
    assert status["expected_chunks"] == 20
    assert status["vector_count"] == 20


async def test_index_status_still_reports_genuine_drift(ctx, project_dir, monkeypatch):
    """Guard against over-correcting: a total that is stable across the
    window and disagrees with Qdrant is real drift (ADR-49) and must still
    be surfaced."""
    pid = state.register_project(ctx.conn, project_dir, "fake")
    monkeypatch.setattr(state, "expected_chunk_total", lambda *_a, **_k: 42)
    monkeypatch.setattr(ctx.store, "count_project_points", lambda *_a, **_k: 0)

    status = await jobs.index_status(ctx, pid)
    assert status["drift"] is True


# --- PR #24 review round 3 ----------------------------------------------------


async def test_drift_needs_two_agreeing_reads(ctx, project_dir, monkeypatch):
    """The in-flight shape the first guard missed.

    `upsert_chunks` lands points before `upsert_file` commits the row, so
    mid-run the Qdrant count sits ABOVE the expected total and BOTH readings
    differ from it. Asking only whether either reading equals the count gave
    that case two chances to coincide and reported drift on a healthy index;
    the readings have to agree with each other first.
    """
    pid = state.register_project(ctx.conn, project_dir, "fake")
    totals = iter([400, 450])
    monkeypatch.setattr(state, "expected_chunk_total", lambda *_a, **_k: next(totals))
    monkeypatch.setattr(ctx.store, "count_project_points", lambda *_a, **_k: 470)

    status = await jobs.index_status(ctx, pid)

    assert status["drift"] is False
    assert status["expected_chunks"] == 450
    assert status["vector_count"] == 470


async def test_drift_still_reported_when_the_total_holds_still(
    ctx, project_dir, monkeypatch
):
    """Over-correction guard: a stable expectation that disagrees with the
    store is exactly the ADR-49 signal and must survive."""
    pid = state.register_project(ctx.conn, project_dir, "fake")
    monkeypatch.setattr(state, "expected_chunk_total", lambda *_a, **_k: 400)
    monkeypatch.setattr(ctx.store, "count_project_points", lambda *_a, **_k: 0)

    status = await jobs.index_status(ctx, pid)
    assert status["drift"] is True


async def test_close_leaves_no_row_for_another_writer_to_inherit(ctx, tmp_path):
    """Replaces two round-3 tests that pinned the *generation* mechanism.

    Those tests modelled a `record_query` racing `close()` onto a shared
    module-global queue: the racing row could sit behind a consumed sentinel
    and be picked up by the NEXT runtime's writer, carrying a `db_path` from
    the torn-down one — which `state.connect` would `mkdir -p` back into
    existence as an empty, schema-less DB. Per-generation queues made that
    unrepresentable; per-`AppContext` writers (ADR-59) make it unrepresentable
    twice over, since there is no shared queue left to race onto and
    `_submit` refuses once `_closed` is set under the lock `close()` holds.

    The two guarantees those tests bought are what this asserts: `close()`
    leaves an empty queue and no inheritable row, and a `flush()` waiter is
    never stranded on a barrier no worker will reach.
    """
    gone = tmp_path / "torn-down" / "state.sqlite"
    await ctx.telemetry.record_query(
        ctx.conn, interface="rest", kind="search", project_id=None
    )
    work = ctx.telemetry._queue
    await asyncio.to_thread(ctx.telemetry.close)

    assert work.qsize() == 0
    # The row this writer accepted went to ITS db, and nothing it held names
    # a torn-down path that a later state.connect() could recreate.
    assert not gone.parent.exists(), "a torn-down DB directory was recreated"

    # A record_query landing after the sentinel is refused outright, so it can
    # neither be written nor left behind for anyone to inherit.
    await ctx.telemetry.record_query(
        ctx.conn, interface="rest", kind="search", project_id=None
    )
    assert work.qsize() == 0

    # A waiter blocked on a barrier must not be stranded by the drain. Put it
    # on a live writer, then close underneath it.
    other = QueryTelemetry()
    barrier = telemetry._Barrier()
    assert other._submit((str(gone), dict(interface="rest", kind="search")))
    other._queue.put_nowait(barrier)
    await asyncio.to_thread(other.close)
    assert barrier.event.is_set()
    assert other._queue.qsize() == 0
