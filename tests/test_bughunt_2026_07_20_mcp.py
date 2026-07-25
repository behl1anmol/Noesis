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

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import jobs, state, telemetry
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore


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
    return AppContext(conn=conn, store=store, embedder=embedder)


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

    await telemetry.record_query(
        ctx.conn,
        interface="rest",
        kind="search",
        project_id=None,
        channel="hybrid",
        latency_ms=1.0,
        result_count=0,
    )

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
        await telemetry.record_query(
            ctx.conn, interface="mcp", kind="search", project_id=None
        )
        elapsed = time.perf_counter() - t0
    finally:
        blocker.rollback()
        blocker.close()
    assert elapsed < 1.0, f"telemetry blocked the caller for {elapsed:.2f}s"


def test_search_request_unaffected_by_foreign_writer_lock(
    db_path, ctx, project_dir
):
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
