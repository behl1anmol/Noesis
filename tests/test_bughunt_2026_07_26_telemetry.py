"""Regression tests for the PR #24 round-5 telemetry findings.

Both are failures of `close()`'s own contract:

* the unconditional drain after a timed-out join swallowed the shutdown
  sentinel, leaving the abandoned writer blocked on `get()` forever with its
  `state.connect()` handles open — the exact leak `close()` exists to prevent;
* `flush()` read the live writer under the lock, released it, and only
  then put its barrier, so a `close()` landing in that window left the barrier
  in a queue nobody would serve and `flush` waited out its whole timeout.

Ported to per-`AppContext` writers (ADR-59, issue #27): the state these tests
drive is now instance state on :class:`QueryTelemetry`, so each test builds its
own writer instead of reaching for module globals. Both findings, and both
fixes, are unchanged — only their owner is.
"""

from __future__ import annotations

import threading
import time

import pytest

from noesis.core import state, telemetry
from noesis.core.telemetry import QueryTelemetry


@pytest.fixture()
def writers():
    """Factory for writers this module is allowed to leave in odd states.

    These tests deliberately abandon a worker mid-backlog, so every writer is
    closed on the way out — a per-instance version of what the module-global
    `_stop_the_writer` fixture did before ADR-59. `close()` is idempotent, so
    a test that already closed its own writer costs nothing here.
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


def test_abandoned_writer_keeps_its_sentinel_and_stops_itself(
    tmp_path, monkeypatch, writers
) -> None:
    """A backlog that outlives the join is the writer's to finish. Draining it
    from `close()` removes the sentinel the writer has not reached yet, and the
    thread then blocks on `get()` forever holding its connections — reproduced
    at 200 rows / 50 ms per row against a 1 s join before the fix."""
    db_path = _db(tmp_path)
    real_log_query = state.log_query

    def slow(conn, **fields):
        time.sleep(0.05)
        return real_log_query(conn, **fields)

    monkeypatch.setattr(state, "log_query", slow)

    tel = writers()
    for _ in range(20):
        assert tel._submit((db_path, dict(_FIELDS)))
    worker = tel._worker
    assert worker is not None

    tel.close(timeout=0.1)

    # The join expired: the writer is still working through the backlog.
    assert worker.is_alive()
    # And it can still stop itself, because the sentinel is still queued.
    worker.join(timeout=10.0)
    assert not worker.is_alive()


def test_close_still_writes_and_drains_when_the_writer_stops_in_time(
    tmp_path, writers
) -> None:
    """Over-correction guard: the early return is only for the timed-out case.
    A writer that exits within the join must still have written everything
    ahead of the sentinel and left its queue empty."""
    db_path = _db(tmp_path)
    tel = writers()
    for _ in range(3):
        assert tel._submit((db_path, dict(_FIELDS)))
    worker = tel._worker
    assert worker is not None

    tel.close(timeout=5.0)

    assert not worker.is_alive()
    assert tel._queue.qsize() == 0
    conn = state.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 3
    finally:
        conn.close()


async def test_flush_is_not_stranded_by_a_racing_close(tmp_path, writers) -> None:
    """The window is between reading the live writer and putting the barrier.
    Constructing the barrier *before* taking the lock is what closes it: a
    `close()` that lands first is then seen by the lock-protected read, and
    `flush` returns instead of waiting out its timeout."""
    db_path = _db(tmp_path)
    tel = writers()
    assert tel._submit((db_path, dict(_FIELDS)))
    assert tel._worker is not None

    original_init = telemetry._Barrier.__init__
    raced = threading.Event()

    def racing_init(self) -> None:
        original_init(self)
        if not raced.is_set():
            raced.set()
            # Exactly the interleaving the old code allowed: the writer is
            # retired and drained between `flush`'s read and its put.
            tel.close(timeout=5.0)

    telemetry._Barrier.__init__ = racing_init
    try:
        started = time.monotonic()
        await tel.flush(timeout=5.0)
        elapsed = time.monotonic() - started
    finally:
        telemetry._Barrier.__init__ = original_init

    assert raced.is_set()
    # Waiting out the 5 s timeout is the bug; returning promptly is the fix.
    assert elapsed < 2.0


async def test_mcp_reindex_does_not_expose_force() -> None:
    """ADR-55 makes the empty-root assertion an operator call, and the caller
    of an MCP tool is an agent. A `force` parameter in this schema is a
    one-call project purge for an LLM that read a zero-file status — the
    outcome the guard exists to prevent. REST keeps it; this must not."""
    from fastmcp import Client

    from noesis.mcp.server import build_mcp

    # The schema is static; no context is needed to read it.
    async with Client(build_mcp(lambda: None)) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools["reindex"].inputSchema["properties"]) == {"project_id"}


def test_rest_reindex_keeps_force_as_a_query_param() -> None:
    """The other half of the same decision: dropping `force` from MCP is only
    defensible if the operator path still exists."""
    from fastapi import FastAPI

    from noesis.api.routes import router

    app = FastAPI()
    app.include_router(router)
    spec = app.openapi()["paths"]["/projects/{project_id}/reindex"]["post"]
    force = next(p for p in spec["parameters"] if p["name"] == "force")
    assert force["in"] == "query"
