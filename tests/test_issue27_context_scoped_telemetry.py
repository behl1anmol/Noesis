"""Issue #27 / ADR-59 — the telemetry writer belongs to its ``AppContext``.

PR #24 left ``core/telemetry.py`` holding its writer in module globals
(``_worker``, ``_queue``, ``_worker_lock``) while ``close_runtime_context``
tore it down per ``AppContext``. Three docstring carve-outs existed only to
explain that mismatch: a one-live-context-per-process *invariant*, a
deliberately non-terminal ``close()``, and a queue owned by the worker
*generation* so a racing ``close()`` could not hand its sentinel to the wrong
worker. Instance state removes all three.

**On lesson 14** (a new test must fail against the pre-fix source for the
reason it was written). Two shapes of test live here, and they are not equally
provable:

* :func:`test_close_is_terminal_and_drops_later_rows` and
  :func:`test_a_new_writer_starts_open` invert **measurably**. Driven through
  the pre-fix module API — ``telemetry.close()`` then
  ``telemetry.record_query(...)`` — the row is written by a resurrected
  worker; here it is dropped. That difference was measured, not reasoned
  about.
* the isolation tests cannot be written against the pre-fix source at all:
  there is no per-context writer to name, so they would fail with
  ``AttributeError``, which proves nothing (lesson 14's own trap). The
  pre-fix loss they correspond to was reproduced separately through the old
  module API — a second context's queued rows discarded by the first
  context's teardown — and the measurement is recorded in the PR, not
  simulated here.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest
from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.telemetry import QueryTelemetry
from noesis.core.vectorstore import VectorStore
from noesis.runtime import AppContext, close_runtime_context

_FIELDS = dict(interface="rest", kind="search", project_id=None)


def _rows(db_path) -> int:
    fresh = sqlite3.connect(db_path)
    try:
        return fresh.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    finally:
        fresh.close()


def _context(tmp_path, name: str) -> AppContext:
    """A context on its own state DB — the shape a second live context takes."""
    conn = state.connect(tmp_path / name / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


@pytest.fixture()
def contexts(tmp_path):
    """Factory for contexts, all torn down at the end of the test."""
    made: list[AppContext] = []

    def _make(name: str) -> AppContext:
        ctx = _context(tmp_path, name)
        made.append(ctx)
        return ctx

    yield _make
    for ctx in made:
        ctx.telemetry.close(timeout=5.0)
        ctx.conn.close()


def _slow_log_query(monkeypatch, seconds: float = 0.05) -> None:
    """Make each write take long enough that a backlog is observable."""
    real = state.log_query

    def slow(conn, **fields):
        time.sleep(seconds)
        return real(conn, **fields)

    monkeypatch.setattr(state, "log_query", slow)


# --- the mechanism: one writer per context ------------------------------------


def test_each_context_gets_its_own_writer(contexts) -> None:
    """The structural fact everything else here rests on. Two contexts in one
    process have two writers, so neither one's lifecycle can reach the other's
    — an invariant replaced by a mechanism, which is what #27 asked for."""
    a, b = contexts("a"), contexts("b")
    assert a.telemetry is not b.telemetry
    assert a.telemetry._queue is not b.telemetry._queue


async def test_one_contexts_teardown_leaves_the_others_backlog_intact(
    contexts, monkeypatch, tmp_path
) -> None:
    """The failure the module docstring used to describe and accept.

    With a process-wide writer, tearing down context A retired the single
    worker and drained the single queue — so rows context B had already
    enqueued were discarded, and B's ``flush()`` then "honestly reported
    nothing pending". Per-context writers make B's backlog unreachable from
    A's teardown.
    """
    _slow_log_query(monkeypatch)
    a, b = contexts("a"), contexts("b")

    await a.telemetry.record_query(a.conn, **_FIELDS)
    for _ in range(8):
        await b.telemetry.record_query(b.conn, **_FIELDS)
    b_worker = b.telemetry._worker
    assert b_worker is not None and b_worker.is_alive()

    await asyncio.to_thread(a.telemetry.close)

    # A is down and its own row was drained on the way out.
    assert not a.telemetry._worker.is_alive()
    assert _rows(tmp_path / "a" / "state.sqlite") == 1
    # B never noticed: same worker, still running, backlog still its own.
    assert b.telemetry._worker is b_worker and b_worker.is_alive()

    await b.telemetry.flush(timeout=10.0)
    assert _rows(tmp_path / "b" / "state.sqlite") == 8


async def test_a_closed_context_does_not_stop_a_live_one_from_recording(
    contexts, tmp_path
) -> None:
    """The same isolation from the other side: after A's teardown, B is still
    a working writer — not left dropping rows because someone else closed."""
    a, b = contexts("a"), contexts("b")
    await asyncio.to_thread(a.telemetry.close)

    await b.telemetry.record_query(b.conn, **_FIELDS)
    await b.telemetry.flush(timeout=10.0)

    assert _rows(tmp_path / "b" / "state.sqlite") == 1
    assert _rows(tmp_path / "a" / "state.sqlite") == 0


async def test_close_runtime_context_closes_only_its_own_writer(
    contexts, tmp_path
) -> None:
    """End-to-end through the production teardown path, which is where the
    per-context/process mismatch actually lived (``runtime.py``)."""
    a, b = contexts("a"), contexts("b")
    await b.telemetry.record_query(b.conn, **_FIELDS)

    await close_runtime_context(a)

    assert b.telemetry._closed is False
    await b.telemetry.flush(timeout=10.0)
    assert _rows(tmp_path / "b" / "state.sqlite") == 1


# --- terminality, which instance state makes free ------------------------------


async def test_close_is_terminal_and_drops_later_rows(contexts, tmp_path) -> None:
    """No resurrection. Pre-fix, ``_ensure_worker`` rebuilt the generation
    whenever ``_worker`` was None, so a ``record_query`` arriving after
    teardown started a fresh worker with a fresh ``state.connect()`` handle
    that the closing context would never reap. Measured pre-fix through the
    module API: the row lands (2 rows). Here it is refused (1)."""
    a = contexts("a")
    db = tmp_path / "a" / "state.sqlite"
    await a.telemetry.record_query(a.conn, **_FIELDS)
    await asyncio.to_thread(a.telemetry.close)
    assert _rows(db) == 1

    closed_worker = a.telemetry._worker
    await a.telemetry.record_query(a.conn, **_FIELDS)
    await a.telemetry.flush(timeout=5.0)

    assert _rows(db) == 1, "a closed writer accepted a row"
    # And it refused without starting a second thread — the handle leak the
    # non-terminal close() could not rule out.
    assert a.telemetry._worker is closed_worker
    assert not closed_worker.is_alive()


async def test_a_new_writer_starts_open(contexts, tmp_path) -> None:
    """The other half of "terminality is free here": there is no global to
    reset, so a second writer in the same process is simply open. That is the
    whole of the round-7 objection to a terminal flag, dissolved."""
    a = contexts("a")
    db = tmp_path / "a" / "state.sqlite"
    await asyncio.to_thread(a.telemetry.close)

    fresh = QueryTelemetry()
    try:
        await fresh.record_query(a.conn, **_FIELDS)
        await fresh.flush(timeout=5.0)
    finally:
        await asyncio.to_thread(fresh.close)
    assert _rows(db) == 1


async def test_rows_queued_before_close_are_still_written(
    contexts, monkeypatch, tmp_path
) -> None:
    """Over-correction guard. Terminal must mean "refuses NEW rows", never
    "discards the ones already accepted" — close() still drains its backlog,
    which is the ADR-52 contract and the reason it is called at teardown at
    all."""
    _slow_log_query(monkeypatch)
    a = contexts("a")
    for _ in range(5):
        await a.telemetry.record_query(a.conn, **_FIELDS)

    await asyncio.to_thread(a.telemetry.close)

    assert _rows(tmp_path / "a" / "state.sqlite") == 5


async def test_close_is_idempotent(contexts) -> None:
    """Called twice — a double teardown, or a fixture closing after
    ``close_runtime_context`` already did — the second call returns without
    raising and without disturbing anything."""
    a = contexts("a")
    await a.telemetry.record_query(a.conn, **_FIELDS)
    await asyncio.to_thread(a.telemetry.close)
    worker = a.telemetry._worker

    await asyncio.to_thread(a.telemetry.close)

    assert a.telemetry._worker is worker
    assert a.telemetry._closed is True


async def test_flush_after_close_returns_promptly(contexts) -> None:
    """A barrier put after teardown would sit in a queue no worker will ever
    serve. ``flush`` must see the terminal state and return, not spend its
    whole timeout."""
    a = contexts("a")
    await a.telemetry.record_query(a.conn, **_FIELDS)
    await asyncio.to_thread(a.telemetry.close)

    started = time.monotonic()
    await a.telemetry.flush(timeout=5.0)
    assert time.monotonic() - started < 1.0


# --- the in-memory shape (test-only) -------------------------------------------


async def test_in_memory_db_writes_inline_then_stops_at_close() -> None:
    """A second connection cannot reach an in-memory DB, so that shape writes
    on the caller's own conn. "Closed" has to mean the same thing on both
    shapes, or a teardown would silence one and not the other."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    state.init_db(conn)
    tel = QueryTelemetry()
    try:
        await tel.record_query(conn, **_FIELDS)
        assert conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 1
        # No worker was needed: the inline path never touches the queue.
        assert tel._worker is None

        await asyncio.to_thread(tel.close)
        await tel.record_query(conn, **_FIELDS)
        assert conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 1
    finally:
        tel.close(timeout=5.0)
        conn.close()


# --- the query path is still never charged for a write -------------------------


async def test_a_closed_writer_still_never_raises_into_the_query_path(
    contexts,
) -> None:
    """Telemetry's top contract outranks terminality: refusing a row is a
    DEBUG log and a return, never an exception. ``LocalSTEmbedder._submit``
    raises after close because a dropped embedding is a correctness loss; a
    dropped telemetry row is this module's documented behaviour."""
    a = contexts("a")
    await asyncio.to_thread(a.telemetry.close)
    # Would raise if the closed check escalated instead of dropping.
    await a.telemetry.record_query(a.conn, **_FIELDS)


async def test_recording_does_not_block_on_a_stalled_writer(
    contexts, monkeypatch
) -> None:
    """Over-correction guard for the whole refactor: the per-instance lock is
    taken only to test a flag and put on a bounded queue, so the query path
    still pays nothing while the writer is stuck on a foreign lock (MCP-3)."""
    released = threading.Event()
    real = state.log_query

    def wedged(conn, **fields):
        released.wait(timeout=30.0)
        return real(conn, **fields)

    monkeypatch.setattr(state, "log_query", wedged)
    a = contexts("a")
    try:
        started = time.monotonic()
        for _ in range(5):
            await a.telemetry.record_query(a.conn, **_FIELDS)
        elapsed = time.monotonic() - started
    finally:
        released.set()
    assert elapsed < 0.1, f"caller waited {elapsed:.3f}s behind a stalled writer"
