"""Usage telemetry — metadata-only query logging (ADR-40).

Feeds the dashboard's usage page. Records *that* a query happened and how
it performed (interface, kind, channel, latency, result count) — never the
query text: queries routinely quote proprietary code, and a local DB is
still a file that gets backed up, synced, and pasted into bug reports
(ADR-25 spirit). Logging is fire-and-forget: a telemetry failure must never
fail — or delay — the search that triggered it.

Writes are handed to a single dedicated writer thread over a bounded queue
and never touch the caller's ``ctx.conn``. Two reasons, both from the
dual-transport deployment (HTTP server + stdio MCP sharing one DB file):

* ``ctx.conn`` has ``busy_timeout=5000``, so an INSERT contending with the
  other process's index-run writes would stall the event loop up to 5s.
* Even off the loop, an *awaited* write keeps the response waiting on that
  contention. Enqueueing returns immediately, so the query path pays
  nothing beyond a queue put.

The writer owns its own connections (one per DB path, ``busy_timeout=250``)
and is the only thread that touches them, so no lock is needed. On a full
queue or a busy DB the row is dropped: telemetry may lose a row, never slow
a query. :func:`close` drains and stops the writer at teardown.

Consequence worth knowing: the usage page is eventually consistent. A query
is normally readable within microseconds, but it is not guaranteed to be
there the instant the response returns — under the contention above it can
lag up to 250ms or be dropped entirely. Anything that needs read-your-writes
(tests, mainly) must await :func:`flush`.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sqlite3
import threading
from typing import Any

from . import state

logger = logging.getLogger(__name__)

# Bounded so a writer stalled on a foreign lock cannot grow memory without
# limit; a full queue drops the row, which is the documented contract. Sized
# well above any realistic burst of concurrent searches.
_QUEUE_MAX = 1000

_SHUTDOWN = object()


class _Barrier:
    """Flush marker. The queue is FIFO, so once the writer reaches this the
    rows enqueued before it are already written."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()

_queue: queue.Queue[Any] = queue.Queue(maxsize=_QUEUE_MAX)
# Guards worker startup only — never held across a write, so writes never
# serialize against each other or against the query path.
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


def _writer_loop() -> None:
    """Sole owner of the telemetry connections. One handle per DB path,
    opened lazily; because only this thread touches them, the writes need
    no lock and concurrent queries never queue behind one another."""
    conns: dict[str, sqlite3.Connection] = {}
    try:
        while True:
            item = _queue.get()
            if item is _SHUTDOWN:
                return
            if isinstance(item, _Barrier):
                item.event.set()
                continue
            db_path, fields = item
            try:
                conn = conns.get(db_path)
                if conn is None:
                    # Same sanctioned constructor as the shared conn
                    # (state.connect is the one place sqlite3.connect
                    # appears) — only the busy_timeout diverges: telemetry
                    # waits at most 250ms, then drops the row.
                    conn = state.connect(db_path)
                    conn.execute("PRAGMA busy_timeout=250")
                    conns[db_path] = conn
                state.log_query(conn, **fields)
            except Exception:  # noqa: BLE001 — telemetry never escalates
                logger.debug("query telemetry write failed", exc_info=True)
    finally:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — best effort at teardown
                logger.debug("telemetry connection close failed", exc_info=True)


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            # Daemon: a dropped telemetry row must never keep the process
            # alive. close() stops it deterministically at teardown.
            _worker = threading.Thread(
                target=_writer_loop, name="noesis-telemetry", daemon=True
            )
            _worker.start()


async def record_query(
    conn: sqlite3.Connection,
    *,
    interface: str,
    kind: str,
    project_id: str | None,
    channel: str | None = None,
    reranked: bool | None = None,
    latency_ms: float | None = None,
    result_count: int | None = None,
) -> None:
    fields: dict[str, Any] = dict(
        interface=interface,
        kind=kind,
        project_id=project_id,
        channel=channel,
        reranked=reranked,
        latency_ms=latency_ms,
        result_count=result_count,
    )
    try:
        # Quick read on the caller's thread: resolve the DB file behind the
        # shared conn so the writer opens the same DB. In-memory pragma, no
        # IO, no lock.
        row = conn.execute("PRAGMA database_list").fetchone()
        db_path = row[2] if row is not None else ""
        if not db_path:
            # In-memory DB: a second connection cannot reach it — write on
            # the shared conn as before (test-only shape; runtime DBs are
            # always files).
            state.log_query(conn, **fields)
            return
        _ensure_worker()
        try:
            _queue.put_nowait((db_path, fields))
        except queue.Full:
            logger.debug("query telemetry queue full; dropping row")
    except Exception:  # noqa: BLE001 — telemetry must never break the query path
        logger.debug("query telemetry write failed", exc_info=True)


async def flush(timeout: float = 5.0) -> None:
    """Wait until every row queued so far has been written.

    The write is asynchronous by design, so anything that reads a row back
    right after recording it (tests, and the dashboard's usage page if it
    ever wants read-your-writes) needs this barrier. Returns when the rows
    land or *timeout* elapses; it never raises."""
    with _worker_lock:
        alive = _worker is not None and _worker.is_alive()
    if not alive:
        return
    barrier = _Barrier()
    try:
        _queue.put_nowait(barrier)
    except queue.Full:
        return
    await asyncio.to_thread(barrier.event.wait, timeout)


def close(timeout: float = 5.0) -> None:
    """Stop the writer thread, draining what is already queued.

    Called from ``close_runtime_context`` beside ``ctx.conn.close()`` so the
    telemetry handles do not outlive the runtime that created them — an open
    second handle to the state DB blocks temp-directory cleanup on Windows
    and WSL, and survives a DB file removal. Bounded like the model workers
    (MCP-1): a stuck writer is abandoned, never waited on forever.
    """
    global _worker
    with _worker_lock:
        worker = _worker
        _worker = None
    if worker is None or not worker.is_alive():
        return
    try:
        _queue.put_nowait(_SHUTDOWN)
    except queue.Full:
        # Full queue means the writer is behind, not gone; block briefly so
        # the sentinel still lands rather than leaking the thread.
        try:
            _queue.put(_SHUTDOWN, timeout=timeout)
        except queue.Full:
            logger.warning("telemetry queue full; writer thread abandoned")
            return
    worker.join(timeout=timeout)
    if worker.is_alive():
        logger.warning(
            "telemetry writer did not stop within %.0fs; abandoning "
            "(daemon thread)",
            timeout,
        )
