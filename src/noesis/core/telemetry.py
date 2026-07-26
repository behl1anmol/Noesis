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

The queue belongs to the worker *generation*, not to the module: a
``record_query`` racing :func:`close` starts a new worker with a new queue
rather than sharing the one being retired. With a single shared queue the
new worker could consume the shutdown sentinel — leaving the old worker
running forever with its connections open, the very leak ``close`` exists to
prevent — and ``close``'s drain could discard rows the new worker was about
to write.

The writer generation is per *process*, not per ``AppContext``, and
:func:`close` retires it — so the module assumes one live ``AppContext`` per
process. That holds by construction: ``app.py`` builds exactly one in its
lifespan and ``mcp/__main__.py`` builds exactly one in its, and the two are
separate processes. If a second context were ever built alongside a live one,
its teardown would discard rows the other had just enqueued (they are dropped,
and :func:`flush` would then honestly report nothing pending) — losing
telemetry rows, never corrupting anything. Making that safe would mean handing
each context its own writer handle rather than sharing one writer across DB
paths.

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
import time
from typing import Any

from . import state

logger = logging.getLogger(__name__)

# Bounded so a writer stalled on a foreign lock cannot grow memory without
# limit; a full queue drops the row, which is the documented contract. Sized
# well above any realistic burst of concurrent searches.
_QUEUE_MAX = 1000

_SHUTDOWN = object()

# How long flush() waits between attempts to slip its barrier into a full
# queue. Short enough that it costs nothing once the writer drains one row,
# long enough that a stalled writer is not polled thousands of times.
_FLUSH_RETRY_S = 0.01


class _Barrier:
    """Flush marker. The queue is FIFO, so once the writer reaches this the
    rows enqueued before it are already written."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()

# The queue belongs to the WORKER GENERATION, not to the module. A shared
# global queue lets a `record_query` that races `close()` start a fresh
# worker which then blocks on the same queue as the one being shut down —
# the new worker can consume the `_SHUTDOWN` sentinel, leaving the old
# worker running forever with its connections open (precisely the handle
# leak close() exists to prevent), while close() drains rows the new worker
# was about to write. One queue per generation makes those cases
# unrepresentable: a new generation gets a new queue, and close() only ever
# touches its own.
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None
_queue: queue.Queue[Any] | None = None


def _writer_loop(work: queue.Queue[Any]) -> None:
    """Sole owner of the telemetry connections. One handle per DB path,
    opened lazily; because only this thread touches them, the writes need
    no lock and concurrent queries never queue behind one another."""
    conns: dict[str, sqlite3.Connection] = {}
    try:
        while True:
            item = work.get()
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


def _ensure_worker() -> queue.Queue[Any]:
    """Return the live generation's queue, starting a worker if needed.

    The lock is held only long enough to start a thread, never across a
    write or a join, so this cannot stall the query path.
    """
    global _worker, _queue
    with _worker_lock:
        if _worker is None or not _worker.is_alive() or _queue is None:
            # Daemon: a dropped telemetry row must never keep the process
            # alive. close() stops it deterministically at teardown.
            _queue = queue.Queue(maxsize=_QUEUE_MAX)
            _worker = threading.Thread(
                target=_writer_loop,
                args=(_queue,),
                name="noesis-telemetry",
                daemon=True,
            )
            _worker.start()
        return _queue


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
        try:
            _ensure_worker().put_nowait((db_path, fields))
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
    # The barrier goes in while the lock is still held. Releasing it first and
    # putting afterwards leaves a window where close() retires the generation
    # and finishes its drain, so the barrier lands in a queue no worker will
    # ever serve and this waits the full timeout for nothing (PR #24 review).
    # close() cannot start until it acquires this same lock, so the barrier is
    # now either served by the live worker or set by close()'s drain.
    barrier = _Barrier()
    deadline = time.monotonic() + timeout
    while True:
        with _worker_lock:
            work = _queue if (_worker is not None and _worker.is_alive()) else None
            if work is None:
                # No live worker means no live generation — nothing to wait
                # for. A generation close() retired is either empty (it
                # drained) or still being served by a worker that outlived its
                # join, and in that second case anything already queued is
                # written, not stranded.
                return
            try:
                work.put_nowait(barrier)
                break
            except queue.Full:
                pass
        # A full queue is the opposite of "nothing to wait for": it is the
        # maximum backlog, reached only when the writer is stalled on a foreign
        # lock — the one time a caller most needs a real barrier. Dropping
        # ROWS on overflow is the documented contract; dropping the BARRIER and
        # returning would report "flushed" with up to _QUEUE_MAX rows pending
        # (PR #24 review). So retry under the same deadline. A blocking
        # `put(barrier, timeout=...)` is not the fix: the queue is only safe to
        # read under the lock, and holding it across a blocking call would
        # stall the `_ensure_worker` on the query path that this whole module
        # exists to keep out of the way.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Waited the caller's whole budget without ever queuing the
            # barrier. A flush that spends its timeout and gives up is honest;
            # one that returns instantly is not.
            return
        await asyncio.sleep(min(_FLUSH_RETRY_S, remaining))
    await asyncio.to_thread(barrier.event.wait, max(0.0, deadline - time.monotonic()))


def close(timeout: float = 5.0) -> None:
    """Stop the writer thread, draining what is already queued.

    Called from ``close_runtime_context`` beside ``ctx.conn.close()`` so the
    telemetry handles do not outlive the runtime that created them — an open
    second handle to the state DB blocks temp-directory cleanup on Windows
    and WSL, and survives a DB file removal. Bounded like the model workers
    (MCP-1): a stuck writer is abandoned, never waited on forever — and an
    abandoned writer keeps its queue, sentinel included, so it can still stop
    itself once the backlog clears.
    """
    # Retire this generation under the lock. Anything that races us from here
    # on calls _ensure_worker(), which sees no worker and builds a NEW queue
    # with a NEW worker — so the queue below is ours alone for the rest of
    # this function, and nothing we do to it can lose a row the caller
    # expected written or hand our sentinel to someone else's worker.
    global _worker, _queue
    with _worker_lock:
        worker, work = _worker, _queue
        _worker, _queue = None, None
    if worker is None or work is None or not worker.is_alive():
        # Still drain: a retired generation must not leave rows sitting in a
        # queue nobody will ever serve, which is what makes flush()'s
        # no-worker early return honest.
        if work is not None:
            _discard_queued(work)
        return
    try:
        work.put_nowait(_SHUTDOWN)
    except queue.Full:
        # The writer is behind, not gone. Clear the backlog so the sentinel
        # lands rather than leaking the thread — those rows were already
        # droppable under the overflow contract.
        _discard_queued(work)
        try:
            work.put_nowait(_SHUTDOWN)
        except queue.Full:  # pragma: no cover — the queue was just emptied
            logger.warning("telemetry queue full; writer thread abandoned")
    worker.join(timeout=timeout)
    if worker.is_alive():
        logger.warning(
            "telemetry writer did not stop within %.0fs; abandoning "
            "(daemon thread)",
            timeout,
        )
        # Do NOT drain here. The worker still owns this queue and has not
        # reached the sentinel yet — `_discard_queued` would swallow it, and
        # the worker would then block on `work.get()` forever with its
        # connections open: precisely the handle leak this function exists to
        # prevent (PR #24 review, reproduced with a 250ms-per-row backlog
        # against a 1s join). The backlog is the worker's to finish; re-arm
        # the stop in case the first sentinel was already consumed while the
        # thread was on its way out.
        try:
            work.put_nowait(_SHUTDOWN)
        except queue.Full:  # pragma: no cover — the worker is draining it
            pass
        return
    # After the join, not before it: rows ahead of the sentinel are written by
    # the exiting worker. Only what landed behind it is discarded.
    _discard_queued(work)


def _discard_queued(work: queue.Queue[Any]) -> None:
    dropped = 0
    while True:
        try:
            item = work.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, _Barrier):
            # Never strand a flush() waiter on a barrier no worker will reach.
            item.event.set()
        elif item is not _SHUTDOWN:
            dropped += 1
    if dropped:
        logger.debug("telemetry: dropped %d row(s) queued at shutdown", dropped)
