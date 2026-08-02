"""Usage telemetry — metadata-only query logging (ADR-40).

Feeds the dashboard's usage page. Records *that* a query happened and how
it performed (interface, kind, channel, latency, result count) — never the
query text: queries routinely quote proprietary code, and a local DB is
still a file that gets backed up, synced, and pasted into bug reports
(ADR-25 spirit). Logging is fire-and-forget: a telemetry failure must never
fail — or delay — the search that triggered it.

Writes are handed to a dedicated writer thread over a bounded queue and
never touch the caller's ``ctx.conn``. Two reasons, both from the
dual-transport deployment (HTTP server + stdio MCP sharing one DB file):

* ``ctx.conn`` has ``busy_timeout=5000``, so an INSERT contending with the
  other process's index-run writes would stall the event loop up to 5s.
* Even off the loop, an *awaited* write keeps the response waiting on that
  contention. Enqueueing returns immediately, so the query path pays
  nothing beyond a queue put.

The writer owns its own connections (one per DB path, ``busy_timeout=250``)
and is the only thread that touches them, so no lock is needed. On a full
queue or a busy DB the row is dropped: telemetry may lose a row, never slow
a query.

:class:`QueryTelemetry` holds all of that as *instance* state, and every
``AppContext`` owns one (ADR-59). That is what makes ``close()`` terminal:
once a context tears its writer down, a late ``record_query`` drops its row
instead of opening a fresh ``state.connect()`` handle the closing context
will never reap. Terminality costs nothing here because there is no global
to reset — a second context simply constructs a second writer, open from
birth. ``LocalSTEmbedder`` is the same shape for the same reason
(``embedder.py``: ``_closed``, a terminal idempotent ``close()``, and a
``_submit`` that refuses afterwards).

Dropping a late row is the right side to err on, and is not the same choice
``LocalSTEmbedder`` makes: a refused embedding is a correctness loss, so it
raises; a refused telemetry row is this module's documented contract, so it
returns quietly at DEBUG.

The bounded join at teardown is MCP-1's, not a caveat about ownership: a
writer stuck on a foreign lock is abandoned rather than waited on forever.
An abandoned writer keeps its own queue with the shutdown sentinel still in
it, so it stops itself once the backlog clears, and it holds no reference
back to the :class:`QueryTelemetry` that started it — an abandoned thread
must not keep an entire ``AppContext`` alive.

Consequence worth knowing: the usage page is eventually consistent. A query
is normally readable within microseconds, but it is not guaranteed to be
there the instant the response returns — under the contention above it can
lag up to 250ms or be dropped entirely. Anything that needs read-your-writes
(tests, mainly) must await :meth:`QueryTelemetry.flush`.
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


def _writer_loop(work: queue.Queue[Any]) -> None:
    """Sole owner of the telemetry connections. One handle per DB path,
    opened lazily; because only this thread touches them, the writes need
    no lock and concurrent queries never queue behind one another.

    Takes the queue rather than the owning :class:`QueryTelemetry` on
    purpose: a writer abandoned by a timed-out join outlives its owner, and
    a thread that held ``self`` would keep the whole ``AppContext`` — its
    connection, its Qdrant client, its loaded models — reachable with it.
    """
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


class QueryTelemetry:
    """One query-telemetry writer, owned by one ``AppContext`` (ADR-59).

    The worker thread starts lazily on the first accepted row (daemon, so
    never calling :meth:`close` is safe) and stops at :meth:`close`, which
    is terminal and idempotent. Two live contexts in one process therefore
    have two independent writers: neither one's teardown can discard the
    other's queued rows, retire the other's worker, or leave a handle to the
    other's DB open. That is a mechanism, not an invariant — the point of
    issue #27.

    *queue_max* bounds the backlog. It is a constructor argument rather than
    a module constant to monkeypatch, which is the same thing instance state
    buys everywhere else here: a test that wants a small queue builds a small
    writer instead of mutating one every other test shares.
    """

    def __init__(self, queue_max: int = _QUEUE_MAX) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_max)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._closed = False

    def _submit(self, item: Any) -> bool:
        """Enqueue *item*, starting the writer thread on first use.

        Returns False when the item was refused — the writer is closed, or
        the queue is at its bound. Both are drops, which is the contract.
        The ``_closed`` test and the put happen under the one lock
        :meth:`close` also takes, so a row can never land in a queue whose
        worker has already been sent its sentinel.
        """
        with self._lock:
            if self._closed:
                return False
            if self._worker is None:
                # Daemon: a dropped telemetry row must never keep the process
                # alive. close() stops it deterministically at teardown.
                self._worker = threading.Thread(
                    target=_writer_loop,
                    args=(self._queue,),
                    name="noesis-telemetry",
                    daemon=True,
                )
                self._worker.start()
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                return False
        return True

    async def record_query(
        self,
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
        """Queue one metadata-only usage row and return.

        *conn* is the caller's shared connection, and is used on the
        caller's own thread only — to name the DB file the writer should
        open, or to write inline when there is no file to open. The writer
        thread never touches it.
        """
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
            if self._closed:
                # Terminal (ADR-59): a torn-down context records nothing more,
                # on either DB shape. Read without the lock deliberately — a
                # bool read cannot tear, this is not a correctness gate, and
                # the authoritative check is _submit's, which tests the same
                # flag under the lock close() holds.
                logger.debug("query telemetry: writer closed; dropping row")
                return
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
            if not self._submit((db_path, fields)):
                logger.debug("query telemetry row not accepted; dropping")
        except Exception:  # noqa: BLE001 — telemetry must never break the query path
            logger.debug("query telemetry write failed", exc_info=True)

    async def flush(self, timeout: float = 5.0) -> None:
        """Wait until every row queued so far has been written.

        The write is asynchronous by design, so anything that reads a row
        back right after recording it (tests, and the dashboard's usage page
        if it ever wants read-your-writes) needs this barrier. Returns when
        the rows land or *timeout* elapses; it never raises."""
        # The barrier goes in while the lock is still held. Releasing it first
        # and putting afterwards leaves a window where close() retires the
        # writer and finishes its drain, so the barrier lands in a queue no
        # worker will ever serve and this waits the full timeout for nothing
        # (PR #24 review). close() cannot start until it acquires this same
        # lock, so the barrier is now either served by the live worker or set
        # by close()'s drain.
        barrier = _Barrier()
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                live = self._worker is not None and self._worker.is_alive()
                if self._closed or not live:
                    # Nothing to wait for. A writer that was never started has
                    # been handed nothing; one that close() stopped has either
                    # written its backlog (the join returned) or is finishing
                    # it on a queue it still owns, and in that second case
                    # anything already queued is written, not stranded.
                    return
                try:
                    self._queue.put_nowait(barrier)
                    break
                except queue.Full:
                    pass
            # A full queue is the opposite of "nothing to wait for": it is the
            # maximum backlog, reached only when the writer is stalled on a
            # foreign lock — the one time a caller most needs a real barrier.
            # Dropping ROWS on overflow is the documented contract; dropping
            # the BARRIER and returning would report "flushed" with a full
            # queue pending (PR #24 review). So retry under the same deadline.
            # A blocking `put(barrier, timeout=...)` is not the fix: the queue
            # is only safe to read under the lock, and holding it across a
            # blocking call would stall the `_submit` on the query path that
            # this whole module exists to keep out of the way.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Waited the caller's whole budget without ever queuing the
                # barrier. A flush that spends its timeout and gives up is
                # honest; one that returns instantly is not.
                return
            await asyncio.sleep(min(_FLUSH_RETRY_S, remaining))
        await asyncio.to_thread(
            barrier.event.wait, max(0.0, deadline - time.monotonic())
        )

    def close(self, timeout: float = 5.0) -> None:
        """Stop the writer thread, draining what is already queued.

        Terminal and idempotent: after this, :meth:`record_query` drops rows
        rather than starting a second worker. Called from
        ``close_runtime_context`` beside ``ctx.conn.close()`` so the
        telemetry handles do not outlive the runtime that created them — an
        open second handle to the state DB blocks temp-directory cleanup on
        Windows and WSL, and survives a DB file removal. Bounded like the
        model workers (MCP-1): a stuck writer is abandoned, never waited on
        forever.
        """
        with self._lock:
            if self._closed:
                # Idempotent, and deliberately does not wait for the first
                # caller's drain — same contract as LocalSTEmbedder.close().
                return
            self._closed = True
            worker = self._worker
        # From here the queue is frozen: every put path tests _closed under
        # the lock just released, so nothing new can arrive and the queue is
        # this function's alone for the rest of its body.
        if worker is None or not worker.is_alive():
            # Still drain: a retired writer must not leave rows sitting in a
            # queue nobody will ever serve, which is what makes flush()'s
            # no-worker early return honest.
            _discard_queued(self._queue)
            return
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            # The writer is behind, not gone. Clear the backlog so the sentinel
            # lands rather than leaking the thread — those rows were already
            # droppable under the overflow contract.
            _discard_queued(self._queue)
            try:
                self._queue.put_nowait(_SHUTDOWN)
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
            # reached the sentinel yet — `_discard_queued` would swallow it,
            # and the worker would then block on `work.get()` forever with its
            # connections open: precisely the handle leak this function exists
            # to prevent (PR #24 review, reproduced with a 250ms-per-row
            # backlog against a 1s join). The backlog is the worker's to
            # finish, and it will: exactly one sentinel is ever enqueued, only
            # `_writer_loop` consumes one, and it returns the moment it does.
            return
        # After the join, not before it: rows ahead of the sentinel are written
        # by the exiting worker. Nothing can have landed behind it, so this
        # normally finds an empty queue — it runs anyway so that the degraded
        # branch above, which discards a backlog and re-puts the sentinel while
        # the worker is concurrently draining, cannot leave a row behind.
        _discard_queued(self._queue)
