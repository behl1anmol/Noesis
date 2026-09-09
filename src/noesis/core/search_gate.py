"""Admission control for the search path (ADR-84).

Two separate jobs, deliberately in one object because they must agree on one
number:

**Bounding.** Every ``store.search`` used to land on CPython's default
executor — ``min(32, cpu+4)`` threads, process-wide, shared with the indexer's
~15 ``to_thread`` sites, its git subprocesses, hashing and file reads. That
pool is neither sized for search nor reserved for it: a large index run and a
burst of agent queries compete inside it, and nothing bounds how many searches
pile up. This gate runs search on its own executor, sized to exactly the number
of Qdrant query connections in ``VectorStore``'s pool, so a thread is free
exactly when a connection is (ADR-83). That 1:1 pairing is the invariant — it
is why a pool checkout never has to wait, and why only search runs here.
``get_chunk`` and the point counts stay on the default executor on purpose:
they use the admin connection, so giving them a slot would consume a thread
that a connection is not backing and break the pairing.

**Rejecting.** A bounded executor alone only moves an unbounded backlog from
one queue to another. Under many agents the honest failure is a fast, legible
rejection rather than a request that sits for an unbounded time — a silent
stall is the exact failure mode this whole change exists to remove. So the gate
admits at most ``connections + queue_depth`` calls and refuses the rest with
``SearchOverloaded``, which the REST layer renders as 429 and the MCP layer as
a ``ToolError`` the agent can act on.

Rejection is not an error condition in the usual sense: nothing is broken, the
service is at capacity. That distinction is why REST answers 429 rather than
503 — 503 is also what a dead or unreachable server returns, and an agent that
cannot tell "busy" from "down" retries the wrong way. Connection-refused and a
failing ``/healthz`` already mean "down".
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# What a rejected caller is told to wait. Deliberately a flat, conservative
# integer rather than a computed estimate: HTTP Retry-After is integer seconds,
# and a derived figure here would be a guess dressed as a measurement. The
# message carries the real, factual state (how many are running and queued) so
# a caller can decide for itself.
RETRY_AFTER_SECONDS = 1


class SearchOverloaded(Exception):
    """Raised instead of queueing when the search path is saturated.

    Carries the live counts rather than a bare message so every surface can
    render the same facts in its own idiom."""

    def __init__(self, in_flight: int, queued: int, capacity: int) -> None:
        self.in_flight = in_flight
        self.queued = queued
        self.capacity = capacity
        self.retry_after_seconds = RETRY_AFTER_SECONDS
        super().__init__(
            f"search capacity reached: {in_flight} running, {queued} queued, "
            f"limit {capacity}"
        )

    def agent_message(self) -> str:
        """A rejection an agent can act on: what happened, what to do now, and
        what to change so it stops happening."""
        return (
            f"Noesis is at search capacity — {self.in_flight} searches running "
            f"and {self.queued} queued, against a limit of {self.capacity}. "
            f"Nothing failed; the request was not run. Retry in about "
            f"{self.retry_after_seconds}s, or reduce how many searches you "
            f"issue at once. To raise the limit, set [qdrant] "
            f"query_connections (and optionally query_queue_depth) in "
            f"config.toml — each connection costs one Qdrant connection and "
            f"one thread."
        )


class SearchGate:
    """A bounded executor for store calls, with fail-fast admission.

    ``connections`` slots run concurrently; ``queue_depth`` more may wait.
    Anything beyond that is refused. ``connections`` must match the size of
    ``VectorStore``'s query pool — pass both from the same derived value."""

    def __init__(self, connections: int, queue_depth: int) -> None:
        if connections < 1:
            raise ValueError(f"connections must be >= 1, got {connections}")
        if queue_depth < 0:
            raise ValueError(f"queue_depth must be >= 0, got {queue_depth}")
        self._connections = connections
        self._queue_depth = queue_depth
        self._capacity = connections + queue_depth
        self._admitted = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=connections, thread_name_prefix="noesis-search"
        )
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def connections(self) -> int:
        return self._connections

    def stats(self) -> dict[str, int]:
        """Live counts. ``queued`` is admitted-but-not-running, so it is what
        a caller is actually waiting behind."""
        with self._lock:
            admitted = self._admitted
        return {
            "in_flight": min(admitted, self._connections),
            "queued": max(0, admitted - self._connections),
            "capacity": self._capacity,
            "connections": self._connections,
        }

    def _enter(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SearchGate is closed")
            if self._admitted >= self._capacity:
                admitted = self._admitted
                raise SearchOverloaded(
                    in_flight=min(admitted, self._connections),
                    queued=max(0, admitted - self._connections),
                    capacity=self._capacity,
                )
            self._admitted += 1

    def _leave(self) -> None:
        with self._lock:
            self._admitted -= 1

    async def run(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Run *fn* on the bounded executor, or raise ``SearchOverloaded``.

        Admission is taken before the submit and released after the call
        settles, so a cancelled await still frees its slot."""
        self._enter()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, lambda: fn(*args, **kwargs)
            )
        finally:
            self._leave()

    def close(self) -> None:
        """Stop accepting work and release the threads. Idempotent, and safe
        to call from the teardown path that closes every other resource."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Not cancelling queued futures: an in-flight search holds a Qdrant
        # connection that VectorStore.close() is about to close, so letting
        # them finish is what keeps teardown ordered.
        self._executor.shutdown(wait=True)
