"""Default-suite regression coverage for ``SearchGate`` (ADR-84).

What this proves, at the gate's own level:

* **Admission is bounded at exactly ``connections + queue_depth``** and the
  refusal is fast rather than a queued wait — the whole point of the change
  is that an overloaded Noesis says so instead of stalling silently. Both
  sides of that boundary are probed with an explicit one-slot margin (the
  capacity-th call must be *admitted*, the capacity+1-th refused), and the
  same burst is replayed against a gate with one more queue slot to show the
  rejection tracks capacity rather than the workload.
* **Slots survive failure.** A callable that raises and an await that is
  cancelled must both give their slot back; a leak here degrades into a
  permanently shrinking service that only a restart fixes, and it would be
  invisible to any test that just checks the happy path.
* **The rejection is actionable.** ``SearchOverloaded`` carries the live
  counts, and ``agent_message()`` names ``[qdrant] query_connections``. An
  agent-facing rejection that does not say what to change is the bug that
  assertion guards — the message is the only channel MCP has (no status
  codes), so its content is behaviour, not prose.
* **Only ``connections`` callables run at once**, with the rest queued and
  still completing. That 1:1 sizing against ``VectorStore``'s query pool is
  the gate's reason to exist as its own executor.

Why this tier: nothing here needs Qdrant, a model, or a network — the gate
is an executor plus a counter, so it is exercised directly with plain
callables and a real event loop (``asyncio_mode = "auto"``; no decorators).
Using ``store.search`` as the callable would only add a dependency on the
one thing this file does not test.

Deliberately NOT tested here:

* The REST 429/``Retry-After`` and MCP ``ToolError`` renderings of
  ``SearchOverloaded`` (``api/routes.py``, ``mcp/server.py``). Those are
  surface-layer translations of this exception; testing them from here would
  drag a whole app context in to re-assert what ``agent_message()`` already
  pins.
* That ``connections`` actually equals ``VectorStore``'s pool size in
  production. That 1:1 pairing is a wiring property of ``runtime.py``, not
  of this class, which cannot see the store at all.
* Throughput, latency or fairness under load. The gate makes no ordering
  promise beyond ``ThreadPoolExecutor``'s, and a timing assertion here would
  measure the machine, not the code (the sleeps below are only overlap
  windows, never thresholds).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from noesis.core.search_gate import (
    RETRY_AFTER_SECONDS,
    SearchGate,
    SearchOverloaded,
)


class _Blocker:
    """A callable that parks inside the executor until released.

    Holding the worker threads is how a test pins the gate at a chosen
    admission count: without it, calls would drain faster than the next one
    is submitted and the boundary could never be observed."""

    def __init__(self) -> None:
        self._released = threading.Event()
        self._lock = threading.Lock()
        self.started = 0

    def __call__(self) -> str:
        with self._lock:
            self.started += 1
        if not self._released.wait(timeout=10):
            raise AssertionError("blocked callable was never released")
        return "done"

    def release(self) -> None:
        self._released.set()


def _admitted(gate: SearchGate) -> int:
    stats = gate.stats()
    return stats["in_flight"] + stats["queued"]


async def _wait_for_admitted(gate: SearchGate, n: int, timeout: float = 5.0) -> None:
    """Await until exactly *n* calls hold admission (never a bare sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _admitted(gate) == n:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"gate never reached {n} admitted; stats={gate.stats()}")


async def _burst(
    gate: SearchGate, count: int
) -> tuple[list[str], list[SearchOverloaded]]:
    """Fire *count* concurrent ``run`` calls; return (completed, rejected)."""
    blocker = _Blocker()
    tasks = [asyncio.create_task(gate.run(blocker)) for _ in range(count)]
    # Let every task reach its admission check before anything can finish and
    # free a slot — otherwise a "burst" of N would really be N sequential calls.
    for _ in range(20):
        await asyncio.sleep(0.005)
        if sum(1 for t in tasks if t.done()) + _admitted(gate) == count:
            break
    blocker.release()
    settled = await asyncio.gather(*tasks, return_exceptions=True)
    completed = [r for r in settled if r == "done"]
    rejected = [r for r in settled if isinstance(r, SearchOverloaded)]
    unexpected = [
        r for r in settled if not isinstance(r, SearchOverloaded) and r != "done"
    ]
    assert not unexpected, f"unexpected outcomes from burst: {unexpected!r}"
    return completed, rejected


async def test_capacity_boundary_admits_the_capacity_th_call_and_refuses_the_next():
    """One-slot margin on both sides of ``connections + queue_depth``.

    Filling to capacity - 1 and then admitting one more is the low side; the
    call after that must be refused. Asserting only the refusal would pass on
    a gate whose capacity is off by one in the safe direction."""
    gate = SearchGate(connections=2, queue_depth=2)
    assert gate.capacity == 4, "capacity is connections + queue_depth"
    blocker = _Blocker()
    tasks: list[asyncio.Task[str]] = []
    try:
        for _ in range(gate.capacity - 1):
            tasks.append(asyncio.create_task(gate.run(blocker)))
        await _wait_for_admitted(gate, gate.capacity - 1)

        # Low side of the boundary: the capacity-th call is admitted, not
        # refused. It parks rather than returning, which is why this is
        # asserted through the admission count instead of an await.
        last = asyncio.create_task(gate.run(blocker))
        tasks.append(last)
        await _wait_for_admitted(gate, gate.capacity)
        assert not last.done(), "capacity-th call must be admitted, not rejected"

        # High side: the very next call is refused, and refused immediately —
        # wait_for turns "queued forever" into a failure instead of a hang.
        with pytest.raises(SearchOverloaded):
            await asyncio.wait_for(gate.run(blocker), timeout=5)
        assert _admitted(gate) == gate.capacity, "a refusal must not take a slot"
    finally:
        blocker.release()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        gate.close()
    assert results == ["done"] * gate.capacity
    assert _admitted(gate) == 0


async def test_the_same_burst_that_overflows_one_gate_fits_a_larger_one():
    """Non-vacuity: five concurrent calls are refused only because capacity
    is four. One extra queue slot — same workload, same timing — and nothing
    is refused, so the rejection is a property of the limit, not of the
    test's own scheduling."""
    tight = SearchGate(connections=2, queue_depth=2)  # capacity 4
    roomy = SearchGate(connections=2, queue_depth=3)  # capacity 5
    try:
        completed, rejected = await _burst(tight, 5)
        assert len(rejected) == 1, f"expected exactly one refusal, got {rejected!r}"
        assert len(completed) == 4

        completed, rejected = await _burst(roomy, 5)
        assert rejected == [], f"capacity 5 must absorb 5 calls, got {rejected!r}"
        assert len(completed) == 5
    finally:
        tight.close()
        roomy.close()


async def test_rejection_carries_live_counts_and_names_the_knob_to_change():
    """The exception is the whole rejection contract: MCP has no status
    codes, so an agent only ever sees these numbers and this text."""
    gate = SearchGate(connections=2, queue_depth=1)  # capacity 3
    blocker = _Blocker()
    tasks = [asyncio.create_task(gate.run(blocker)) for _ in range(3)]
    try:
        await _wait_for_admitted(gate, 3)
        with pytest.raises(SearchOverloaded) as excinfo:
            await gate.run(blocker)
        exc = excinfo.value
        # 2 running (one per connection) and 1 queued behind them — not 3/0.
        assert (exc.in_flight, exc.queued, exc.capacity) == (2, 1, 3)
        assert exc.retry_after_seconds == RETRY_AFTER_SECONDS

        message = exc.agent_message()
        assert "[qdrant] query_connections" in message, (
            "an agent-facing rejection that does not name the knob that "
            f"raises the limit is the bug this guards: {message!r}"
        )
        assert "Nothing failed" in message, (
            "the agent must be told the request was not run, so it retries "
            f"rather than treating this as a broken search: {message!r}"
        )
        for count in ("2", "1", "3"):
            assert count in message
    finally:
        blocker.release()
        await asyncio.gather(*tasks, return_exceptions=True)
        gate.close()


async def test_a_slot_is_released_when_the_submitted_callable_raises():
    """A failing search must not consume a slot permanently: three failures
    against a capacity of two would otherwise leave the gate wedged shut."""
    gate = SearchGate(connections=1, queue_depth=1)  # capacity 2

    def boom() -> str:
        raise ValueError("search blew up")

    try:
        for _ in range(gate.capacity + 1):
            with pytest.raises(ValueError):
                await gate.run(boom)
        assert gate.stats() == {
            "in_flight": 0,
            "queued": 0,
            "capacity": 2,
            "connections": 1,
        }

        # Capacity is *fully* available again, checked on both sides: a burst
        # of exactly capacity is absorbed, one more is refused. Asserting only
        # the first would still pass if one slot of two had leaked.
        completed, rejected = await _burst(gate, gate.capacity)
        assert rejected == [] and len(completed) == gate.capacity
        completed, rejected = await _burst(gate, gate.capacity + 1)
        assert len(rejected) == 1 and len(completed) == gate.capacity
    finally:
        gate.close()


async def test_a_cancelled_await_frees_its_slot_only_once_the_job_is_done():
    """Cancellation frees the slot when the WORK stops, not when the await
    returns.

    An earlier version of this test asserted the slot came back the instant
    the await was cancelled, which is what the pre-fix code did and is wrong:
    ``run_in_executor`` cannot cancel a job the executor has already started,
    so that slot was freed while the call — and the pooled Qdrant connection
    it holds — was still running. The two cases differ and both matter:

    * a job still QUEUED is genuinely cancelled, and must free its slot at once;
    * a job already RUNNING keeps its slot until it finishes, or the gate
      over-admits and the bound stops meaning anything.
    """
    gate = SearchGate(connections=1, queue_depth=1)  # capacity 2
    running = _Blocker()
    queued = _Blocker()
    running_task = asyncio.create_task(gate.run(running))
    await _wait_for_admitted(gate, 1)
    queued_task = asyncio.create_task(gate.run(queued))
    await _wait_for_admitted(gate, 2)
    try:
        # The queued one never started: cancelling it must free its slot now.
        queued_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued_task
        await _wait_for_admitted(gate, 1)
        assert _admitted(gate) == 1, (
            "a job that never started must release immediately, and the "
            "still-running one must keep its slot"
        )

        # The running one keeps its slot through cancellation ...
        running_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running_task
        assert _admitted(gate) == 1, (
            "a cancelled await must NOT release a slot whose job is still "
            "running — the work still holds a pooled connection"
        )

        # ... and gives it back when the work actually stops.
        running.release()
        await _wait_for_admitted(gate, 0)
        result = await asyncio.wait_for(gate.run(lambda: "after-cancel"), timeout=5)
        assert result == "after-cancel"
    finally:
        running.release()
        queued.release()
        gate.close()


def test_close_is_idempotent():
    """Teardown closes the gate before the store, and runs on paths that may
    already have closed it — a second call must be a no-op, not a raise."""
    gate = SearchGate(connections=1, queue_depth=0)
    gate.close()
    gate.close()  # must not raise
    gate.close()


async def test_run_after_close_raises_rather_than_hanging():
    """After close there is no executor left to run on. The failure mode
    that matters is a silent hang, so this is bounded by ``wait_for``: a
    timeout fails the test just as loudly as a wrong exception type."""
    gate = SearchGate(connections=1, queue_depth=0)
    gate.close()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(gate.run(lambda: "nope"), timeout=5)


async def test_only_connections_callables_run_concurrently_and_the_rest_queue():
    """Concurrency is capped at ``connections``, not at ``capacity``: the
    queue slots are waiting room, not extra threads. The barrier proves the
    lower bound (two really do overlap) and the peak counter the upper one;
    with capacity 6 against connections 2, a peak of 6 would be the shape of
    a gate sized to its capacity."""
    gate = SearchGate(connections=2, queue_depth=4)  # capacity 6
    assert gate.connections < gate.capacity, "otherwise the peak check is trivial"
    barrier = threading.Barrier(gate.connections, timeout=10)
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(i: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Forces a genuine overlap of exactly ``connections`` threads: with
        # fewer workers than parties this times out, which is the lower-bound
        # half of the assertion.
        barrier.wait()
        time.sleep(0.05)
        with lock:
            active -= 1
        return i

    try:
        results = await asyncio.gather(*[gate.run(work, i) for i in range(6)])
        assert sorted(results) == list(range(6)), "queued calls must still run"
        assert peak == gate.connections, (
            f"observed {peak} concurrent callables against connections="
            f"{gate.connections} (capacity {gate.capacity})"
        )
    finally:
        gate.close()


async def test_cancelling_awaits_cannot_over_admit_work_to_the_executor():
    """A cancelled await must not free a slot whose job is still running.

    ``run_in_executor`` cannot cancel a job the executor has already started,
    so releasing admission in a ``finally`` around the await frees the slot
    while the call — and the pooled Qdrant connection it holds — is still in
    flight. That is not an exotic case: a REST client disconnecting mid
    ``/search`` makes Starlette cancel the handler, and an MCP client can
    cancel ``search_code``, so ordinary client behaviour rather than load
    would have defeated the bound.

    Watched failing against the pre-fix ``finally``-release: the gate reported
    ``in_flight=0, queued=0`` immediately after two cancellations while three
    jobs were still queued in the executor, and this test's peak reached 4
    against ``connections=2``.

    The oracle is the executor's real concurrency, not the gate's own
    counters, precisely because the counters are what was wrong.
    """
    def job() -> None:
        time.sleep(1.0)

    gate = SearchGate(connections=2, queue_depth=2)  # capacity 4
    try:
        cancelled = [asyncio.create_task(gate.run(job)) for _ in range(4)]
        await _wait_for_admitted(gate, 4)
        for task in cancelled:
            task.cancel()
        await asyncio.gather(*cancelled, return_exceptions=True)

        # Exactly two of those four had started (connections=2) and are still
        # running, so two slots are legitimately still held; the two that were
        # merely queued were really cancelled and really freed. A client that
        # gave up retries at once, so fire all six together — sequentially
        # they would drain and prove nothing.
        #
        # The count admitted is the oracle, not the count refused: BOTH builds
        # refuse something here (capacity is 4 against 6 retries), so refusals
        # alone cannot tell them apart. Post-fix at most 2 get in, because 2
        # slots are still held by running work. Pre-fix all four slots read as
        # free, so 4 get in. The margin between 2 and 4 is the whole defect.
        retried = [asyncio.create_task(gate.run(job)) for _ in range(6)]
        results = await asyncio.gather(*retried, return_exceptions=True)
        admitted = sum(not isinstance(r, SearchOverloaded) for r in results)
        assert admitted <= 2, (
            f"{admitted} of 6 retries were admitted against a capacity of 4 "
            f"with 2 slots still held by running work — cancelling the awaits "
            f"freed slots whose jobs had already started, so the gate is "
            f"over-admitting and its bound no longer means anything"
        )
    finally:
        gate.close()
