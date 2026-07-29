"""Regression test for the 2026-07-20 bug hunt, finding SRV-1.

``delete_project`` called ``ctx.store.delete_project_points`` synchronously
on the event loop. That method issues a Qdrant delete with ``wait=True`` — a
blocking network round-trip that can take seconds for a large project,
freezing healthz, dashboard polling, and in-flight searches. Every other
per-request Qdrant round-trip is offloaded via ``asyncio.to_thread``; the
wipe now is too. The launcher-silencing-before-wipe ordering (PR #18) is
untouched — only the wipe call moved off the loop.
"""

from __future__ import annotations

import threading
import time

from qdrant_client import QdrantClient

from noesis.app import AppContext
from noesis.core import dashboard as core_dashboard
from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore


def make_ctx(tmp_path) -> AppContext:
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder, reranker=None)


async def test_delete_project_wipes_points_off_the_event_loop(tmp_path):
    """The Qdrant wipe must run on a worker thread, not the loop thread —
    pre-fix, ``delete_project_points`` ran inline and its thread id equalled
    the event loop's."""
    ctx = make_ctx(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)

    loop_thread = threading.get_ident()
    wipe_threads: list[int] = []
    real_wipe = ctx.store.delete_project_points

    def spy(pid: str) -> None:
        wipe_threads.append(threading.get_ident())
        real_wipe(pid)

    ctx.store.delete_project_points = spy  # type: ignore[method-assign]

    assert await core_dashboard.delete_project(ctx, project_id) is True
    assert wipe_threads, "delete_project never wiped the project's points"
    assert wipe_threads[0] != loop_thread, (
        "delete_project_points ran on the event loop thread (blocking wait=True "
        "Qdrant round-trip)"
    )
    # The wipe still actually happened and the project row is gone.
    assert state.get_project(ctx.conn, project_id) is None


# --- PR #24 review: bounded is not the same as free ---------------------------


async def test_close_runtime_context_does_not_block_the_event_loop() -> None:
    """MCP-1 bounded the model workers' join at 5s, but `close()` was still
    called synchronously from an async teardown, so the loop stalled for up
    to 5s per resource (10s for embedder + reranker) — stalling the MCP
    session manager's own shutdown in the combined lifespan. The joins now
    run off the loop."""
    import asyncio

    from noesis.core import state
    from noesis.runtime import AppContext, close_runtime_context

    class SlowClosing:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            time.sleep(0.4)  # stands in for a worker join near its bound
            self.closed = True

    conn = state.connect(":memory:")
    embedder, reranker = SlowClosing(), SlowClosing()
    ctx = AppContext(conn=conn, store=None, embedder=embedder, reranker=reranker)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(ticker())
    try:
        await close_runtime_context(ctx)
    finally:
        beat.cancel()

    assert embedder.closed and reranker.closed
    # ~0.8s of closing: a loop-blocking teardown lets the ticker run zero
    # times. Off the loop it keeps ticking throughout.
    assert ticks > 5, f"event loop advanced only {ticks} times during teardown"
