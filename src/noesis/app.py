"""FastAPI application factory (Overview §4.2, expanded §3.1).

One process, localhost-only (CLAUDE.md rule 2 — run with
``uvicorn noesis.app:app --host 127.0.0.1``; never a wildcard bind). The
lifespan owns every core resource: SQLite connection, Qdrant client, the
Embedder. The FastMCP server (M6) mounts at ``/mcp`` via the Draft's
verified pattern — ``mcp.http_app(path="/")`` combined into this app's
lifespan with ``combine_lifespans`` so the MCP session manager's task
group is initialized alongside our resources.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastmcp.utilities.lifespan import combine_lifespans

from noesis.api.dashboard import STATIC_DIR, dashboard_router
from noesis.api.routes import router
from noesis.core.config import Settings, load_settings
from noesis.mcp import build_mcp
from noesis.runtime import (
    AppContext,
    build_runtime_context,
    close_runtime_context,
)

# AppContext/build_runtime_context/close_runtime_context live in
# noesis.runtime (L3) and are re-exported here for the many callers (and
# tests) that import them from noesis.app.
__all__ = [
    "AppContext",
    "build_runtime_context",
    "close_runtime_context",
    "create_app",
    "app",
]


def create_app(
    settings: Settings | None = None, ctx: AppContext | None = None
) -> FastAPI:
    """Build the app. Tests pass a pre-built *ctx* (fake embedder,
    in-memory Qdrant); production builds one from settings."""
    cfg = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if ctx is not None:
            app.state.ctx = ctx
        else:
            app.state.ctx = await build_runtime_context(cfg)
        # Watcher lives in the lifespan for prod and test contexts alike —
        # the Observer thread is lazy (spawns on the first scheduled watch),
        # so apps with no watched projects pay one asyncio task, nothing more.
        from noesis.core.watcher import WatcherManager

        app.state.ctx.watcher = WatcherManager(app.state.ctx)
        app.state.ctx.watcher.start()
        try:
            yield
        finally:
            await app.state.ctx.watcher.stop()
            if ctx is None:
                await close_runtime_context(app.state.ctx)
            else:
                # Test-supplied ctx owns its own conn/embedder teardown, but
                # still await the cancelled index tasks so their run rows are
                # marked failed before the fixture tears the ctx down (H5).
                tasks = [t for t in app.state.ctx.jobs.values() if not t.done()]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

    # MCP tools resolve the context lazily (per call) from this app's
    # state — the closure is late-bound, and tools can only run after the
    # combined lifespan has set app.state.ctx.
    mcp = build_mcp(lambda: app.state.ctx)
    mcp_app = mcp.http_app(path="/")

    app = FastAPI(
        title="noesis", lifespan=combine_lifespans(lifespan, mcp_app.lifespan)
    )
    # DNS-rebinding guard (PR #10 security review): binding 127.0.0.1 stops
    # remote hosts, but a browser on this machine visiting a page whose
    # domain re-resolves to 127.0.0.1 would reach the mutation endpoints
    # with readable responses. Rejecting foreign Host headers closes that
    # class; "testserver" is Starlette's TestClient default.
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    app.include_router(router)
    app.include_router(dashboard_router)
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/mcp", mcp_app)
    return app


app = create_app()
