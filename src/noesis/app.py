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
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import AsyncIterator

from fastapi import FastAPI
from fastmcp.utilities.lifespan import combine_lifespans
from qdrant_client import QdrantClient

from noesis.api.dashboard import STATIC_DIR, dashboard_router
from noesis.api.routes import router
from noesis.core import state
from noesis.core.config import Settings, StructuralSettings, load_settings
from noesis.core.embedder import Embedder, LocalSTEmbedder
from noesis.core.reranker import LocalCrossEncoderReranker, Reranker
from noesis.core.vectorstore import VectorStore
from noesis.mcp import build_mcp


@dataclass
class AppContext:
    """Core resources shared by all adapters. ``reranker`` is None when
    ``reranker.enabled=false`` — the kill switch (§3.3) removes the model
    entirely; ``rerank_candidates`` is the fused-candidate depth reranked
    per request (config ``reranker.candidates``)."""

    conn: Connection
    store: VectorStore
    embedder: Embedder
    reranker: Reranker | None = None
    rerank_candidates: int = 50
    structural: StructuralSettings = field(default_factory=StructuralSettings)
    git_fast_path: bool = True
    jobs: dict[str, asyncio.Task] = field(default_factory=dict)
    # M8 (ADR-40): live run progress (jobs.run_progress reads it) and the
    # watcher manager, both owned by the lifespan.
    progress: dict[str, dict] = field(default_factory=dict)
    watcher: object | None = None
    # Effective device pins from config.toml, if any — the dashboard's
    # device setting defers to them (operator config wins over UI state).
    # Both tracked: a reranker-only pin must also block the UI, or
    # set_compute_device would silently override it (PR #10 review).
    config_device_pin: str | None = None
    config_reranker_device_pin: str | None = None


async def build_runtime_context(cfg: Settings) -> AppContext:
    """Construct production core resources from settings. Shared by the
    FastAPI lifespan and the stdio MCP entry point (noesis.mcp.__main__) —
    one build path, so the two transports cannot diverge on wiring."""
    # Same persistent fastembed cache default as noesis.prefetch —
    # without it, the BM25 assets land in the system tmp dir and
    # get re-fetched after a reboot (runtime network, ADR-25-adjacent).
    import os

    from noesis.prefetch import FASTEMBED_CACHE_DEFAULT, FASTEMBED_CACHE_ENV

    os.environ.setdefault(FASTEMBED_CACHE_ENV, FASTEMBED_CACHE_DEFAULT)
    conn = state.connect(cfg.db_path)
    state.init_db(conn)
    # Crash recovery: a fresh process has no live index tasks, so any
    # 'running' row is a leftover that would jam the launch guard forever.
    orphaned = state.fail_orphaned_runs(conn)
    if orphaned:
        import logging

        logging.getLogger(__name__).warning(
            "marked %d orphaned 'running' index run(s) as failed (interrupted)",
            orphaned,
        )
    # Device precedence (ADR-40): an explicit config.toml pin wins (operator
    # config is never second-guessed by UI state); otherwise the dashboard's
    # persisted app_settings choice; otherwise auto-detect (None).
    stored_device = state.get_setting(conn, "compute_device")
    if stored_device == "auto":
        stored_device = None
    embedder_device = cfg.embedder.device or stored_device
    reranker_device = cfg.reranker.device or stored_device
    embedder = LocalSTEmbedder(
        model_id=cfg.embedder.model,
        dim=cfg.embedder.dim,
        batch_size=cfg.embedder.batch_size,
        device=embedder_device,
    )
    store = VectorStore(
        QdrantClient(url=cfg.qdrant.url), collection_name=cfg.qdrant.collection
    )
    store.ensure_collection(embedder)
    reranker: LocalCrossEncoderReranker | None = None
    if cfg.reranker.enabled:
        reranker = LocalCrossEncoderReranker(
            model_id=cfg.reranker.model,
            batch_size=cfg.reranker.batch_size,
            device=reranker_device,
        )
        if cfg.reranker.preload:
            await reranker.preload()
    return AppContext(
        conn=conn,
        store=store,
        embedder=embedder,
        reranker=reranker,
        rerank_candidates=cfg.reranker.candidates,
        structural=cfg.structural,
        git_fast_path=cfg.git.fast_path,
        config_device_pin=cfg.embedder.device,
        config_reranker_device_pin=cfg.reranker.device,
    )


def close_runtime_context(ctx: AppContext) -> None:
    """Tear down what build_runtime_context created: cancel orphan index
    jobs, stop model worker threads, close SQLite."""
    for task in list(ctx.jobs.values()):
        task.cancel()
    for resource in (ctx.embedder, ctx.reranker):
        close = getattr(resource, "close", None)
        if close is not None:
            close()
    ctx.conn.close()


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
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
            app.state.ctx.watcher.stop()
            if ctx is None:
                close_runtime_context(app.state.ctx)
            else:
                for task in list(app.state.ctx.jobs.values()):
                    task.cancel()

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
