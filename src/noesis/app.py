"""FastAPI application factory (Overview §4.2, expanded §3.1).

One process, localhost-only (CLAUDE.md rule 2 — run with
``uvicorn noesis.app:app --host 127.0.0.1``; never a wildcard bind). The lifespan
owns every core resource: SQLite connection, Qdrant client, the Embedder.
FastMCP mounts into this same app with a shared lifespan in M6.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import AsyncIterator

from fastapi import FastAPI
from qdrant_client import QdrantClient

from noesis.api.routes import router
from noesis.core import state
from noesis.core.config import Settings, load_settings
from noesis.core.embedder import Embedder, LocalSTEmbedder
from noesis.core.reranker import LocalCrossEncoderReranker, Reranker
from noesis.core.vectorstore import VectorStore


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
    jobs: dict[str, asyncio.Task] = field(default_factory=dict)


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    """Build the app. Tests pass a pre-built *ctx* (fake embedder,
    in-memory Qdrant); production builds one from settings."""
    cfg = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if ctx is not None:
            app.state.ctx = ctx
        else:
            # Same persistent fastembed cache default as noesis.prefetch —
            # without it, the BM25 assets land in the system tmp dir and
            # get re-fetched after a reboot (runtime network, ADR-25-adjacent).
            import os

            from noesis.prefetch import FASTEMBED_CACHE_DEFAULT, FASTEMBED_CACHE_ENV

            os.environ.setdefault(FASTEMBED_CACHE_ENV, FASTEMBED_CACHE_DEFAULT)
            conn = state.connect(cfg.db_path)
            state.init_db(conn)
            embedder = LocalSTEmbedder(
                model_id=cfg.embedder.model,
                dim=cfg.embedder.dim,
                batch_size=cfg.embedder.batch_size,
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
                )
                if cfg.reranker.preload:
                    await reranker.preload()
            app.state.ctx = AppContext(
                conn=conn,
                store=store,
                embedder=embedder,
                reranker=reranker,
                rerank_candidates=cfg.reranker.candidates,
            )
        try:
            yield
        finally:
            for task in list(app.state.ctx.jobs.values()):
                task.cancel()
            if ctx is None:
                for resource in (app.state.ctx.embedder, app.state.ctx.reranker):
                    close = getattr(resource, "close", None)
                    if close is not None:
                        close()
                app.state.ctx.conn.close()

    app = FastAPI(title="noesis", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
