"""Core runtime context — construction and teardown, no web framework.

Split out of ``noesis.app`` (L3) so the stdio MCP entry point
(``python -m noesis.mcp``) can build/tear down core resources without
importing ``noesis.app``, whose module body calls ``create_app()`` and would
otherwise build an entire unused FastAPI app (second FastMCP instance, static
mount, config read) at import — any failure there killing the stdio server
before ``main()`` runs. Nothing here has an import-time side effect.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from sqlite3 import Connection

from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.config import Settings, StructuralSettings
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
    # Outer embed-batch size for index runs (config ``embedder.batch_size``).
    # The embedder re-batches internally for values below this, but the
    # indexer's slicing is the effective ceiling — without carrying the
    # config here, values above the indexer's default were silently capped.
    embed_batch_size: int = 32
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
    import logging
    import os

    from noesis.prefetch import FASTEMBED_CACHE_ENV, default_fastembed_cache

    os.environ.setdefault(FASTEMBED_CACHE_ENV, default_fastembed_cache())
    # Log the resolved DB so a divergence between the HTTP and stdio
    # transports (each logs its own path at startup) is visible, not silent.
    logging.getLogger(__name__).info("state db: %s", cfg.db_path)
    conn = state.connect(cfg.db_path)
    state.init_db(conn)
    # Crash recovery: mark runs whose owning process is dead as failed, so a
    # leftover 'running' row can't jam the launch guard forever. Owner-gated
    # (M7) so a co-running process's live run is never mislabelled.
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
    log = logging.getLogger(__name__)
    # A silent hang here (Qdrant down/unreachable) is a common false "bug"
    # report — name what we're waiting on before the blocking round-trips.
    log.info("connecting to Qdrant at %s", cfg.qdrant.url)
    store = VectorStore(
        QdrantClient(url=cfg.qdrant.url), collection_name=cfg.qdrant.collection
    )
    store.ensure_collection(embedder)
    log.info("Qdrant collection %s ready", cfg.qdrant.collection)
    # The Qdrant-side half of the crash recovery above: fail_orphaned_runs
    # cleans SQLite rows a dead process left behind, this cleans the points
    # one left behind. Startup is the only safe moment — no run of ours is in
    # flight yet, and a project row is always committed before its first point
    # is written, so a project another transport is mid-indexing cannot look
    # like an orphan. Skipped on an empty project table (see the method).
    swept = await asyncio.to_thread(
        store.delete_orphan_points, [row["id"] for row in state.list_projects(conn)]
    )
    if swept:
        log.warning(
            "swept %d orphaned point(s) belonging to no registered project", swept
        )
    reranker: LocalCrossEncoderReranker | None = None
    if cfg.reranker.enabled:
        reranker = LocalCrossEncoderReranker(
            model_id=cfg.reranker.model,
            batch_size=cfg.reranker.batch_size,
            device=reranker_device,
        )
        if cfg.reranker.preload:
            # preload() forces the ~2.3GB model load now; embedder/reranker
            # loads log their own start/ready lines (core.embedder/reranker).
            log.info("preloading reranker model (may take a while)")
            await reranker.preload()
    log.info("runtime ready")
    return AppContext(
        conn=conn,
        store=store,
        embedder=embedder,
        reranker=reranker,
        rerank_candidates=cfg.reranker.candidates,
        embed_batch_size=cfg.embedder.batch_size,
        structural=cfg.structural,
        git_fast_path=cfg.git.fast_path,
        config_device_pin=cfg.embedder.device,
        config_reranker_device_pin=cfg.reranker.device,
    )


async def close_runtime_context(ctx: AppContext) -> None:
    """Tear down what build_runtime_context created.

    Cancelling a task only *schedules* CancelledError; the cancelled
    ``execute_run`` still has to resume and run its ``except BaseException``
    handler, which calls ``state.finish_run`` to mark the run failed. So we
    must AWAIT the tasks' unwind before closing anything they touch — closing
    ``conn`` first would make that final write raise ``ProgrammingError`` and
    leave the run row stuck ``running`` (H5). Order: cancel → await → stop
    model workers → close SQLite."""
    tasks = [t for t in ctx.jobs.values() if not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for resource in (ctx.embedder, ctx.reranker):
        close = getattr(resource, "close", None)
        if close is not None:
            close()
    ctx.conn.close()
