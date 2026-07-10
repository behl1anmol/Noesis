"""Retrieval pipeline — hybrid dense + BM25 with RRF fusion (M3), optional
cross-encoder rerank (M4) per §3.3 and Overview §6.

Query embedding goes through the Embedder Protocol at HIGH priority so
searches preempt any running index batch; the sparse channel needs only
the raw query text (BM25 TF is encoded inside the vector store's client,
IDF server-side). ``channel`` selects hybrid (default), dense-only, or
sparse-only — the single-channel modes exist because the M3 eval gate
measures hybrid *against* them.

Reranking (§3.3 contract): when a reranker is wired and the request wants
it, the store returns the top ``candidates`` (~50) fused results with their
stored chunk text, the reranker scores (query, chunk_text) pairs on its own
worker thread, and the reordered ``top_k`` carry ``rerank_score``. The
``rerank`` flag defaults to reranker availability (config
``reranker.enabled``); ``rerank=true`` without a wired reranker is not an
error — the response just states reranking was not applied. Results are
candidates, not ground truth — callers read the live file before acting.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .embedder import Embedder
from .reranker import Reranker
from .vectorstore import SearchChannel, VectorStore


async def search_code(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    project_id: str,
    *,
    top_k: int = 10,
    language: str | None = None,
    channel: SearchChannel = "hybrid",
    reranker: Reranker | None = None,
    rerank: bool | None = None,
    candidates: int = 50,
) -> dict[str, Any]:
    """Search one project's chunks; returns ``{"hits": [...], "reranked":
    bool}`` so adapters state whether reranking was applied (§3.3) without
    re-deriving the decision. Skips the query embed entirely for sparse-only
    searches — no reason to queue on the embed worker for a channel that
    won't use the vector."""
    dense_vector = await embedder.embed_query(query) if channel != "sparse" else None
    apply_rerank = (
        rerank if rerank is not None else reranker is not None
    ) and reranker is not None
    pool = max(top_k, candidates) if apply_rerank else top_k
    # Per-channel prefetch stays `candidates` deep even when no reranker
    # runs: RRF recall depends on each channel's candidate depth, not on how
    # many fused results the caller keeps. With prefetch collapsed to top_k,
    # a hit ranked just outside one channel's top_k loses that channel's
    # RRF contribution entirely and can drop out of the fused top_k.
    prefetch = max(top_k, candidates)
    hits = await asyncio.to_thread(
        store.search,
        project_id,
        dense_vector=dense_vector,
        query_text=query,
        top_k=pool,
        language=language,
        channel=channel,
        prefetch_limit=prefetch,
        with_text=apply_rerank,
    )
    if apply_rerank and hits:
        scores = await reranker.rerank(query, [hit["text"] for hit in hits])
        # Stable sort: fusion order breaks rerank-score ties.
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        hits = [hits[i] | {"rerank_score": scores[i]} for i in order[:top_k]]
    for hit in hits:
        hit.pop("text", None)
    return {"hits": hits, "reranked": apply_rerank}
