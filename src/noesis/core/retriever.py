"""Retrieval pipeline — hybrid dense + BM25 with RRF fusion in M3 (§3.3,
Overview §6).

Query embedding goes through the Embedder Protocol at HIGH priority so
searches preempt any running index batch; the sparse channel needs only
the raw query text (BM25 TF is encoded inside the vector store's client,
IDF server-side). ``channel`` selects hybrid (default), dense-only, or
sparse-only — the single-channel modes exist because the M3 eval gate
measures hybrid *against* them. Results are candidates, not ground truth —
callers read the live file before acting. M4 adds the optional reranker.
"""

from __future__ import annotations

from typing import Any

from .embedder import Embedder
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
) -> list[dict[str, Any]]:
    """Search one project's chunks. Skips the query embed entirely for
    sparse-only searches — no reason to queue on the embed worker for a
    channel that won't use the vector."""
    dense_vector = (
        await embedder.embed_query(query) if channel != "sparse" else None
    )
    return store.search(
        project_id,
        dense_vector=dense_vector,
        query_text=query,
        top_k=top_k,
        language=language,
        channel=channel,
    )
