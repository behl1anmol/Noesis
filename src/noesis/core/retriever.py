"""Retrieval pipeline, dense-only in M2 (§3.3, Overview §6).

Query embedding goes through the Embedder Protocol at HIGH priority so
searches preempt any running index batch. Results are candidates, not
ground truth — callers read the live file before acting. M3 adds the
sparse channel + RRF fusion here; M4 adds the optional reranker.
"""

from __future__ import annotations

from typing import Any

from .embedder import Embedder
from .vectorstore import VectorStore


async def search_code(
    store: VectorStore,
    embedder: Embedder,
    query: str,
    project_id: str,
    *,
    top_k: int = 10,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Dense semantic search over one project's chunks."""
    vector = await embedder.embed_query(query)
    return store.search(
        vector, project_id, top_k=top_k, language=language
    )
