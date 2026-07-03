"""Embedding boundary for the code indexer.

All embedding calls in this codebase go through the ``Embedder`` Protocol
defined here (CLAUDE.md hard rule 1). Implementations are local-only —
remote/hosted embedding was rejected (ADR-25) and the Protocol deliberately
exposes no credentials or transport surface.

``model_id`` is the versioning key (§3.4 rule 2): it is written to the Qdrant
payload and SQLite ``projects.embedding_model``, and changing implementations
triggers a full re-embed. Nothing outside this boundary may know the vector
size except by reading ``dim`` (§3.4 rule 1). Query-instruction quirks (e.g.
CodeRankEmbed's query prefix) live inside implementations' ``embed_query``
(§3.4 rule 3) — callers never apply prefixes themselves.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Async embedding interface. Local implementations only (ADR-25)."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """Deterministic in-memory Embedder for tests (M1 deliverable).

    Vectors are derived from sha256 of the input text, so equal texts embed
    identically and the suite needs no model download. ``embed_query`` hashes
    a ``"query:"``-prefixed variant of the text, mirroring the instruction-
    prefix seam of real implementations: a query embedding of some text is
    never equal to its document embedding, so tests can catch prefix misuse.
    """

    def __init__(self, dim: int = 8, model_id: str = "fake-embedder-v1") -> None:
        self._dim = dim
        self._model_id = model_id
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        # Stream sha256 blocks keyed by text + block index until dim bytes
        # are available; map each byte to [-1, 1].
        raw = bytearray()
        block = 0
        while len(raw) < self._dim:
            raw.extend(hashlib.sha256(f"{block}:{text}".encode()).digest())
            block += 1
        return [b / 255.0 * 2.0 - 1.0 for b in raw[: self._dim]]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(f"query:{text}")
