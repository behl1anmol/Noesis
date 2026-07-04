"""Qdrant vector store — hybrid dense + BM25-sparse collection wrapper (M3).

Thin wrapper over ``qdrant_client.QdrantClient`` implementing the approved
Overview §7 Qdrant point and the expanded doc's Embedder-boundary rules:
the dense ``VectorParams(size=...)`` is read from ``embedder.dim`` at
collection-creation time, never hardcoded (§3.4 rule 1), and
``embedding_model`` is written to every payload as the versioning key
(§3.4 rule 2). Payload keyword indexes on ``project_id`` and ``language``
are created right after the collection so they exist before any bulk load
(§3.6).

The sparse channel is BM25 with server-side IDF (``Modifier.IDF``, §3.6).
The TF half is computed client-side by qdrant-client's fastembed
integration via ``models.Document(model="Qdrant/bm25")`` — verified
against qdrant-client 1.18: there is no server-side text inference on a
local deployment, so fastembed is a runtime dependency (ADR-32; the
expanded doc's "fastembed deliberately absent" note predates this check).
Fusion is server-side RRF via ``query_points`` prefetch (§3.3); the RRF
constant is fixed by the server and not exposed by ``FusionQuery``, so
the doc's k=60 default is aspirational, not configured (ADR-32).

Point ids are deterministic UUIDv5s of
``project_id:file_path:start_line:file_hash`` (Overview §6 step "Upsert"),
so re-indexing an unchanged file rewrites the same points — idempotent.

Still deliberately absent: no server-side ``path_prefix`` filter —
Qdrant's ``MatchText`` needs a full-text payload index (a keyword index
won't serve it on a real server, even though local ``:memory:`` mode
happens to accept it), so path filtering lands with the M5 filter work
rather than shipping a filter that only works in tests.

The Qdrant client talks only to localhost or ``:memory:`` — no code or
metadata leaves the machine (ADR-25). The caller constructs the client and
decides which.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Literal, Protocol

from qdrant_client import QdrantClient, models

# Fixed namespace for deterministic chunk point ids. Never change this:
# it would orphan every existing point on the next re-index.
CHUNK_NAMESPACE = uuid.UUID("6a3d61a6-97a1-4a3a-9f6b-2e5a8f0c4d17")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
# fastembed model id for the client-side BM25 TF encoding. Changing it
# changes tokenization, which silently invalidates every stored sparse
# vector — treat like an embedding-model change (full re-index).
BM25_MODEL_ID = "Qdrant/bm25"

SearchChannel = Literal["hybrid", "dense", "sparse"]

_SNIPPET_CHARS = 200


class _ChunkLike(Protocol):
    """What upsert_chunks needs from a chunk (the M2 cAST chunker's output)."""

    file_path: str
    start_line: int
    end_line: int
    language: str | None
    node_type: str | None
    symbol_name: str | None
    file_hash: str
    text: str


def chunk_point_id(
    project_id: str, file_path: str, start_line: int, file_hash: str
) -> str:
    """Deterministic point id — re-runs over unchanged content are idempotent."""
    return str(
        uuid.uuid5(CHUNK_NAMESPACE, f"{project_id}:{file_path}:{start_line}:{file_hash}")
    )


class VectorStore:
    """Dense-only Qdrant collection wrapper. One shared collection,
    ``project_id`` payload filter at query time (Overview §6)."""

    def __init__(
        self, client: QdrantClient, collection_name: str = "noesis_chunks"
    ) -> None:
        self._client = client
        self._collection = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self, embedder: Any) -> None:
        """Create the collection if missing, sized from ``embedder.dim``
        (§3.4 rule 1). Raises ValueError if it exists with a different
        dense size — the system refuses to serve mixed-model results
        (§3.4 rule 2) — or without the BM25 sparse vector (an M2-era
        collection): points written before M3 carry no sparse vector, so
        silently adding the config would leave the lexical channel empty
        for every existing chunk. The fix is the same as a model change:
        drop the collection and state, re-index fully."""
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            existing = (
                vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
            )
            if existing is None or existing.size != embedder.dim:
                found = "absent" if existing is None else f"size {existing.size}"
                raise ValueError(
                    f"collection {self._collection!r} has dense vector {found}, "
                    f"but embedder {embedder.model_id!r} produces dim "
                    f"{embedder.dim}; refusing mixed-model state. A model "
                    f"change requires a full re-embed into a fresh collection."
                )
            sparse = info.config.params.sparse_vectors or {}
            if SPARSE_VECTOR_NAME not in sparse:
                raise ValueError(
                    f"collection {self._collection!r} predates the M3 sparse "
                    f"channel (no {SPARSE_VECTOR_NAME!r} sparse vector). Its "
                    f"points have no BM25 vectors, so hybrid search would "
                    f"silently lose the lexical channel. Delete the collection "
                    f"and the project file state, then re-index."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedder.dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        # Keyword indexes must exist before the first bulk load (§3.6).
        for field in ("project_id", "language"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(
        self,
        project_id: str,
        chunks: list[_ChunkLike],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """Write chunks with the Overview §7 payload. ``text`` is stored so
        ``search`` snippets and the M4 reranker can read chunk content
        without re-opening files."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=chunk_point_id(
                    project_id, chunk.file_path, chunk.start_line, chunk.file_hash
                ),
                vector={
                    DENSE_VECTOR_NAME: vector,
                    # TF encoding runs client-side (fastembed) inside the
                    # qdrant-client; IDF weighting is applied server-side
                    # via Modifier.IDF. Nothing leaves the machine.
                    SPARSE_VECTOR_NAME: models.Document(
                        text=chunk.text, model=BM25_MODEL_ID
                    ),
                },
                payload={
                    "project_id": project_id,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "node_type": chunk.node_type,
                    "symbol_name": chunk.symbol_name,
                    "file_hash": chunk.file_hash,
                    "embedding_model": embedding_model,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(
            collection_name=self._collection, points=points, wait=True
        )

    def delete_file_chunks(
        self,
        project_id: str,
        file_paths: Iterable[str],
        *,
        exclude_file_hash: str | None = None,
    ) -> None:
        """Delete every point for the given files within one project.

        ``exclude_file_hash`` spares points carrying that hash — the indexer
        upserts a file's new chunks first, then prunes the old ones, so a
        failure never leaves a file with zero searchable chunks."""
        paths = list(file_paths)
        if not paths:
            return
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id),
                ),
                models.FieldCondition(
                    key="file_path", match=models.MatchAny(any=paths)
                ),
            ],
            must_not=(
                [
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=exclude_file_hash),
                    )
                ]
                if exclude_file_hash is not None
                else None
            ),
        )
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )

    def search(
        self,
        project_id: str,
        *,
        dense_vector: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 10,
        language: str | None = None,
        channel: SearchChannel = "hybrid",
        prefetch_limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search one project's chunks over the requested channel (§3.3).

        ``hybrid`` (default) runs dense + BM25 prefetches of
        ``prefetch_limit`` each and fuses server-side with RRF; ``dense``
        and ``sparse`` query a single channel — kept first-class because
        the M3 eval gate compares hybrid against exactly those baselines.
        The payload filter is applied inside each prefetch so both
        candidate lists are project-scoped before fusion.

        Returns span dicts per §3.3; ``snippet`` is the first ~200 chars
        of the stored chunk text. No ``path_prefix`` yet — see the module
        docstring."""
        must: list[models.FieldCondition] = [
            models.FieldCondition(
                key="project_id", match=models.MatchValue(value=project_id)
            )
        ]
        if language is not None:
            must.append(
                models.FieldCondition(
                    key="language", match=models.MatchValue(value=language)
                )
            )
        query_filter = models.Filter(must=must)

        if channel in ("dense", "hybrid") and dense_vector is None:
            raise ValueError(f"channel {channel!r} requires dense_vector")
        if channel in ("sparse", "hybrid") and query_text is None:
            raise ValueError(f"channel {channel!r} requires query_text")

        if channel == "dense":
            response = self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        elif channel == "sparse":
            response = self._client.query_points(
                collection_name=self._collection,
                query=models.Document(text=query_text, model=BM25_MODEL_ID),
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query_text, model=BM25_MODEL_ID
                        ),
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "file_path": payload.get("file_path"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "language": payload.get("language"),
                    "symbol_name": payload.get("symbol_name"),
                    "score": point.score,
                    "snippet": (payload.get("text") or "")[:_SNIPPET_CHARS],
                }
            )
        return results
