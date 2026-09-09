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

import contextlib
import queue
import threading
import uuid
from collections import Counter
from typing import Any, Iterable, Iterator, Literal, Protocol, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

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
        uuid.uuid5(
            CHUNK_NAMESPACE, f"{project_id}:{file_path}:{start_line}:{file_hash}"
        )
    )


class VectorStore:
    """Dense-only Qdrant collection wrapper. One shared collection,
    ``project_id`` payload filter at query time (Overview §6).

    **The hazard this class exists to contain (ADR-83, issue #48).**
    qdrant-client keeps its client-side inference bookkeeping — a
    ``_batch_accumulator`` dict and an ``_embed_storage`` FIFO — on the
    ``QdrantClient`` *object*, unsynchronized and keyed only by model name.
    Any two threads that build a ``models.Document`` on one client can
    receive each other's embeddings. Worse, the drain step recomputes only
    when the FIFO is empty and otherwise ``pop(0)``s, so a single race that
    leaves one residual entry offsets that client permanently: every later
    query silently returns another query's results, with no exception, until
    the process restarts. Reproduced 5/5 through this class's own methods,
    and measured at up to 15,864 wrong results in 16,000 queries against a
    live server. Independent of transport — REST and gRPC latch at the same
    rate — and unfixed upstream as of qdrant-client 1.19.

    **Which methods are exposed.** Only ``upsert_chunks`` (write) and
    ``search``'s sparse and hybrid channels (read) build a
    ``models.Document``. Dense-only search and every other method here
    (deletes, counts, scroll, retrieve, collection setup) touch no inference
    state at all — verified by instrumenting ``ModelEmbedder._accumulate``
    and ``_drain_accumulator``: zero calls for a dense query, one each for
    sparse and hybrid. Those paths are deliberately left unpooled and
    unlocked on ``client``, because there is nothing there to protect.

    **How it is contained, in two layers.**

    ``query_clients`` is the primary mechanism: a pool of connections, one
    checked out for the duration of each sparse/hybrid query, so no two
    threads ever meet on one client's inference state. Structural, not
    defensive — there is no shared object left to corrupt. Sized by
    ``[qdrant] query_connections`` and paired 1:1 with the search
    executor's slots (ADR-84), so a checkout finds a free connection
    whenever a thread is free. Checkout *blocks* when all are busy: that is
    the correct back-pressure primitive at this layer, and admission
    control — a bounded queue that rejects rather than queues without limit
    — belongs one layer up in the executor.

    ``_locks`` is the correctness floor, one lock per distinct client
    *object*. It matters because ``query_clients`` cannot always be a pool:
    two ``QdrantClient(":memory:")`` instances are two different databases,
    not two connections to one, so the ~25 in-memory tests and any embedded
    caller necessarily share a single client. Keying the locks to the object
    rather than to the role is what makes that configuration correct: when
    ``client``, ``index_client`` and the pool are all the same object, every
    path takes the *same* mutex. Keying by role instead would have search
    and upsert holding two different locks over one ``ModelEmbedder`` —
    issue #48's original race, reopened. In the production wiring the
    objects are distinct, so each lock is uncontended and costs an atomic.

    ``index_client`` stays separate from the pool so an index run never
    consumes a query connection, and its lock also guards it against
    itself: ``jobs.launch_index_run``'s guard (``state.try_start_run``) is
    per ``project_id``, so two *different* projects can reach
    ``upsert_chunks`` at once (routine under opt-in ``auto_reindex`` on 2+
    projects). ``max_concurrent_index_runs`` (ADR-85) bounds how many such
    runs exist at all, but that cap is machine-wide and this lock is what
    makes the overlap safe within a process."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "noesis_chunks",
        index_client: QdrantClient | None = None,
        query_clients: Sequence[QdrantClient] | None = None,
    ) -> None:
        self._client = client
        self._index_client = index_client if index_client is not None else client
        self._collection = collection_name
        # Falling back to ``[client]`` keeps the single-connection shape every
        # existing caller has: correct (the shared lock below serializes it),
        # just not concurrent. Only ``runtime.py`` passes a real pool.
        self._query_clients: tuple[QdrantClient, ...] = tuple(
            query_clients if query_clients else (client,)
        )
        self._query_pool: queue.Queue[QdrantClient] = queue.Queue()
        for qc in self._query_clients:
            self._query_pool.put(qc)
        # One lock per distinct client OBJECT, not per role — see the class
        # docstring. Built once here from the complete, fixed set of clients,
        # and every one of them is retained for this object's lifetime, so
        # keying on ``id()`` cannot collide with a recycled address.
        self._locks: dict[int, threading.Lock] = {}
        for c in (self._client, self._index_client, *self._query_clients):
            self._locks.setdefault(id(c), threading.Lock())

    def _lock_for(self, client: QdrantClient) -> threading.Lock:
        return self._locks[id(client)]

    @contextlib.contextmanager
    def _checked_out_query_client(self) -> Iterator[QdrantClient]:
        """Borrow a query connection and hold its lock for the whole call.

        Blocks until one is free; see the class docstring for why blocking
        rather than rejecting is right at this layer. The connection is
        returned to the pool even if the query raises, or a single failed
        search would shrink the pool permanently."""
        client = self._query_pool.get()
        try:
            with self._lock_for(client):
                yield client
        finally:
            self._query_pool.put(client)

    def close(self) -> None:
        """Close every distinct connection this store holds (issue #48
        teardown gap: the client was never closed at all before ADR-76).

        De-duplicated by object identity, so the single-client default
        closes exactly once however many roles that one object fills. Each
        close is attempted even if an earlier one raised — one failure must
        never leak the remaining connections (PR #50 review finding 3,
        generalized from two connections to the pool) — and the first
        exception is re-raised once all of them have been tried, which is
        the contract the two-connection version already had."""
        seen: set[int] = set()
        first: BaseException | None = None
        for client in (self._client, self._index_client, *self._query_clients):
            if id(client) in seen:
                continue
            seen.add(id(client))
            try:
                client.close()
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                if first is None:
                    first = exc
        if first is not None:
            raise first

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self, embedder: Any) -> bool:
        """Create the collection if missing, sized from ``embedder.dim``
        (§3.4 rule 1). Raises ValueError if it exists with a different
        dense size — the system refuses to serve mixed-model results
        (§3.4 rule 2) — or without the BM25 sparse vector (an M2-era
        collection): points written before M3 carry no sparse vector, so
        silently adding the config would leave the lexical channel empty
        for every existing chunk. The fix is the same as a model change:
        drop the collection and state, re-index fully.

        Returns ``True`` iff this call actually created the collection.
        The caller uses that to detect the wipe signature — a missing
        collection while the state DB still tracks indexed files."""
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
            return False
        try:
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
        except (UnexpectedResponse, ValueError):
            # Lost a first-startup race: the co-process (HTTP + stdio MCP
            # starting together against an empty Qdrant) created the
            # collection between our exists-check and this create. Not an
            # error — re-enter to run the same shape verification the
            # exists branch applies; the winner also creates the indexes.
            if not self._client.collection_exists(self._collection):
                raise
            # The race winner created the collection (and its indexes); this
            # process did not, so report False like the exists branch.
            self.ensure_collection(embedder)
            return False
        # Keyword indexes must exist before the first bulk load (§3.6).
        for field in ("project_id", "language"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        return True

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
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
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
        # index_client, never a pooled query connection (ADR-83): this is
        # the only write-path call that builds a models.Document, so it must
        # not share inference state with a concurrent search, and keeping it
        # off the pool means an index run can never consume a query slot.
        # Its lock guards it against ITSELF too — two different projects'
        # runs can reach here at once, since the launch guard is per
        # project_id — and, in the single-client configuration, it is the
        # very same lock the query path takes, which is what keeps that
        # configuration correct.
        with self._lock_for(self._index_client):
            self._index_client.upsert(
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

    def delete_project_points(self, project_id: str) -> None:
        """Delete every point belonging to a project (ADR-43 project
        deletion). Filter-only — never touches other projects' points."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def count_project_points(self, project_id: str) -> int:
        """Exact number of points stored for one project.

        Drift gate for the indexer: compared against the state DB's expected
        chunk total to detect a vector store that lost data (an externally
        dropped/recreated collection) while state still reports files as
        indexed. Same filtered ``count(exact=True)`` path as
        delete_orphan_points."""
        return self._client.count(
            collection_name=self._collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_id),
                    )
                ]
            ),
            exact=True,
        ).count

    def per_file_point_counts(self, project_id: str) -> dict[str, int]:
        """Point count per ``file_path`` for one project, via payload-only
        scroll.

        Only called after count_project_points has already detected drift,
        so the full-project scroll is paid for only on genuine divergence.
        ``file_path`` has no payload index, so this scans rather than
        filters on it — acceptable at recovery time, not per query."""
        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_id),
                        )
                    ]
                ),
                with_payload=["file_path"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in points:
                path = (point.payload or {}).get("file_path")
                if path is not None:
                    counts[str(path)] += 1
            if offset is None:
                break
        return dict(counts)

    def delete_orphan_points(self, live_project_ids: Iterable[str]) -> int:
        """Delete points whose project_id is not in *live_project_ids*, and
        return how many went. Startup-only crash recovery for the collection —
        the Qdrant-side counterpart of state.fail_orphaned_runs, and the
        catch-all for orphans no wipe could have prevented: a process killed
        mid-run, a delete that raced an in-flight write, a run whose upsert
        outlived the event loop.

        REFUSES to sweep when *live_project_ids* is empty. An empty project
        table cannot be told apart from a state DB that resolved to the wrong
        path — the exact misconfiguration the 2026-07-11 hunt found (a
        cwd-relative db_path silently opening a fresh, empty DB) — and this
        operation is destructive: on the wrong DB, "no live projects" would
        read as "delete the entire collection". A sweep skipped here costs a
        few dead points until a project exists; a sweep run there costs the
        whole index. Safe under the dual-transport deployment: a project row
        is committed before its first point is ever written, so a project
        another process is mid-indexing is never mistaken for an orphan.
        """
        live = list(live_project_ids)
        if not live:
            return 0
        # must_not(project_id in live) — the complement of the live set, so a
        # project registered but never indexed simply matches nothing.
        orphans = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="project_id", match=models.MatchAny(any=live)
                )
            ]
        )
        # Counted before the delete: Qdrant's delete result carries no count,
        # and a silent destructive startup step is exactly what an operator
        # needs to see in the log.
        count = self._client.count(
            collection_name=self._collection, count_filter=orphans, exact=True
        ).count
        if count:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=orphans),
                wait=True,
            )
        return count

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
        with_text: bool = False,
    ) -> list[dict[str, Any]]:
        """Search one project's chunks over the requested channel (§3.3).

        ``hybrid`` (default) runs dense + BM25 prefetches of
        ``prefetch_limit`` each and fuses server-side with RRF; ``dense``
        and ``sparse`` query a single channel — kept first-class because
        the M3 eval gate compares hybrid against exactly those baselines.
        The payload filter is applied inside each prefetch so both
        candidate lists are project-scoped before fusion.

        Returns span dicts per §3.3; ``snippet`` is the first ~200 chars
        of the stored chunk text. ``with_text=True`` additionally returns
        the full stored chunk ``text`` — the M4 reranker scores (query,
        chunk_text) pairs from the payload instead of re-opening files; the
        retriever strips it before results leave core. No ``path_prefix``
        yet — see the module docstring."""
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
            # No models.Document anywhere on this path, so it touches no
            # per-client inference state and needs neither a pooled
            # connection nor a lock (ADR-83; verified by instrumenting
            # ModelEmbedder — zero accumulate/drain calls for dense).
            # Deliberately left on ``client`` so a dense-only caller (the
            # eval harness's baseline channels) never contends for a slot.
            response = self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        elif channel == "sparse":
            with self._checked_out_query_client() as qc:
                response = qc.query_points(
                    collection_name=self._collection,
                    query=models.Document(text=query_text, model=BM25_MODEL_ID),
                    using=SPARSE_VECTOR_NAME,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
        else:
            effective_prefetch = max(prefetch_limit, top_k)
            with self._checked_out_query_client() as qc:
                response = qc.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using=DENSE_VECTOR_NAME,
                            filter=query_filter,
                            limit=effective_prefetch,
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query_text, model=BM25_MODEL_ID
                            ),
                            using=SPARSE_VECTOR_NAME,
                            filter=query_filter,
                            limit=effective_prefetch,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            result = {
                "chunk_id": str(point.id),
                "file_path": payload.get("file_path"),
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
                "language": payload.get("language"),
                "symbol_name": payload.get("symbol_name"),
                "score": point.score,
                "snippet": (payload.get("text") or "")[:_SNIPPET_CHARS],
            }
            if with_text:
                result["text"] = payload.get("text") or ""
            results.append(result)
        return results

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch one stored chunk by its point id (M6 ``get_chunk`` tool).

        ``chunk_id`` values come from search hits — the deterministic
        UUIDv5 point ids. Returns the exact stored span with full chunk
        ``content`` (the payload ``text``), or None for an unknown id.
        Content is the *indexed* snapshot, not ground truth — callers read
        the live file before acting (§3.3)."""
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[chunk_id],
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            # A remote server rejects malformed point ids (non-UUID strings)
            # with 400 — same meaning as unknown. Anything else (connection
            # down, 5xx) must propagate, not masquerade as "unknown chunk_id".
            if exc.status_code == 400:
                return None
            raise
        except ValueError:
            # Local (in-process) Qdrant raises ValueError on a malformed
            # (non-UUID) point id where a remote server answers 400 — both
            # mean "no such chunk". Keeps the contract transport-independent.
            return None
        if not points:
            return None
        payload = points[0].payload or {}
        return {
            "chunk_id": str(points[0].id),
            "project_id": payload.get("project_id"),
            "file_path": payload.get("file_path"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
            "language": payload.get("language"),
            "symbol_name": payload.get("symbol_name"),
            "content": payload.get("text") or "",
        }
