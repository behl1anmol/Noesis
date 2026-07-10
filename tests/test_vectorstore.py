"""Tests for the Qdrant VectorStore wrapper — hybrid dense+BM25, in-memory.

The BM25 TF encoding runs through qdrant-client's fastembed integration,
so upserts here exercise the real sparse path (tiny local model assets,
prefetched by ``noesis.prefetch``)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    VectorStore,
    chunk_point_id,
)


def dense_search(store, query_vector, project_id, **kw):
    """Dense-only search shim — most tests below assert payload/filter
    behavior that is channel-independent; dense keeps them deterministic
    under FakeEmbedder. Hybrid/sparse behavior has its own tests."""
    return store.search(project_id, dense_vector=query_vector, channel="dense", **kw)


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    language: str | None
    node_type: str | None
    symbol_name: str | None
    file_hash: str
    text: str


def make_chunk(
    file_path: str = "src/app.py",
    start_line: int = 1,
    text: str = "def handler(request):\n    return respond(request)\n",
    **overrides,
) -> Chunk:
    fields = dict(
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + text.count("\n"),
        language="python",
        node_type="function_definition",
        symbol_name="handler",
        file_hash="aaaa1111",
        text=text,
    )
    fields.update(overrides)
    return Chunk(**fields)


@pytest.fixture()
def store() -> VectorStore:
    with warnings.catch_warnings():
        # Local :memory: mode warns that payload indexes are no-ops; the
        # index creation itself is the behavior under test.
        warnings.simplefilter("ignore", UserWarning)
        client = QdrantClient(":memory:")
        vs = VectorStore(client)
        yield vs


async def index_chunks(
    store: VectorStore, embedder: FakeEmbedder, project_id: str, chunks: list[Chunk]
) -> None:
    vectors = await embedder.embed_documents([c.text for c in chunks])
    store.upsert_chunks(project_id, chunks, vectors, embedder.model_id)


def count(store: VectorStore) -> int:
    return store._client.count(store.collection_name).count


async def test_collection_created_with_dim_from_embedder(store: VectorStore):
    embedder = FakeEmbedder(dim=12)
    store.ensure_collection(embedder)
    info = store._client.get_collection(store.collection_name)
    assert info.config.params.vectors["dense"].size == 12
    # Idempotent when the size matches.
    store.ensure_collection(embedder)


async def test_dim_mismatch_guard_raises(store: VectorStore):
    store.ensure_collection(FakeEmbedder(dim=8))
    with pytest.raises(ValueError, match="mixed-model"):
        store.ensure_collection(FakeEmbedder(dim=16, model_id="other-model-v2"))


async def test_upsert_then_search_returns_payload_fields(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    chunk = make_chunk(text="def parse_config(path):\n" + "    x = 1\n" * 100)
    await index_chunks(store, embedder, "proj1", [chunk])

    results = dense_search(store, await embedder.embed_query("parse config"), "proj1")
    assert len(results) == 1
    hit = results[0]
    assert hit["file_path"] == "src/app.py"
    assert hit["start_line"] == chunk.start_line
    assert hit["end_line"] == chunk.end_line
    assert hit["language"] == "python"
    assert hit["symbol_name"] == "handler"
    assert isinstance(hit["score"], float)
    assert hit["snippet"] == chunk.text[:200]
    assert len(hit["snippet"]) == 200  # long text truncated to ~200 chars
    assert "text" not in hit  # full text only on request (with_text)


async def test_with_text_returns_full_chunk_text(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    chunk = make_chunk(text="def parse_config(path):\n" + "    x = 1\n" * 100)
    await index_chunks(store, embedder, "proj1", [chunk])

    results = dense_search(
        store, await embedder.embed_query("parse config"), "proj1", with_text=True
    )
    assert results[0]["text"] == chunk.text  # untruncated, for the reranker


async def test_project_id_filter_isolates_projects(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store,
        embedder,
        "proj1",
        [make_chunk(file_path="src/one.py", symbol_name="one")],
    )
    await index_chunks(
        store,
        embedder,
        "proj2",
        [make_chunk(file_path="src/two.py", symbol_name="two")],
    )

    query = await embedder.embed_query("anything")
    hits1 = dense_search(store, query, "proj1")
    hits2 = dense_search(store, query, "proj2")
    assert [h["file_path"] for h in hits1] == ["src/one.py"]
    assert [h["file_path"] for h in hits2] == ["src/two.py"]
    assert dense_search(store, query, "proj-unknown") == []


async def test_language_filter(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store,
        embedder,
        "proj1",
        [
            make_chunk(file_path="src/a.py", language="python"),
            make_chunk(file_path="src/a.go", language="go"),
        ],
    )
    query = await embedder.embed_query("anything")
    assert {h["file_path"] for h in dense_search(store, query, "proj1")} == {
        "src/a.py",
        "src/a.go",
    }
    assert [
        h["file_path"] for h in dense_search(store, query, "proj1", language="go")
    ] == ["src/a.go"]


async def test_deterministic_chunk_id_reupsert_is_idempotent(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    chunks = [make_chunk(start_line=1), make_chunk(start_line=40)]
    await index_chunks(store, embedder, "proj1", chunks)
    assert count(store) == 2
    # Re-upserting identical chunks rewrites the same points.
    await index_chunks(store, embedder, "proj1", chunks)
    assert count(store) == 2
    # A changed file_hash (new content) yields new point ids.
    assert chunk_point_id("p", "f.py", 1, "hash-a") != chunk_point_id(
        "p", "f.py", 1, "hash-b"
    )
    assert chunk_point_id("p", "f.py", 1, "hash-a") == chunk_point_id(
        "p", "f.py", 1, "hash-a"
    )


async def test_delete_file_chunks_removes_only_that_file(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store,
        embedder,
        "proj1",
        [
            make_chunk(file_path="src/keep.py", start_line=1),
            make_chunk(file_path="src/gone.py", start_line=1),
            make_chunk(file_path="src/gone.py", start_line=50),
        ],
    )
    # Same path in another project must be untouched.
    await index_chunks(
        store, embedder, "proj2", [make_chunk(file_path="src/gone.py", start_line=1)]
    )
    assert count(store) == 4

    store.delete_file_chunks("proj1", ["src/gone.py"])
    assert count(store) == 2
    query = await embedder.embed_query("anything")
    assert [h["file_path"] for h in dense_search(store, query, "proj1")] == [
        "src/keep.py"
    ]
    assert [h["file_path"] for h in dense_search(store, query, "proj2")] == [
        "src/gone.py"
    ]


async def test_upsert_length_mismatch_raises(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    with pytest.raises(ValueError, match="chunks"):
        store.upsert_chunks("proj1", [make_chunk()], [], embedder.model_id)


async def test_embedding_model_written_to_payload(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(store, embedder, "proj1", [make_chunk()])
    points, _ = store._client.scroll(store.collection_name, with_payload=True)
    assert points[0].payload["embedding_model"] == embedder.model_id
    assert points[0].payload["file_hash"] == "aaaa1111"
    assert points[0].payload["project_id"] == "proj1"


# --- M3: sparse channel + hybrid fusion ---


async def test_sparse_vector_written_on_upsert(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(store, embedder, "proj1", [make_chunk()])
    points, _ = store._client.scroll(
        store.collection_name, with_payload=False, with_vectors=True
    )
    assert DENSE_VECTOR_NAME in points[0].vector
    sparse = points[0].vector[SPARSE_VECTOR_NAME]
    assert len(sparse.indices) == len(sparse.values) > 0


async def test_sparse_channel_matches_exact_symbol(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store,
        embedder,
        "proj1",
        [
            make_chunk(
                file_path="src/auth.py",
                symbol_name="parse_jwt_claims",
                text="def parse_jwt_claims(token):\n    return claims\n",
            ),
            make_chunk(
                file_path="src/geometry.py",
                symbol_name="circle_area",
                text="def circle_area(radius):\n    return 3.14 * radius ** 2\n",
            ),
        ],
    )
    hits = store.search("proj1", query_text="parse_jwt_claims", channel="sparse")
    assert hits and hits[0]["file_path"] == "src/auth.py"
    # A term absent from the corpus matches nothing.
    assert store.search("proj1", query_text="zzz_nonexistent", channel="sparse") == []


async def test_hybrid_fuses_both_channels(store: VectorStore):
    """The lexical hit must surface in hybrid results even when the dense
    channel (FakeEmbedder = content hashes, semantically meaningless)
    contributes nothing useful — the M3 thesis in miniature."""
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    chunks = [
        make_chunk(
            file_path=f"src/f{i}.py",
            symbol_name=f"fn{i}",
            text=f"def fn{i}(x):\n    return x + {i}\n",
        )
        for i in range(5)
    ]
    chunks.append(
        make_chunk(
            file_path="src/limiter.py",
            symbol_name="RateLimiter",
            text="class RateLimiter:\n    def acquire(self):\n        pass\n",
        )
    )
    await index_chunks(store, embedder, "proj1", chunks)

    query = await embedder.embed_query("RateLimiter")
    hits = store.search(
        "proj1", dense_vector=query, query_text="RateLimiter", channel="hybrid"
    )
    assert "src/limiter.py" in [h["file_path"] for h in hits]
    # Fused result set respects top_k.
    assert (
        len(
            store.search("proj1", dense_vector=query, query_text="RateLimiter", top_k=3)
        )
        <= 3
    )


async def test_hybrid_respects_project_filter(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store,
        embedder,
        "proj1",
        [make_chunk(file_path="src/one.py", text="def shared_symbol(): pass\n")],
    )
    await index_chunks(
        store,
        embedder,
        "proj2",
        [make_chunk(file_path="src/two.py", text="def shared_symbol(): pass\n")],
    )
    query = await embedder.embed_query("shared_symbol")
    hits = store.search(
        "proj1", dense_vector=query, query_text="shared_symbol", channel="hybrid"
    )
    assert [h["file_path"] for h in hits] == ["src/one.py"]


async def test_m2_era_collection_without_sparse_raises(store: VectorStore):
    """A collection created before M3 has no BM25 vectors on its points;
    ensure_collection must refuse rather than serve a silently
    dense-only 'hybrid'."""
    from qdrant_client import models

    embedder = FakeEmbedder(dim=8)
    store._client.create_collection(
        collection_name=store.collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=8, distance=models.Distance.COSINE
            )
        },
    )
    with pytest.raises(ValueError, match="sparse"):
        store.ensure_collection(embedder)


async def test_search_channel_argument_validation(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    with pytest.raises(ValueError, match="dense_vector"):
        store.search("proj1", query_text="q", channel="dense")
    with pytest.raises(ValueError, match="query_text"):
        store.search("proj1", dense_vector=[0.0] * 8, channel="sparse")
    with pytest.raises(ValueError, match="query_text"):
        store.search("proj1", dense_vector=[0.0] * 8, channel="hybrid")
