"""M2 tests for the Qdrant VectorStore wrapper — dense-only, in-memory."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore, chunk_point_id


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

    results = store.search(await embedder.embed_query("parse config"), "proj1")
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


async def test_project_id_filter_isolates_projects(store: VectorStore):
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    await index_chunks(
        store, embedder, "proj1", [make_chunk(file_path="src/one.py", symbol_name="one")]
    )
    await index_chunks(
        store, embedder, "proj2", [make_chunk(file_path="src/two.py", symbol_name="two")]
    )

    query = await embedder.embed_query("anything")
    hits1 = store.search(query, "proj1")
    hits2 = store.search(query, "proj2")
    assert [h["file_path"] for h in hits1] == ["src/one.py"]
    assert [h["file_path"] for h in hits2] == ["src/two.py"]
    assert store.search(query, "proj-unknown") == []


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
    assert {h["file_path"] for h in store.search(query, "proj1")} == {
        "src/a.py",
        "src/a.go",
    }
    assert [h["file_path"] for h in store.search(query, "proj1", language="go")] == [
        "src/a.go"
    ]


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
    assert [h["file_path"] for h in store.search(query, "proj1")] == ["src/keep.py"]
    assert [h["file_path"] for h in store.search(query, "proj2")] == ["src/gone.py"]


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
