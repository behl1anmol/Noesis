"""M4 tests for the retriever's rerank path — flag semantics, reordering,
candidate depth, text hygiene.

The store is a stub returning canned hits: ``search_code`` only calls
``store.search``, so these stay pure unit tests (the real store's query
building is covered by test_vectorstore.py, the end-to-end path by
test_api.py and the golden run).
"""

from __future__ import annotations

from noesis.core.embedder import FakeEmbedder
from noesis.core.reranker import FakeReranker
from noesis.core.retriever import search_code


class StubStore:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.calls: list[dict] = []

    def search(self, project_id: str, **kwargs) -> list[dict]:
        self.calls.append({"project_id": project_id, **kwargs})
        return [dict(hit) for hit in self._hits]


def hit(path: str, text: str, score: float = 0.5) -> dict:
    return {
        "file_path": path,
        "start_line": 1,
        "end_line": 10,
        "language": "python",
        "symbol_name": None,
        "score": score,
        "snippet": text[:200],
        "text": text,
    }


def fusion_hits() -> list[dict]:
    # Fusion order: db.py first. The query "validate token" overlaps auth.py's
    # text fully, db.py's not at all — a reranker must flip the order.
    return [
        hit("db.py", "def connect(dsn): return Driver(dsn)", 0.9),
        hit("auth.py", "def validate_token(token): check expiry", 0.8),
        hit("util.py", "def validate(x): pass", 0.7),
    ]


async def test_no_reranker_returns_fusion_order_without_text():
    store = StubStore(fusion_hits())
    result = await search_code(store, FakeEmbedder(), "validate token", "p1")
    assert result["reranked"] is False
    assert [h["file_path"] for h in result["hits"]] == ["db.py", "auth.py", "util.py"]
    assert all("text" not in h and "rerank_score" not in h for h in result["hits"])
    assert store.calls[0]["with_text"] is False
    assert store.calls[0]["top_k"] == 10


async def test_rerank_defaults_on_when_reranker_wired():
    store = StubStore(fusion_hits())
    reranker = FakeReranker()
    result = await search_code(
        store, FakeEmbedder(), "validate token", "p1", reranker=reranker
    )
    assert result["reranked"] is True
    # Overlap scores: auth.py=1.0, util.py=0.5, db.py=0.0 — fusion order flipped.
    assert [h["file_path"] for h in result["hits"]] == ["auth.py", "util.py", "db.py"]
    assert result["hits"][0]["rerank_score"] == 1.0
    assert all("text" not in h for h in result["hits"])
    assert len(reranker.calls) == 1


async def test_rerank_false_opts_out_per_request():
    store = StubStore(fusion_hits())
    reranker = FakeReranker()
    result = await search_code(
        store, FakeEmbedder(), "validate token", "p1", reranker=reranker, rerank=False
    )
    assert result["reranked"] is False
    assert reranker.calls == []
    assert store.calls[0]["with_text"] is False


async def test_rerank_true_without_reranker_states_not_applied():
    store = StubStore(fusion_hits())
    result = await search_code(
        store, FakeEmbedder(), "validate token", "p1", rerank=True
    )
    assert result["reranked"] is False
    assert all("rerank_score" not in h for h in result["hits"])


async def test_rerank_fetches_candidate_depth_then_truncates_to_top_k():
    store = StubStore(fusion_hits())
    result = await search_code(
        store,
        FakeEmbedder(),
        "validate token",
        "p1",
        top_k=2,
        reranker=FakeReranker(),
        candidates=50,
    )
    # Store is asked for the rerank candidate depth, response is top_k.
    assert store.calls[0]["top_k"] == 50
    assert store.calls[0]["with_text"] is True
    assert len(result["hits"]) == 2
    assert [h["file_path"] for h in result["hits"]] == ["auth.py", "util.py"]


async def test_rerank_candidates_never_shrink_top_k():
    store = StubStore(fusion_hits())
    await search_code(
        store,
        FakeEmbedder(),
        "q",
        "p1",
        top_k=80,
        reranker=FakeReranker(),
        candidates=50,
    )
    assert store.calls[0]["top_k"] == 80


async def test_rerank_ties_keep_fusion_order():
    store = StubStore(
        [hit("a.py", "nothing relevant", 0.9), hit("b.py", "also nothing", 0.8)]
    )
    result = await search_code(
        store, FakeEmbedder(), "zzz", "p1", reranker=FakeReranker()
    )
    # Both score 0.0 — stable sort preserves fusion order.
    assert [h["file_path"] for h in result["hits"]] == ["a.py", "b.py"]


async def test_rerank_with_no_hits_never_calls_reranker():
    store = StubStore([])
    reranker = FakeReranker()
    result = await search_code(
        store, FakeEmbedder(), "q", "p1", reranker=reranker, rerank=True
    )
    assert result == {"hits": [], "reranked": True}
    assert reranker.calls == []


async def test_sparse_channel_skips_query_embed_with_rerank():
    store = StubStore(fusion_hits())
    embedder = FakeEmbedder()
    result = await search_code(
        store, embedder, "validate token", "p1", channel="sparse",
        reranker=FakeReranker(),
    )
    assert embedder.query_calls == []
    assert result["reranked"] is True
