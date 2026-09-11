"""Opt-in integration tests: real models, in-memory Qdrant.

Skipped by default (pyproject deselects the ``integration`` marker) so the
suite stays fast and offline. Run with ``uv run pytest -m integration``.
The first is the automated half of the M2 exit criterion — NL→code returns
sane spans with the real model; the live ``POST /search`` check against the
Docker Qdrant is done manually at milestone close. The second (issue #52)
holds the cold-start readiness signals to the REAL reranker, since every
other test in the suite answers them from a stub: it downloads
``BAAI/bge-reranker-v2-m3`` (~2.3GB) on a cold cache, which is exactly what
``-m integration`` opts into.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.embedder import LocalSTEmbedder
from noesis.core.indexer import index_project
from noesis.core.retriever import search_code
from noesis.core.vectorstore import VectorStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    return LocalSTEmbedder()


async def test_nl_query_returns_sane_spans(tmp_path, embedder):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "import time\n"
        "\n"
        "def validate_jwt_expiry(claims):\n"
        '    """Reject tokens whose exp claim is in the past."""\n'
        "    return claims['exp'] > time.time()\n"
    )
    (repo / "geometry.py").write_text(
        "import math\n\ndef circle_area(radius):\n    return math.pi * radius ** 2\n"
    )

    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)

    result = await index_project(conn, store, embedder, str(repo))
    assert result.chunks_written > 0

    hits = (
        await search_code(
            store,
            embedder,
            "where do we validate JWT expiry",
            result.project_id,
            top_k=2,
            gate=None,
        )
    )["hits"]
    assert hits
    assert hits[0]["file_path"] == "auth.py", (
        f"expected the JWT chunk first, got {hits[0]}"
    )
    assert hits[0]["start_line"] >= 1 and hits[0]["end_line"] >= hits[0]["start_line"]


async def test_reranker_readiness_tracks_the_real_model_load():
    """Issue #52's two signals, against the real cross-encoder.

    Observed when this was written, on a cold HF cache: ``("missing", False)``
    before the download, ``("ready", True)`` after it, with
    ``resolved_device == "cpu"``. The ``False`` before ``preload()`` is the
    whole point of the field — assets alone do not mean the next reranked
    search is fast, because the ``CrossEncoder`` construction still has to
    happen.
    """
    from noesis.core.reranker import LocalCrossEncoderReranker
    from noesis.prefetch import model_assets_ready, reranker_readiness

    reranker = LocalCrossEncoderReranker(device="cpu")
    try:
        assert reranker.resolved_device is None
        # Whatever the cache holds, an unloaded model is not ready.
        _, ready_before = await reranker_readiness(reranker)
        assert ready_before is False

        await reranker.preload()

        assert reranker.resolved_device == "cpu"
        assert model_assets_ready(reranker.model_id) is True
        assert await reranker_readiness(reranker) == ("ready", True)
        # The kill-switch case needs no model at all.
        assert await reranker_readiness(None) == ("disabled", "disabled")
    finally:
        reranker.close()
