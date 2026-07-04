"""Opt-in integration test: real CodeRankEmbed, in-memory Qdrant.

Skipped by default (pyproject deselects the ``integration`` marker) so the
suite stays fast and offline. Run with ``uv run pytest -m integration``.
This is the automated half of the M2 exit criterion — NL→code returns sane
spans with the real model; the live ``POST /search`` check against the
Docker Qdrant is done manually at milestone close.
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
        "import math\n"
        "\n"
        "def circle_area(radius):\n"
        "    return math.pi * radius ** 2\n"
    )

    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)

    result = await index_project(conn, store, embedder, str(repo))
    assert result.chunks_written > 0

    hits = await search_code(
        store, embedder, "where do we validate JWT expiry", result.project_id, top_k=2
    )
    assert hits
    assert hits[0]["file_path"] == "auth.py", (
        f"expected the JWT chunk first, got {hits[0]}"
    )
    assert hits[0]["start_line"] >= 1 and hits[0]["end_line"] >= hits[0]["start_line"]
