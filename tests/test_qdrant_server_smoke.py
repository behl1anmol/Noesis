"""Opt-in smoke test against a real Qdrant server (issue #37, ADR-62).

Run with ``docker compose up -d`` then ``uv run pytest -m server``. Skipped
by default (pyproject deselects the ``server`` marker), and skipped rather
than failed when nothing is listening, so a contributor without Docker is
never blocked by it.

Why this tier exists at all: every other store-touching test in this repo uses
``QdrantClient(":memory:")``. The embedded client is a Python reimplementation
of the query semantics — it never performs the version handshake, never speaks
the wire protocol, and (as it warns) ignores payload indexes. So the entire
suite is structurally incapable of noticing that the pinned client and the
pinned server disagree, which is how issue #37 survived three minor versions.

Two things are asserted here and nowhere else:

1. The pinned pair is compatible *as the live server reports itself*, not as
   a YAML file claims. ``tests/test_qdrant_pin_compat.py`` checks the pins on
   disk; this checks the thing that actually answered the socket.
2. The M3 hybrid path — ``Modifier.IDF`` sparse config, ``models.Document``
   BM25 encoding, and ``FusionQuery`` RRF over two prefetches — round-trips
   against a real server.
"""

from __future__ import annotations

import os
import uuid
from importlib.metadata import version as dist_version

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.common.version_check import is_compatible

from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import index_project
from noesis.core.retriever import search_code
from noesis.core.vectorstore import VectorStore

from .test_qdrant_pin_compat import _compose_server_version

pytestmark = pytest.mark.server

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")


def _server_version_or_skip() -> str:
    """Report the live server's version, or skip if nothing is there.

    Deliberately a plain REST GET rather than constructing a client: the
    client runs its compatibility check on a *daemon thread*, so anything
    built on catching its warning would race. This asks the same endpoint the
    client's own check asks, synchronously.
    """
    try:
        response = httpx.get(QDRANT_URL, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — any failure means "not available"
        pytest.skip(
            f"no Qdrant server at {QDRANT_URL} ({exc}) — `docker compose up -d`"
        )
    server_version = response.json().get("version")
    if not server_version:
        pytest.fail(f"{QDRANT_URL} answered but reported no version: {response.text!r}")
    return server_version


@pytest.fixture(scope="module")
def server_version() -> str:
    return _server_version_or_skip()


@pytest.fixture()
def store(server_version):
    """A VectorStore on a throwaway collection.

    The collection name is unique per run and dropped afterwards: this test
    talks to whatever server the developer has running, which is very likely
    the one holding their real index. It must never touch the default
    ``noesis_chunks`` collection.
    """
    client = QdrantClient(url=QDRANT_URL)
    name = f"noesis_server_smoke_{uuid.uuid4().hex[:12]}"
    try:
        # Both halves are yielded so the payload-index check can query the
        # collection directly without reaching into VectorStore's internals.
        yield client, VectorStore(client, collection_name=name)
    finally:
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
        finally:
            client.close()


def test_live_server_is_compatible_with_the_pinned_client(server_version):
    """The issue's exact symptom, asserted instead of warned about.

    ``is_compatible`` is the predicate behind the client's own
    "incompatible with server version" UserWarning, so this fails precisely
    when a developer would see that warning in the app's startup log.
    """
    client_version = dist_version("qdrant-client")
    assert is_compatible(client_version, server_version), (
        f"qdrant-client {client_version} is incompatible with the server "
        f"answering at {QDRANT_URL} (version {server_version}). This is the "
        f"warning issue #37 was filed for."
    )


def test_live_server_matches_the_compose_pin(server_version):
    """A green compatibility check against an unpinned server would prove
    nothing about the documented setup, so pin what answered to what
    `docker compose up -d` is supposed to start."""
    pinned = _compose_server_version()
    assert server_version == pinned, (
        f"the server at {QDRANT_URL} reports {server_version}, but "
        f"docker-compose.yml pins {pinned}. Either a stale container is still "
        f"running (`docker compose up -d --force-recreate`) or this run is "
        f"testing a server nobody ships."
    )


async def test_hybrid_path_round_trips_against_a_real_server(tmp_path, store):
    """Index and search through the real client/server pair.

    ``FakeEmbedder`` vectors are sha256-derived and carry no semantics, so
    this asserts what BM25 and the plumbing can actually deliver — the sparse
    channel finding a verbatim identifier, and hybrid fusion returning
    well-formed hits — and deliberately does not assert a dense *ranking*,
    which for hash vectors would be an assertion about noise.
    """
    _client, store = store
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "def validate_jwt_expiry(claims):\n"
        '    """Reject tokens whose exp claim is in the past."""\n'
        "    return claims['exp'] > 0\n"
    )
    (repo / "geometry.py").write_text(
        "import math\n\ndef circle_area(radius):\n    return math.pi * radius ** 2\n"
    )

    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    assert store.ensure_collection(embedder) is True

    indexed = await index_project(conn, store, embedder, str(repo))
    assert indexed.chunks_written > 0
    assert indexed.files_failed == 0

    try:
        # Sparse: BM25 over a verbatim identifier. This is the channel that
        # needs models.Document to survive the round trip to the server.
        sparse = await search_code(
            store, embedder, "validate_jwt_expiry", indexed.project_id, channel="sparse"
        )
        assert [hit["file_path"] for hit in sparse["hits"]][:1] == ["auth.py"]

        # Hybrid: two prefetches fused server-side with RRF.
        hybrid = await search_code(
            store, embedder, "validate_jwt_expiry", indexed.project_id, channel="hybrid"
        )
        assert hybrid["hits"], "hybrid fusion returned nothing against a real server"
        for hit in hybrid["hits"]:
            assert hit["file_path"] in {"auth.py", "geometry.py"}
            assert hit["start_line"] >= 1
            assert hit["end_line"] >= hit["start_line"]

        # Dense: exercised for wire-format correctness only (see docstring).
        dense = await search_code(
            store, embedder, "validate_jwt_expiry", indexed.project_id, channel="dense"
        )
        assert dense["hits"]
    finally:
        conn.close()


def test_payload_indexes_are_real_on_a_server(store):
    """The embedded client warns that payload indexes have no effect, so the
    default suite cannot check that `ensure_collection` actually creates them.
    A missing `project_id` index is a silent per-query full scan, not an
    error — exactly the kind of thing only a real server can show."""
    client, store = store
    embedder = FakeEmbedder(dim=8)
    store.ensure_collection(embedder)
    info = client.get_collection(store.collection_name)
    assert {"project_id", "language"} <= set(info.payload_schema or {}), (
        f"expected payload indexes on project_id and language, found "
        f"{sorted(info.payload_schema or {})}"
    )
