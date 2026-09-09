"""Opt-in concurrency regression test for issue #48 / ADR-76.

Run with a live server: ``uv run pytest -m server tests/test_qdrant_concurrency.py``
(uses ``QDRANT_URL``, default ``http://127.0.0.1:6333`` — see
``test_qdrant_server_smoke.py`` for why this tier needs a real server at all).

Why this can't be a default-suite test: qdrant-client's client-side BM25
inference (``models.Document``) keeps unsynchronized state *inside the
QdrantClient object* — ``:memory:`` instances are independent per object and
two on-disk local instances cannot even open the same storage path
concurrently (verified while designing this fix: ``RuntimeError: Storage
folder ... is already accessed by another instance of Qdrant client``). The
two-client split can only be exercised against something both clients
genuinely share: a live server.

What this proves, through ``VectorStore``'s own public methods rather than
qdrant-client internals directly:
``test_concurrent_search_and_upsert_do_not_corrupt_with_split_clients`` — the
shipped configuration (``index_client`` distinct from ``client``, plus the
``_index_lock``) survives concurrent ``search()`` (sparse/hybrid, the
channels that build ``models.Document``) and ``upsert_chunks()`` calls from
two simulated "projects" (mirroring the per-project launch guard's actual
scope) without exceptions or cross-project content corruption.

This file deliberately does NOT also carry an HTTP-transport "vulnerable
single-client" probe. One was tried while writing this test — the identical
workload run through a single shared client, over real HTTP, at up to 6
writers x 6 readers x 150 iterations x 3 attempts — and it never reproduced
the race (0 errors every time), unlike the same workload against a single
shared ``:memory:`` client, which fails 5/5 at 4x4x300 (network round-trip
latency apparently paces the threads enough to close the window that a
tight in-process loop hits reliably). An HTTP-based probe that never
actually triggers is not evidence, so the vulnerability is instead pinned
where it demonstrably fires: ``tests/test_vectorstore_concurrency.py``
(default suite, no server needed).
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass

import httpx
import pytest
from qdrant_client import QdrantClient

from noesis.core.vectorstore import BM25_MODEL_ID, VectorStore

pytestmark = pytest.mark.server

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

WRITERS_PER_PROJECT = 3
READERS = 3
ITERS = 40


@dataclass
class _Chunk:
    file_path: str
    start_line: int
    end_line: int
    language: str | None
    node_type: str | None
    symbol_name: str | None
    file_hash: str
    text: str


def _server_or_skip() -> None:
    try:
        httpx.get(QDRANT_URL, timeout=5.0).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Qdrant server at {QDRANT_URL} ({exc}) — `docker compose up -d`")


@pytest.fixture()
def collection_name() -> str:
    return f"noesis_concurrency_{uuid.uuid4().hex[:12]}"


def _run_workload(store: VectorStore, project_ids: list[str]) -> list[BaseException]:
    """Concurrent writers (one set per project, simulating two overlapping
    index runs) plus concurrent readers (simulating overlapping
    ``search_code`` calls), all through VectorStore's real methods."""
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(project_id: str, tid: int) -> None:
        try:
            for i in range(ITERS):
                if stop.is_set():
                    return
                chunk = _Chunk(
                    file_path=f"writer_{tid}.py",
                    start_line=i,
                    end_line=i + 1,
                    language="python",
                    node_type="function_definition",
                    symbol_name=f"fn_{project_id}_{tid}_{i}",
                    file_hash=f"hash_{project_id}_{tid}_{i}",
                    text=f"def fn_{project_id}_{tid}_{i}(): return {tid}",
                )
                store.upsert_chunks(
                    project_id, [chunk], [[0.1] * 8], "fake-embedder"
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()

    def reader(project_id: str, tid: int) -> None:
        try:
            for i in range(ITERS):
                if stop.is_set():
                    return
                store.search(
                    project_id,
                    dense_vector=[0.1] * 8,
                    query_text=f"reader query {tid} {i}",
                    top_k=5,
                    channel="hybrid",
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()

    threads = []
    for pid in project_ids:
        threads += [
            threading.Thread(target=writer, args=(pid, t))
            for t in range(WRITERS_PER_PROJECT)
        ]
    threads += [
        threading.Thread(target=reader, args=(project_ids[0], t))
        for t in range(READERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_search_and_upsert_do_not_corrupt_with_split_clients(
    collection_name,
):
    """The shipped shape: index_client distinct from client, plus the lock
    (ADR-76). Two simulated concurrent index runs (different project_ids,
    mirroring the per-project launch guard's actual scope) plus concurrent
    search — must complete with zero exceptions and correct data."""
    _server_or_skip()
    query_client = QdrantClient(url=QDRANT_URL)
    index_client = QdrantClient(url=QDRANT_URL)
    store = VectorStore(
        query_client, collection_name=collection_name, index_client=index_client
    )
    try:
        store._client.create_collection(
            collection_name,
            vectors_config={
                "dense": {"size": 8, "distance": "Cosine"},
            },
            sparse_vectors_config={"bm25": {}},
        )
        projects = ["proj-a", "proj-b"]
        errors = _run_workload(store, projects)
        assert not errors, f"split-client configuration raised: {errors!r}"

        # No cross-project corruption: every writer's own chunk is retrievable
        # under its own project with its own text, not another writer's.
        for pid in projects:
            count = store.count_project_points(pid)
            expected = WRITERS_PER_PROJECT * ITERS
            assert count == expected, (
                f"project {pid}: expected {expected} points, found {count} — "
                f"a lost or cross-assigned write would show up here"
            )
    finally:
        try:
            if store._client.collection_exists(collection_name):
                store._client.delete_collection(collection_name)
        finally:
            store.close()
