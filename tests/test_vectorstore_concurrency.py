"""Default-suite concurrency regression coverage for issue #48 / ADR-76.

No live server needed: this exercises the underlying qdrant-client defect
(client-side BM25 inference has no locking, ``models.Document`` shares
unsynchronized accumulate/drain state per ``QdrantClient`` instance) through
concurrent threads sharing ONE ``:memory:`` client — the same shape as the
issue's own reproducer, just driven through ``VectorStore``'s public methods
instead of raw qdrant-client calls.

``test_single_client_configuration_races_under_concurrent_load`` is the
non-vacuous half: witnessed failing 5/5 attempts at these exact parameters
through this exact code path while writing this test (``dictionary changed
size during iteration``, ``AttributeError: 'list' object has no attribute
'indices'``, ``IndexError`` — the same three symptoms the issue reported).
Kept ``xfail(strict=False)`` rather than a hard assertion because it pins a
third-party data race, not our own code — a rare clean run must not redden
CI, and a run that stops failing entirely across many CI runs would be a
signal qdrant-client fixed its locking upstream (issue #48's Option 4),
worth revisiting then, not something to paper over with a tighter loop.

This file deliberately does NOT also carry a "split client configuration
does not race" test: two independent ``QdrantClient(":memory:")`` objects
have independent storage (verified while designing this fix — each
``:memory:`` instance is its own isolated store, never shared between
objects), so a test built that way would have zero shared state between the
writer and reader threads and pass trivially regardless of whether the fix
is correct — indistinguishable from a vacuous check. Proving the split
actually prevents corruption requires two clients that genuinely share
storage, which only a real server provides; that proof lives in
``tests/test_qdrant_concurrency.py`` (``@pytest.mark.server``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from noesis.core.vectorstore import VectorStore

WRITERS_PER_PROJECT = 4
READERS = 4
ITERS = 300


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


def _run_workload(store: VectorStore, project_ids: list[str]) -> list[BaseException]:
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
                store.upsert_chunks(project_id, [chunk], [[0.1] * 8], "fake-embedder")
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


def _make_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        name,
        vectors_config={"dense": {"size": 8, "distance": "Cosine"}},
        sparse_vectors_config={"bm25": {}},
    )


@pytest.mark.xfail(
    reason=(
        "third-party data race in qdrant-client's ModelEmbedder (issue #48); "
        "witnessed failing 5/5 at these parameters, kept non-strict so a rare "
        "clean run doesn't redden CI and a consistently-clean run signals the "
        "upstream race may have been fixed"
    ),
    strict=False,
)
def test_single_client_configuration_races_under_concurrent_load():
    """index_client omitted -> reuses client, exactly like every other test
    in this suite and Noesis's actual pre-ADR-76 production shape. This is
    the vulnerability the split exists to close."""
    client = QdrantClient(":memory:")
    store = VectorStore(client, collection_name="c")
    _make_collection(client, "c")
    errors = _run_workload(store, ["proj-a", "proj-b"])
    assert not errors, (
        f"single-client configuration raced, as expected: {errors[:3]!r} "
        f"(+{max(0, len(errors) - 3)} more)"
    )
