"""Default-suite concurrency regression coverage for issues #48 / #47
(ADR-76, ADR-83).

No live server needed: this exercises the underlying qdrant-client defect
(client-side BM25 inference has no locking, ``models.Document`` shares
unsynchronized accumulate/drain state per ``QdrantClient`` instance) through
concurrent threads sharing ONE ``:memory:`` client — the same shape as the
issue's own reproducer, just driven through ``VectorStore``'s public methods
instead of raw qdrant-client calls.

What each test pins:

* ``test_concurrent_readers_each_get_their_own_document`` is the important
  one. Reader-vs-reader corruption is mostly *silent*: measured against a
  live server, up to 15,864 of 16,000 queries returned another query's
  document while only 4-7 raised. Counting exceptions is therefore not
  coverage — the six review rounds this bug survived all counted exceptions
  — so this test seeds a corpus where chunk *i* holds a token no other chunk
  contains and asserts every reader gets back its OWN chunk. The oracle is
  shown to discriminate (a single-threaded query for token *i* returns chunk
  *i* and nothing else) before it is trusted under load, and the corpus is
  re-queried single-threaded afterwards because the corruption latches: one
  race that leaves a residual entry in the FIFO offsets that client for the
  rest of the process.
* ``test_single_client_configuration_survives_concurrent_load`` is the
  writer-vs-reader half, and used to be an ``xfail(strict=False)`` probe
  asserting the opposite: at these exact parameters the pre-ADR-83 code
  failed 5/5 with ``dictionary changed size during iteration``,
  ``AttributeError: 'list' object has no attribute 'indices'`` and
  ``IndexError`` — the three symptoms issue #48 reported. The query pool
  plus the per-client-object locks closed that race, so the xfail started
  xpassing and the marker was removed: this is now a plain regression test,
  and a failure here means the guard is gone, not that a third party is
  flaky.
* ``test_single_client_configuration_shares_one_lock_across_every_role``
  pins the design property the fix rests on — locks keyed by client OBJECT,
  not by role. It is what makes the in-memory wiring correct, and it is
  invisible to any behavioural test that happens to pass.
* ``test_pooled_configuration_gives_each_client_its_own_lock_and_returns_
  failed_connections`` pins the other wiring: distinct clients must not
  contend, and a query that raises must still hand its connection back or
  the pool shrinks to nothing one failed search at a time.

This file deliberately does NOT also carry a "split client configuration
does not race" test: two independent ``QdrantClient(":memory:")`` objects
have independent storage (verified while designing this fix — each
``:memory:`` instance is its own isolated store, never shared between
objects), so a test built that way would have zero shared state between the
writer and reader threads and pass trivially regardless of whether the fix
is correct — indistinguishable from a vacuous check. That is also why the
pooled test below asserts only lock topology and pool bookkeeping, which are
real per-object properties, and never cross-client query results. Proving
the split actually prevents corruption requires clients that genuinely share
storage, which only a real server provides; that proof lives in
``tests/test_qdrant_concurrency.py`` (``@pytest.mark.server``).

Also deliberately absent: any assertion about the dense channel, which
builds no ``models.Document`` and is left unpooled and unlocked on purpose
(ADR-83) — a test locking it down would pin the absence of a guard that was
never needed.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from noesis.core.vectorstore import VectorStore

WRITERS_PER_PROJECT = 4
READERS = 4
ITERS = 300

# Reader-vs-reader oracle: one distinct token per chunk, one reader thread
# per token.
ORACLE_CHUNKS = 16
ORACLE_ITERS = 50
# GIL switch interval used only while the reader threads run — see the
# comment at its call site.
SWITCH_INTERVAL_S = 1e-5

# Lock-topology probe windows. A search or upsert against ``:memory:`` costs
# single-digit milliseconds, so 0.5s of "still blocked" is a ~50x margin over
# a call that merely happens to be slow, and 5s is a ~1000x margin for the
# same call once it is unblocked. Neither is a threshold being measured — the
# behaviour under test is binary (the lock is taken, or it is not).
BLOCK_WINDOW_S = 0.5
RELEASE_WINDOW_S = 5.0


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


def _oracle_token(i: int) -> str:
    """A word that appears in chunk *i*'s text and in no other chunk's.

    Purely alphabetic so BM25's tokenizer keeps it whole, and prefixed so it
    cannot collide with anything else in the corpus."""
    return "zqx" + "".join(chr(ord("a") + (i // 26**k) % 26) for k in (2, 1, 0))


def _seed_oracle_corpus(store: VectorStore, project_id: str, n: int) -> None:
    chunks = [
        _Chunk(
            file_path=f"oracle_{i}.py",
            start_line=i,
            end_line=i + 1,
            language="python",
            node_type="function_definition",
            symbol_name=f"fn_{i}",
            file_hash=f"hash_{i}",
            text=f"def fn_{i}(): return '{_oracle_token(i)}'",
        )
        for i in range(n)
    ]
    store.upsert_chunks(project_id, chunks, [[0.1] * 8] * n, "fake-embedder")


def _hits_for_token(store: VectorStore, project_id: str, i: int) -> list[str | None]:
    """Symbol names the sparse channel returns for chunk *i*'s private token."""
    return [
        hit["symbol_name"]
        for hit in store.search(
            project_id, query_text=_oracle_token(i), top_k=3, channel="sparse"
        )
    ]


def test_concurrent_readers_each_get_their_own_document():
    """Readers alone — no writer in sight — must each get their own chunk back.

    This is the gap that let issue #47 survive six review rounds: the race is
    almost entirely silent, so a test that only counts exceptions passes while
    the service hands agents another agent's search results. The assertion here
    is on CONTENT, and the corpus is built so a crossed result is unambiguous:
    chunk i contains a token no other chunk contains, so the only correct
    answer to a query for that token is chunk i."""
    client = QdrantClient(":memory:")
    store = VectorStore(client, collection_name="c")
    _make_collection(client, "c")
    try:
        _seed_oracle_corpus(store, "proj", ORACLE_CHUNKS)

        # The oracle discriminates BEFORE it is trusted under load: token i
        # returns chunk i and no other chunk, single-threaded. Without this,
        # a corpus where every query returned every chunk would make the
        # concurrent assertion below unfalsifiable.
        for i in (0, ORACLE_CHUNKS // 2, ORACLE_CHUNKS - 1):
            hits = _hits_for_token(store, "proj", i)
            assert hits == [f"fn_{i}"], (
                f"oracle is not discriminating: token {_oracle_token(i)!r} must "
                f"match chunk fn_{i} and nothing else, so that any other "
                f"chunk coming back is detectably wrong — got {hits!r}"
            )
            # Stated the other way round, explicitly: no OTHER chunk answers
            # to this token, which is exactly what a crossed result would be.
            other = (i + 1) % ORACLE_CHUNKS
            assert f"fn_{other}" not in hits

        wrong: list[tuple[int, int, list[str | None]]] = []
        errors: list[BaseException] = []

        def reader(i: int) -> None:
            # Each iteration is caught on its own so an exception does not
            # end the thread: the interesting signal is how many reads came
            # back WRONG, and stopping at the first raise would hide it.
            for it in range(ORACLE_ITERS):
                try:
                    got = _hits_for_token(store, "proj", i)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                    continue
                if got != [f"fn_{i}"]:
                    wrong.append((i, it, got))

        threads = [
            threading.Thread(target=reader, args=(i,)) for i in range(ORACLE_CHUNKS)
        ]
        # The unguarded window is a few Python statements wide (accumulate,
        # then drain), so at the default 5ms switch interval an unlocked
        # build can run thousands of queries without two threads landing
        # inside it. Shortening the interval for the concurrent phase makes
        # this test actually sensitive to the regression it exists to catch
        # rather than dependent on scheduling luck; it changes nothing about
        # what is asserted. Restored in ``finally`` so no other test inherits it.
        previous_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(SWITCH_INTERVAL_S)
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(previous_switch_interval)

        # Wrong results first: they are the silent failure this test exists
        # for. Exceptions are reported too, but a build that only raises is
        # the easy case — six review rounds already caught those.
        assert not wrong, (
            f"{len(wrong)} of {ORACLE_CHUNKS * ORACLE_ITERS} concurrent reads "
            f"returned another reader's document: {wrong[:5]!r} "
            f"({len(errors)} reads also raised)"
        )
        assert not errors, f"concurrent readers raised: {errors[:3]!r}"

        # The corruption latches — one race that leaves a residual entry in
        # qdrant-client's FIFO offsets that client for the rest of the
        # process — so a quiet, single-threaded pass after all load has
        # stopped is a second, independent oracle.
        for i in range(ORACLE_CHUNKS):
            assert _hits_for_token(store, "proj", i) == [f"fn_{i}"], (
                f"post-load quiet query for chunk {i} was wrong: the client's "
                f"inference state stayed offset after the concurrent phase"
            )
    finally:
        store.close()


def test_single_client_configuration_survives_concurrent_load():
    """index_client omitted -> reuses client, exactly like every other test in
    this suite and Noesis's actual pre-ADR-76 production shape.

    Was an xfail probe of the race; the pool plus the per-client-object locks
    closed it, so this now asserts the fix rather than the defect."""
    client = QdrantClient(":memory:")
    store = VectorStore(client, collection_name="c")
    _make_collection(client, "c")
    try:
        errors = _run_workload(store, ["proj-a", "proj-b"])
        assert not errors, (
            f"single-client configuration raced: {errors[:3]!r} "
            f"(+{max(0, len(errors) - 3)} more)"
        )
        # Every write landed: a corrupted upsert would show up as a missing
        # or cross-assigned point, which exception counting would miss.
        for pid in ("proj-a", "proj-b"):
            expected = WRITERS_PER_PROJECT * ITERS
            assert store.count_project_points(pid) == expected
    finally:
        store.close()


def test_single_client_configuration_shares_one_lock_across_every_role():
    """Locks are keyed by client OBJECT, not by role.

    This is the property that makes the in-memory wiring correct: two
    ``QdrantClient(":memory:")`` instances are two databases, not two
    connections, so an embedded caller necessarily passes one object for
    every role. Keying by role instead would put search and upsert on
    different mutexes over one ``ModelEmbedder`` — issue #48, reopened —
    and no behavioural test would notice until it corrupted something."""
    client = QdrantClient(":memory:")
    store = VectorStore(client, collection_name="c")
    _make_collection(client, "c")
    try:
        admin_lock = store._lock_for(store._client)
        assert store._lock_for(store._index_client) is admin_lock, (
            "the index role must take the same mutex as the admin role when "
            "they are the same client object"
        )
        for qc in store._query_clients:
            assert store._lock_for(qc) is admin_lock, (
                "a query connection must take the same mutex as the index "
                "role when they are the same client object"
            )
        # One object -> exactly one lock, and repeated lookups are stable
        # (a fresh lock per call would serialize nothing).
        assert len(store._locks) == 1
        assert store._lock_for(client) is store._lock_for(client)

        # Identity of the lock objects is not enough on its own — the paths
        # have to actually take THAT lock. Holding it here must block both a
        # sparse search and an upsert; a role-keyed design would leave each
        # path on its own mutex and let them sail straight through.
        for label, call in (
            (
                "sparse search",
                lambda: store.search(
                    "proj", query_text="anything", top_k=3, channel="sparse"
                ),
            ),
            (
                "upsert",
                lambda: store.upsert_chunks(
                    "proj",
                    [
                        _Chunk(
                            file_path="locked.py",
                            start_line=1,
                            end_line=2,
                            language="python",
                            node_type="function_definition",
                            symbol_name="fn_locked",
                            file_hash="hash_locked",
                            text="def fn_locked(): return 1",
                        )
                    ],
                    [[0.1] * 8],
                    "fake-embedder",
                ),
            ),
        ):
            finished = threading.Event()

            def run() -> None:
                call()
                finished.set()

            with admin_lock:
                worker = threading.Thread(target=run)
                worker.start()
                # BLOCK_WINDOW is ~50x the unblocked cost of either call
                # (both are single-digit milliseconds against :memory:), so
                # completing inside it means the lock was not taken at all.
                assert not finished.wait(BLOCK_WINDOW_S), (
                    f"{label} ran while the client's own lock was held, so it "
                    f"is taking some other mutex"
                )
            assert finished.wait(RELEASE_WINDOW_S), (
                f"{label} never completed after the lock was released"
            )
            worker.join(timeout=RELEASE_WINDOW_S)
            assert not worker.is_alive()
    finally:
        store.close()


def test_pooled_configuration_locks_per_client_and_returns_failed_connections():
    """The other wiring: distinct client objects get distinct locks (they
    share no inference state, so contending would be pure loss), and a query
    that RAISES still hands its connection back — a pool that leaked one
    connection per failed search would silently shrink to zero and then
    block every search forever.

    Only lock topology and pool bookkeeping are asserted, never query
    results: these ``:memory:`` clients are independent databases (see the
    module docstring), so any cross-client data assertion here would be
    vacuous."""
    admin = QdrantClient(":memory:")
    pool = [QdrantClient(":memory:") for _ in range(3)]
    store = VectorStore(admin, collection_name="missing", query_clients=pool)
    try:
        locks = [store._lock_for(c) for c in (admin, *pool)]
        assert len({id(lock) for lock in locks}) == 4, (
            "four distinct client objects must hold four distinct locks; "
            "sharing one would serialize connections that share no state"
        )
        assert len(store._locks) == 4

        assert store._query_pool.qsize() == len(pool)
        # More failures than the pool has members: a leak of even one
        # connection per failure would be certain to show up.
        for attempt in range(len(pool) + 2):
            with pytest.raises(ValueError):
                store.search("proj", query_text="anything", top_k=3, channel="sparse")
            assert store._query_pool.qsize() == len(pool), (
                f"pool shrank to {store._query_pool.qsize()} after "
                f"{attempt + 1} failed searches"
            )

        # And the connections are really back, not just counted: borrowing
        # all of them at once yields each distinct client exactly once. Safe
        # to do without blocking because the size was just asserted.
        with contextlib.ExitStack() as stack:
            borrowed = [
                stack.enter_context(store._checked_out_query_client())
                for _ in range(len(pool))
            ]
        assert {id(c) for c in borrowed} == {id(c) for c in pool}
    finally:
        store.close()
