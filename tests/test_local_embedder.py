"""M2 tests for LocalSTEmbedder — worker thread, priority queue, prefix.

Uses the ``_load_model`` injection seam with a fake in-process model so the
suite never touches the network or downloads CodeRankEmbed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from noesis.core.embedder import Embedder, LocalSTEmbedder

QUERY_PREFIX = "Represent this query for searching relevant code: "


class FakeModel:
    """Records encode() calls; returns deterministic numpy vectors.

    ``block_first_call`` makes the first encode() wait on an Event so tests
    can hold the worker busy and observe queue-priority behavior.
    """

    def __init__(self, dim: int = 4, block_first_call: bool = False) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.started = threading.Event()  # set when encode() first entered
        self.release = threading.Event()
        self._block_first_call = block_first_call
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            first = not self.calls
            self.calls.append(list(texts))
        if first:
            self.started.set()
            if self._block_first_call:
                assert self.release.wait(timeout=5.0), "test never released model"
        return np.array(
            [
                [float(len(t)), float(i), 0.0, 1.0][: self.dim]
                for i, t in enumerate(texts)
            ]
        )


def make_embedder(model: FakeModel, **kwargs) -> LocalSTEmbedder:
    return LocalSTEmbedder(dim=model.dim, _load_model=lambda: model, **kwargs)


async def test_satisfies_embedder_protocol():
    embedder = make_embedder(FakeModel())
    assert isinstance(embedder, Embedder)
    assert embedder.model_id == "nomic-ai/CodeRankEmbed"
    assert embedder.dim == 4


async def test_query_prefix_applied_in_embed_query_only():
    model = FakeModel()
    embedder = make_embedder(model)
    await embedder.embed_documents(["def foo(): pass", "class Bar: ..."])
    await embedder.embed_query("find foo")
    embedder.close()
    assert model.calls[0] == ["def foo(): pass", "class Bar: ..."]  # raw texts
    assert model.calls[1] == [QUERY_PREFIX + "find foo"]
    # Prefix has the documented trailing space: prefix + query, no separator.
    assert model.calls[1][0].endswith(": find foo")


async def test_vectors_are_plain_float_lists():
    embedder = make_embedder(FakeModel())
    vectors = await embedder.embed_documents(["abc"])
    query_vec = await embedder.embed_query("abc")
    embedder.close()
    assert isinstance(vectors[0], list) and isinstance(vectors[0][0], float)
    assert isinstance(query_vec, list) and len(query_vec) == 4
    assert not isinstance(vectors[0], np.ndarray)


async def test_batching_splits_documents_by_batch_size():
    model = FakeModel()
    embedder = make_embedder(model, batch_size=32)
    texts = [f"text {i}" for i in range(70)]
    vectors = await embedder.embed_documents(texts)
    embedder.close()
    assert len(model.calls) == 3  # 32 + 32 + 6
    assert [len(c) for c in model.calls] == [32, 32, 6]
    assert len(vectors) == 70
    # Order preserved across batch boundaries.
    assert [t for call in model.calls for t in call] == texts


async def test_empty_documents_short_circuit():
    model = FakeModel()
    embedder = make_embedder(model)
    assert await embedder.embed_documents([]) == []
    assert model.calls == []


async def test_high_priority_query_preempts_queued_documents():
    model = FakeModel(block_first_call=True)
    embedder = make_embedder(model)

    # LOW job A: worker picks it up and blocks inside encode().
    job_a = asyncio.ensure_future(embedder.embed_documents(["low job A"]))
    assert await asyncio.to_thread(model.started.wait, 5.0)

    # While the worker is busy: enqueue LOW job B first, then HIGH query C.
    job_b = asyncio.ensure_future(embedder.embed_documents(["low job B"]))
    await asyncio.sleep(0.05)  # B demonstrably enqueued before C
    job_c = asyncio.ensure_future(embedder.embed_query("high query C"))
    await asyncio.sleep(0.05)

    model.release.set()
    await asyncio.gather(job_a, job_b, job_c)
    embedder.close()

    order = [call[0] for call in model.calls]
    assert order == ["low job A", QUERY_PREFIX + "high query C", "low job B"]


async def test_worker_survives_job_exception():
    inner = FakeModel()

    class FirstCallExplodes:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("encode exploded")
            return inner.encode(texts)

    model = FirstCallExplodes()
    embedder = LocalSTEmbedder(dim=4, _load_model=lambda: model)

    # The failing job's exception reaches the awaiting caller...
    with pytest.raises(RuntimeError, match="encode exploded"):
        await embedder.embed_documents(["a"])
    # ...and the SAME worker thread serves the next jobs successfully.
    assert len(await embedder.embed_documents(["b", "c"])) == 2
    assert len(await embedder.embed_query("d")) == 4
    embedder.close()
    assert model.calls == 3


async def test_model_load_failure_propagates_and_worker_survives():
    attempts: list[int] = []
    model = FakeModel()

    def flaky_load() -> FakeModel:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("no network in tests")
        return model

    embedder = LocalSTEmbedder(dim=4, _load_model=flaky_load)
    with pytest.raises(OSError, match="no network"):
        await embedder.embed_query("first")
    # Worker thread is still alive and retries the load on the next job.
    assert (await embedder.embed_query("second")) is not None
    embedder.close()
    assert len(attempts) == 2


async def test_close_is_idempotent_and_rejects_new_work():
    embedder = make_embedder(FakeModel())
    await embedder.embed_documents(["x"])
    embedder.close()
    embedder.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        await embedder.embed_query("nope")


async def test_close_before_any_work_is_safe():
    embedder = make_embedder(FakeModel())
    embedder.close()


async def test_close_is_bounded_while_worker_is_stuck_in_model_load():
    """Shutdown must not hang on ``worker.join()`` while the worker is mid
    model-load (the cold-cache download can take minutes): the join is bounded
    at 5s and the daemon worker is abandoned. Pre-fix, close() blocked until
    the load finished — here ~60s, the stall cap that also keeps a regression
    from hanging the suite forever."""
    load_started = threading.Event()
    release = threading.Event()
    model = FakeModel()

    def stuck_load():
        load_started.set()
        release.wait(timeout=60.0)  # hard cap: regression fails, never hangs
        return model

    embedder = LocalSTEmbedder(dim=4, _load_model=stuck_load)
    pending = asyncio.ensure_future(embedder.embed_query("held"))
    assert await asyncio.to_thread(load_started.wait, 5.0)
    started = time.monotonic()
    embedder.close()
    elapsed = time.monotonic() - started
    assert elapsed < 6.0, f"close() blocked {elapsed:.1f}s on a stuck worker"
    release.set()
    await pending  # abandoned worker still finishes the in-flight job


async def test_resolved_device_stays_none_until_model_load_succeeds():
    """PR #50 review finding 1: ``resolved_device`` (and therefore
    ``/healthz``'s ``embedder_ready``) must not go truthy while the real
    ``SentenceTransformer(...)`` constructor is still mid-flight — only
    ``_default_load`` is exercised here (real ``_load_model`` seam bypassed
    on purpose), with ``SentenceTransformer`` and ``resolve_device`` stubbed
    so no network/weights are touched."""
    started = threading.Event()
    release = threading.Event()

    class StubSentenceTransformer:
        def __init__(self, model_id: str, trust_remote_code: bool = True, device=None):
            started.set()
            assert release.wait(timeout=5.0), "test never released model"

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.array([[0.0, 1.0, 2.0, 3.0] for _ in texts])

    with (
        patch("sentence_transformers.SentenceTransformer", StubSentenceTransformer),
        patch("noesis.core.compute.resolve_device", return_value="cpu"),
    ):
        embedder = LocalSTEmbedder(dim=4)
        assert embedder.resolved_device is None
        pending = asyncio.ensure_future(embedder.embed_query("q"))
        assert await asyncio.to_thread(started.wait, 5.0)
        # Constructor is mid-flight: must still read as not-ready.
        assert embedder.resolved_device is None
        release.set()
        await pending
        assert embedder.resolved_device == "cpu"
        embedder.close()


async def test_resolved_device_stays_none_after_failed_load():
    """Companion to the above: a failed load must not leave
    ``resolved_device`` truthy either — pre-fix it was set before the
    constructor ran, so a caller polling ``/healthz`` saw
    ``embedder_ready: true`` forever after a load that never succeeded."""

    class ExplodingSentenceTransformer:
        def __init__(self, model_id: str, trust_remote_code: bool = True, device=None):
            raise OSError("simulated interrupted download")

    with (
        patch(
            "sentence_transformers.SentenceTransformer", ExplodingSentenceTransformer
        ),
        patch("noesis.core.compute.resolve_device", return_value="cpu"),
    ):
        embedder = LocalSTEmbedder(dim=4)
        with pytest.raises(OSError, match="simulated interrupted download"):
            await embedder.embed_query("q")
        assert embedder.resolved_device is None
        embedder.close()


def test_constructor_defaults():
    embedder = LocalSTEmbedder()
    assert embedder.model_id == "nomic-ai/CodeRankEmbed"
    assert embedder.dim == 768
    embedder.close()  # no worker ever started; must not hang


async def test_a_superseded_load_does_not_publish_its_device(caplog):
    """Companion to the reranker's identical test (issue #52 review). The two
    model boundaries are deliberate structural mirrors, so the same
    ``set_device``-during-load race lives here: an in-flight generation-0 load
    finishing after a device switch must not overwrite the ``None``
    ``set_device`` just wrote, or ``/healthz``'s ``embedder_ready`` goes true
    for a model the worker is about to drop and reload."""
    started = threading.Event()
    release = threading.Event()
    caplog.set_level(logging.INFO, logger="noesis.core.embedder")

    class StubSentenceTransformer:
        def __init__(self, model_id: str, trust_remote_code: bool = True, device=None):
            started.set()
            assert release.wait(timeout=5.0), "test never released model"

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.array([[0.0, 1.0, 2.0, 3.0] for _ in texts])

    with (
        patch("sentence_transformers.SentenceTransformer", StubSentenceTransformer),
        patch("noesis.core.compute.resolve_device", side_effect=lambda d: d or "cpu"),
    ):
        embedder = LocalSTEmbedder(dim=4)
        pending = asyncio.ensure_future(embedder.embed_query("q"))
        assert await asyncio.to_thread(started.wait, 5.0)
        embedder.set_device("cuda")
        assert embedder.resolved_device is None
        release.set()
        await pending
        assert embedder.resolved_device is None, (
            "a superseded load published its device — health would report ready "
            "for a model the worker is about to reload"
        )
        ready_lines = [
            r.getMessage() for r in caplog.records if "ready on" in r.getMessage()
        ]
        assert ready_lines, "no completion log line at all"
        assert "None" not in ready_lines[-1], ready_lines[-1]
        assert "cpu" in ready_lines[-1]
        embedder.close()


class _BumpDeviceOnFirstWorkerLock:
    """Drives a ``set_device`` into the window between the worker loop's
    generation read and the loader's own — a gap of two adjacent statements,
    unreachable by timing, so it is driven deterministically: the first time
    the MODEL WORKER thread takes the lock, switch the device first.

    Wrapping the lock rather than patching a method keeps the production code
    path intact; ``set_device`` re-enters this wrapper, which passes straight
    through after the first hit (the real lock is not held at that point, so
    there is no deadlock). Mirrors the copy in tests/test_reranker.py — the
    two boundaries are mirrors, and so is their coverage."""

    def __init__(self, real, worker_name: str, bump) -> None:
        self._real = real
        self._worker_name = worker_name
        self._bump = bump
        self.fired = False

    def __enter__(self):
        if not self.fired and threading.current_thread().name == self._worker_name:
            self.fired = True
            self._bump()
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)


async def test_a_device_switch_racing_the_load_does_not_cost_a_second_load():
    """The worker read the generation, then the loader read it AGAIN. A
    ``set_device`` landing between the two made the loader publish under the
    NEW generation while the worker recorded the OLD one — so the freshly
    loaded, already-correct model was thrown away and reloaded from scratch on
    the very next embed (minutes, for ~570MB on a cold cache), while ``/healthz`` reported
    ready throughout. One snapshot, taken once by the worker, removes the
    second read entirely: pre-fix this test sees two loads, post-fix one."""
    loads: list[str] = []

    class StubSentenceTransformer:
        def __init__(self, model_id: str, trust_remote_code: bool = True, device=None):
            loads.append(device)

        def encode(self, texts: list[str]):
            return np.array([[0.0, 1.0, 2.0, 3.0] for _ in texts])

    with (
        patch("sentence_transformers.SentenceTransformer", StubSentenceTransformer),
        patch("noesis.core.compute.resolve_device", side_effect=lambda d: d or "cpu"),
    ):
        embedder = LocalSTEmbedder(dim=4)
        embedder._lock = _BumpDeviceOnFirstWorkerLock(
            embedder._lock, "noesis-embedder", lambda: embedder.set_device("cuda")
        )
        await embedder.embed_query("q")
        await embedder.embed_query("q")
        assert embedder._lock.fired, "the race was never driven — test is vacuous"
        assert loads == ["cuda"], (
            f"expected one load on the switched-to device, got {loads}"
        )
        assert embedder.resolved_device == "cuda"
        embedder.close()
