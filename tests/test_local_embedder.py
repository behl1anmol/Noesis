"""M2 tests for LocalSTEmbedder — worker thread, priority queue, prefix.

Uses the ``_load_model`` injection seam with a fake in-process model so the
suite never touches the network or downloads CodeRankEmbed.
"""

from __future__ import annotations

import asyncio
import threading

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


def test_constructor_defaults():
    embedder = LocalSTEmbedder()
    assert embedder.model_id == "nomic-ai/CodeRankEmbed"
    assert embedder.dim == 768
    embedder.close()  # no worker ever started; must not hang
