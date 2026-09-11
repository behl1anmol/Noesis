"""M4 tests for the Reranker boundary — worker thread, lazy load, scoring.

Uses the ``_load_model`` injection seam with a fake in-process CrossEncoder
so the suite never touches the network or downloads bge-reranker-v2-m3,
mirroring the LocalSTEmbedder test approach.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import patch

import pytest

from noesis.core.embedder import LocalSTEmbedder
from noesis.core.reranker import FakeReranker, LocalCrossEncoderReranker, Reranker


class FakePredictModel:
    """Records predict() calls; scores each pair by candidate text length."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self.calls.append(
            {
                "pairs": list(pairs),
                "batch_size": batch_size,
                "thread": threading.current_thread().name,
            }
        )
        return [float(len(text)) for _query, text in pairs]


def make_reranker(model: FakePredictModel, **kwargs) -> LocalCrossEncoderReranker:
    return LocalCrossEncoderReranker(_load_model=lambda: model, **kwargs)


# --- FakeReranker -----------------------------------------------------------


async def test_fake_reranker_satisfies_protocol_and_is_deterministic():
    fake = FakeReranker()
    assert isinstance(fake, Reranker)
    scores = await fake.rerank(
        "validate token", ["def validate_token(): ...", "def connect(dsn): ..."]
    )
    assert scores[0] > scores[1]  # full overlap beats none
    assert scores == await fake.rerank(
        "validate token", ["def validate_token(): ...", "def connect(dsn): ..."]
    )
    assert len(fake.calls) == 2


async def test_fake_reranker_empty_query_scores_zero():
    assert await FakeReranker().rerank("!!!", ["anything"]) == [0.0]


# --- LocalCrossEncoderReranker ----------------------------------------------


async def test_satisfies_reranker_protocol():
    reranker = make_reranker(FakePredictModel())
    assert isinstance(reranker, Reranker)
    assert reranker.model_id == "BAAI/bge-reranker-v2-m3"
    reranker.close()


async def test_scores_are_floats_in_input_order():
    model = FakePredictModel()
    reranker = make_reranker(model)
    scores = await reranker.rerank("q", ["aaaa", "a", "aa"])
    reranker.close()
    assert scores == [4.0, 1.0, 2.0]  # input order, not sorted
    assert all(isinstance(s, float) for s in scores)
    assert model.calls[0]["pairs"] == [("q", "aaaa"), ("q", "a"), ("q", "aa")]


async def test_empty_texts_short_circuit_without_loading():
    loads: list[int] = []

    def loader() -> FakePredictModel:
        loads.append(1)
        return FakePredictModel()

    reranker = LocalCrossEncoderReranker(_load_model=loader)
    assert await reranker.rerank("q", []) == []
    reranker.close()
    assert loads == []  # no work → model never loaded


async def test_lazy_load_and_preload():
    loads: list[int] = []
    model = FakePredictModel()

    def loader() -> FakePredictModel:
        loads.append(1)
        return model

    reranker = LocalCrossEncoderReranker(_load_model=loader)
    assert loads == []  # constructor does not load
    await reranker.preload()
    assert loads == [1]
    await reranker.rerank("q", ["x"])
    reranker.close()
    assert loads == [1]  # loaded exactly once


async def test_runs_on_dedicated_thread_not_the_embedders():
    import numpy as np

    class EmbedModel:
        def __init__(self) -> None:
            self.threads: list[str] = []

        def encode(self, texts):
            self.threads.append(threading.current_thread().name)
            return np.zeros((len(texts), 4))

    embed_model = EmbedModel()
    embedder = LocalSTEmbedder(dim=4, _load_model=lambda: embed_model)
    rerank_model = FakePredictModel()
    reranker = make_reranker(rerank_model)

    await embedder.embed_query("q")
    await reranker.rerank("q", ["x"])
    embedder.close()
    reranker.close()

    assert rerank_model.calls[0]["thread"] == "noesis-reranker"
    assert embed_model.threads[0] == "noesis-embedder"


async def test_batch_size_passed_through():
    model = FakePredictModel()
    reranker = make_reranker(model, batch_size=7)
    await reranker.rerank("q", ["a", "b"])
    reranker.close()
    assert model.calls[0]["batch_size"] == 7


async def test_worker_survives_job_exception():
    inner = FakePredictModel()

    class FirstCallExplodes:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, pairs, batch_size):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("predict exploded")
            return inner.predict(pairs, batch_size)

    reranker = LocalCrossEncoderReranker(_load_model=FirstCallExplodes)
    with pytest.raises(RuntimeError, match="predict exploded"):
        await reranker.rerank("q", ["a"])
    # The SAME worker thread serves the next job successfully.
    assert await reranker.rerank("q", ["bb"]) == [2.0]
    reranker.close()


async def test_close_is_idempotent_and_rejects_new_work():
    reranker = make_reranker(FakePredictModel())
    await reranker.rerank("q", ["x"])
    reranker.close()
    reranker.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        await reranker.rerank("q", ["y"])


def test_close_before_any_work_is_safe():
    make_reranker(FakePredictModel()).close()


async def test_close_is_bounded_while_worker_is_stuck_in_model_load():
    """Shutdown must not hang on ``worker.join()`` while the worker is mid
    model-load (the ~2.3GB cold-cache download can take minutes): the join is
    bounded at 5s and the daemon worker is abandoned. Pre-fix, close() blocked
    until the load finished — here ~60s, the stall cap that also keeps a
    regression from hanging the suite forever."""
    load_started = threading.Event()
    release = threading.Event()
    model = FakePredictModel()

    def stuck_load():
        load_started.set()
        release.wait(timeout=60.0)  # hard cap: regression fails, never hangs
        return model

    reranker = LocalCrossEncoderReranker(_load_model=stuck_load)
    pending = asyncio.ensure_future(reranker.rerank("q", ["held"]))
    assert await asyncio.to_thread(load_started.wait, 5.0)
    started = time.monotonic()
    reranker.close()
    elapsed = time.monotonic() - started
    assert elapsed < 6.0, f"close() blocked {elapsed:.1f}s on a stuck worker"
    release.set()
    await pending  # abandoned worker still finishes the in-flight job


async def test_truncated_pairs_are_logged(caplog):
    class Tokenizer:
        def __call__(self, query: str, text: str, truncation: bool) -> dict:
            assert truncation is False
            return {"input_ids": list(range(len(query) + len(text)))}

    class TruncatingModel(FakePredictModel):
        tokenizer = Tokenizer()
        max_length = 8

    reranker = make_reranker(TruncatingModel())
    with caplog.at_level(logging.WARNING, logger="noesis.core.reranker"):
        # query len 1: "aaaaaaaaaa" → 11 ids > 8 (truncated); "bb" → 3 ids.
        await reranker.rerank("q", ["aaaaaaaaaa", "bb"])
    reranker.close()
    assert "truncated 1/2" in caplog.text


async def test_no_truncation_check_without_tokenizer_surface(caplog):
    reranker = make_reranker(FakePredictModel())  # no tokenizer attribute
    with caplog.at_level(logging.WARNING, logger="noesis.core.reranker"):
        await reranker.rerank("q", ["some very long text " * 100])
    reranker.close()
    assert "truncated" not in caplog.text


# --- resolved_device is a readiness signal, not a "we picked a device" flag --
#
# Issue #52: /healthz's new reranker_ready reads resolved_device, so the same
# premature-assignment pattern ADR-79 fixed in embedder.py (and explicitly
# DECLINED to fix here, because nothing read it then) becomes load-bearing the
# moment this PR wires it up. Both tests below fail against the pre-fix
# reranker.py, which assigned self._resolved_device before CrossEncoder(...)
# ran: the mid-flight assertion saw 'cpu' instead of None, and the failed-load
# assertion saw 'cpu' after an exception that left no model at all.


async def test_resolved_device_stays_none_until_model_load_succeeds():
    """Mirror of tests/test_local_embedder.py's embedder test. Exercises the
    real ``_default_load`` (the ``_load_model`` seam is deliberately NOT used)
    with ``CrossEncoder`` and ``resolve_device`` stubbed, so no weights are
    fetched and no network is touched."""
    started = threading.Event()
    release = threading.Event()

    class StubCrossEncoder:
        def __init__(self, model_id: str, device=None):
            started.set()
            assert release.wait(timeout=5.0), "test never released model"

        def predict(self, pairs, batch_size: int):
            return [0.0 for _ in pairs]

    with (
        patch("sentence_transformers.CrossEncoder", StubCrossEncoder),
        patch("noesis.core.compute.resolve_device", return_value="cpu"),
    ):
        reranker = LocalCrossEncoderReranker()
        assert reranker.resolved_device is None
        pending = asyncio.ensure_future(reranker.rerank("q", ["a"]))
        assert await asyncio.to_thread(started.wait, 5.0)
        # Constructor is mid-flight — a ~2.3GB load that can run for minutes.
        # /healthz must still read this as not-ready.
        assert reranker.resolved_device is None
        release.set()
        await pending
        assert reranker.resolved_device == "cpu"
        reranker.close()


async def test_resolved_device_stays_none_after_failed_load():
    """A load that raises (interrupted download, OOM) must leave the reranker
    reporting not-ready forever, not truthy — pre-fix the assignment ran
    before the constructor that raised, so /healthz would have reported
    ``reranker_ready: true`` for a model that never existed."""

    class ExplodingCrossEncoder:
        def __init__(self, model_id: str, device=None):
            raise OSError("simulated interrupted download")

    with (
        patch("sentence_transformers.CrossEncoder", ExplodingCrossEncoder),
        patch("noesis.core.compute.resolve_device", return_value="cpu"),
    ):
        reranker = LocalCrossEncoderReranker()
        with pytest.raises(OSError, match="simulated interrupted download"):
            await reranker.rerank("q", ["a"])
        assert reranker.resolved_device is None
        reranker.close()


async def test_a_superseded_load_does_not_publish_its_device(caplog):
    """PR review of issue #52: deferring the assignment until after the
    constructor returns opened a race with ``set_device`` (ADR-40).

    The dashboard can retarget the device while a load is in flight — and with
    the new startup warm-up that window is now minutes wide on a cold cache.
    ``set_device`` bumps the generation and clears ``resolved_device``; the
    finishing OLD-generation load must not write its device back over that,
    or ``/healthz`` reports ``reranker_ready: true`` for a model the worker is
    about to throw away and reload.
    """
    started = threading.Event()
    release = threading.Event()
    caplog.set_level(logging.INFO, logger="noesis.core.reranker")

    class StubCrossEncoder:
        def __init__(self, model_id: str, device=None):
            started.set()
            assert release.wait(timeout=5.0), "test never released model"

        def predict(self, pairs, batch_size: int):
            return [0.0 for _ in pairs]

    with (
        patch("sentence_transformers.CrossEncoder", StubCrossEncoder),
        patch("noesis.core.compute.resolve_device", side_effect=lambda d: d or "cpu"),
    ):
        reranker = LocalCrossEncoderReranker()
        pending = asyncio.ensure_future(reranker.rerank("q", ["a"]))
        assert await asyncio.to_thread(started.wait, 5.0)
        # Operator switches device mid-load.
        reranker.set_device("cuda")
        assert reranker.resolved_device is None
        release.set()
        await pending
        assert reranker.resolved_device is None, (
            "a superseded load published its device — health would report ready "
            "for a model the worker is about to reload"
        )
        # The completion log must still name the device the load actually ran
        # on. Reading the (now correctly empty) attribute instead printed
        # "ready on None", which reads like a load that resolved nothing.
        ready_lines = [
            r.getMessage() for r in caplog.records if "ready on" in r.getMessage()
        ]
        assert ready_lines, "no completion log line at all"
        assert "None" not in ready_lines[-1], ready_lines[-1]
        assert "cpu" in ready_lines[-1]
        reranker.close()


class _BumpDeviceOnFirstWorkerLock:
    """Drives a ``set_device`` into the window between the worker loop's
    generation read and the loader's own — a gap of two adjacent statements,
    unreachable by timing, so it is driven deterministically: the first time
    the MODEL WORKER thread takes the lock, switch the device first.

    Wrapping the lock rather than patching a method keeps the production code
    path intact; ``set_device`` re-enters this wrapper, which passes straight
    through after the first hit (the real lock is not held at that point, so
    there is no deadlock)."""

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
    the very next rerank (minutes, for ~2.3GB), while ``/healthz`` reported
    ready throughout. One snapshot, taken once by the worker, removes the
    second read entirely: pre-fix this test sees two loads, post-fix one."""
    loads: list[str] = []

    class StubCrossEncoder:
        def __init__(self, model_id: str, device=None):
            loads.append(device)

        def predict(self, pairs, batch_size: int):
            return [0.0 for _ in pairs]

    with (
        patch("sentence_transformers.CrossEncoder", StubCrossEncoder),
        patch("noesis.core.compute.resolve_device", side_effect=lambda d: d or "cpu"),
    ):
        reranker = LocalCrossEncoderReranker()
        reranker._lock = _BumpDeviceOnFirstWorkerLock(
            reranker._lock, "noesis-reranker", lambda: reranker.set_device("cuda")
        )
        await reranker.rerank("q", ["a"])
        await reranker.rerank("q", ["a"])
        assert reranker._lock.fired, "the race was never driven — test is vacuous"
        assert loads == ["cuda"], (
            f"expected one load on the switched-to device, got {loads}"
        )
        assert reranker.resolved_device == "cuda"
        reranker.close()
