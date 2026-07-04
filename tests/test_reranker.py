"""M4 tests for the Reranker boundary — worker thread, lazy load, scoring.

Uses the ``_load_model`` injection seam with a fake in-process CrossEncoder
so the suite never touches the network or downloads bge-reranker-v2-m3,
mirroring the LocalSTEmbedder test approach.
"""

from __future__ import annotations

import logging
import threading

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
