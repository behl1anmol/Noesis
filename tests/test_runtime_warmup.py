"""Background model warm-up wiring in ``build_runtime_context`` (ADR-77 for
the embedder, issue #52 for the reranker).

The reranker loads lazily on its first ``rerank`` call unless
``reranker.preload=true``, so an operator who turned reranking on paid a cold
~2.3GB load on the first reranked search — the same silent stall ADR-77 fixed
for the embedder, on a model an order of magnitude larger. This file pins the
wiring: a warm-up task exists exactly when it should, it runs AFTER the
embedder's (the embedder gates every search; the reranker only reranks one
that already ran), and teardown cancels and awaits it.

No real model, client or server is touched: the four things
``build_runtime_context`` constructs are patched at their import site in
``noesis.runtime``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading

import pytest

from noesis.core import state
from noesis.core.config import Settings
from noesis.runtime import AppContext, build_runtime_context, close_runtime_context


class _StubEmbedder:
    def __init__(self, *args, **kwargs) -> None:
        self.model_id = kwargs.get("model_id", "stub-embedder")
        self.dim = kwargs.get("dim", 8)
        self.preload_calls = 0
        self.gate: threading.Event | None = None
        _StubEmbedder.last = self

    async def preload(self) -> None:
        self.preload_calls += 1
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait, 5.0)

    def close(self) -> None:
        pass


class _StubReranker:
    def __init__(self, *args, **kwargs) -> None:
        self.model_id = kwargs.get("model_id", "stub-reranker")
        self.preload_calls = 0
        self.preloaded = asyncio.Event()
        _StubReranker.last = self

    async def preload(self) -> None:
        self.preload_calls += 1
        self.preloaded.set()

    def close(self) -> None:
        pass


class _StubStore:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def ensure_collection(self, embedder) -> bool:
        return False

    def delete_orphan_points(self, project_ids) -> int:
        return 0

    def close(self) -> None:
        pass


class _StubQdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture()
def patched_runtime(monkeypatch):
    monkeypatch.setattr("noesis.runtime.LocalSTEmbedder", _StubEmbedder)
    monkeypatch.setattr("noesis.runtime.LocalCrossEncoderReranker", _StubReranker)
    monkeypatch.setattr("noesis.runtime.VectorStore", _StubStore)
    monkeypatch.setattr("noesis.runtime.QdrantClient", _StubQdrantClient)


def _settings(tmp_path, **reranker_overrides) -> Settings:
    cfg = Settings()
    return dataclasses.replace(
        cfg,
        db_path=tmp_path / "state.sqlite",
        reranker=dataclasses.replace(cfg.reranker, **reranker_overrides),
    )


async def test_reranker_warm_up_runs_when_enabled_without_preload(
    patched_runtime, tmp_path
):
    ctx = await build_runtime_context(_settings(tmp_path, enabled=True))
    try:
        assert ctx.reranker_warmup is not None
        await asyncio.wait_for(ctx.reranker_warmup, timeout=5.0)
        assert ctx.reranker.preload_calls == 1
    finally:
        await close_runtime_context(ctx)


async def test_no_reranker_warm_up_when_reranking_is_disabled(
    patched_runtime, tmp_path
):
    # The shipped default. The kill switch means there is no model to warm.
    ctx = await build_runtime_context(_settings(tmp_path, enabled=False))
    try:
        assert ctx.reranker is None
        assert ctx.reranker_warmup is None
    finally:
        await close_runtime_context(ctx)


async def test_no_second_warm_up_when_preload_already_loaded_the_model(
    patched_runtime, tmp_path
):
    # reranker.preload=true already awaits the load inline; a background task
    # on top would be a second, pointless load request behind it.
    ctx = await build_runtime_context(_settings(tmp_path, enabled=True, preload=True))
    try:
        assert ctx.reranker.preload_calls == 1
        assert ctx.reranker_warmup is None
    finally:
        await close_runtime_context(ctx)


async def test_reranker_warm_up_waits_for_the_embedder_warm_up(
    patched_runtime, monkeypatch, tmp_path
):
    """The embedder is on every search's critical path; the reranker only
    scores results a search already produced. Loading both at once makes the
    load that gates the first search slower for no gain, so the reranker's
    warm-up starts only once the embedder's has finished."""
    gate = threading.Event()

    class _GatedEmbedder(_StubEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gate = gate

    monkeypatch.setattr("noesis.runtime.LocalSTEmbedder", _GatedEmbedder)
    ctx = await build_runtime_context(_settings(tmp_path, enabled=True))
    try:
        # The embedder warm-up is parked inside preload(); the reranker's must
        # not have started. Yield generously so a wrongly-concurrent task would
        # have had every chance to run.
        for _ in range(20):
            await asyncio.sleep(0)
        assert ctx.reranker.preload_calls == 0, (
            "reranker warm-up started while the embedder was still loading"
        )
        gate.set()
        await asyncio.wait_for(ctx.reranker_warmup, timeout=5.0)
        assert ctx.reranker.preload_calls == 1
    finally:
        gate.set()
        await close_runtime_context(ctx)


async def test_close_runtime_context_cancels_and_awaits_the_reranker_warm_up(
    tmp_path,
):
    """Teardown must not leave the warm-up task pending — same treatment
    ``embedder_warmup`` gets, and for the same reason: it holds a reference to
    the reranker this function is about to close."""
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    started = asyncio.Event()

    async def never_finishes() -> None:
        started.set()
        await asyncio.Event().wait()

    ctx = AppContext(conn=conn, store=_StubStore(), embedder=_StubEmbedder())
    ctx.reranker_warmup = asyncio.create_task(never_finishes())
    await started.wait()

    await close_runtime_context(ctx)

    assert ctx.reranker_warmup.done()
    assert ctx.reranker_warmup.cancelled()
