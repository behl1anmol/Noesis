"""Reranking boundary for the code indexer (M4, §3.3).

All rerank calls go through the ``Reranker`` Protocol defined here — the
second and last model-loading boundary next to ``core/embedder.py``
(CLAUDE.md hard rule 1, amended by ADR-33). Implementations are local-only
(ADR-25): the Protocol exposes no credentials or transport surface, and this
module is the only place besides the embedder allowed to import
``sentence_transformers``.

Concurrency (§3.3, ADR-20): the cross-encoder runs on its own dedicated
single worker thread, never the embedder's — a rerank of ~50 pairs takes
materially longer than a query embed, and sharing one worker would let a
rerank head-of-line-block the next query's embedding. Every rerank job is
interactive (there is no indexing-path job class on this worker), so a plain
FIFO queue implements the HIGH/LOW discipline vacuously.

``rerank_score`` is response-only (§3.7): nothing here touches the Qdrant
schema or SQLite state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Async reranking interface. Local implementations only (ADR-25).

    ``rerank`` scores each candidate text against the query and returns one
    relevance score per text, same order as the input — reordering is the
    caller's job, so implementations stay a pure scoring function.
    """

    @property
    def model_id(self) -> str: ...

    async def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class FakeReranker:
    """Deterministic in-memory Reranker for tests.

    Scores are lexical-overlap: the fraction of the query's lowercased
    alphanumeric tokens that appear in the candidate text. Tests can steer
    the ordering by controlling token overlap, and equal texts always score
    equally. Records calls for assertion.
    """

    def __init__(self, model_id: str = "fake-reranker-v1") -> None:
        self._model_id = model_id
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @staticmethod
    def _tokens(text: str) -> set[str]:
        # Splits on underscores too, so snake_case identifiers overlap their
        # natural-language spellings ("validate_token" ↔ "validate token").
        out: set[str] = set()
        word: list[str] = []
        for ch in text.lower():
            if ch.isalnum():
                word.append(ch)
            elif word:
                out.add("".join(word))
                word.clear()
        if word:
            out.add("".join(word))
        return out

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        query_tokens = self._tokens(query)
        if not query_tokens:
            return [0.0 for _ in texts]
        return [
            len(query_tokens & self._tokens(text)) / len(query_tokens)
            for text in texts
        ]


# Shutdown sentinel priority — mirrors the embedder's queue shape so the
# worker loop stays structurally identical, even though all rerank jobs
# share one (interactive) priority class.
_JOB = 0
_SHUTDOWN = 1


class LocalCrossEncoderReranker:
    """Default local Reranker: bge-reranker-v2-m3 via sentence-transformers.

    One dedicated single worker thread owns the model — forward passes are
    never concurrent, and the thread is *not* the embedder's (ADR-20). The
    worker starts lazily on the first ``rerank`` call (daemon, so never
    calling ``close()`` is safe) and the ~568M model loads inside the worker
    on its first job — ``reranker.preload=true`` calls :meth:`preload` at
    startup instead. ``sentence_transformers`` is imported lazily in the
    loader, never at module top level.

    Pairs longer than the model's max sequence length are truncated by the
    model; §3.3 says that should be rare (chunks target 300–800 tokens), so
    it is counted and logged per call rather than silently ignored.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
        _load_model: Callable[[], Any] | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._resolved_device: str | None = None  # set at model load
        self._batch_size = batch_size
        self._load_model = _load_model or self._default_load
        self._queue: queue.Queue[
            tuple[int, tuple[Callable[[Any], Any], concurrent.futures.Future] | None]
        ] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._closed = False
        # Bumped by set_device (ADR-40): the worker reloads the model when
        # its loaded generation falls behind.
        self._generation = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def resolved_device(self) -> str | None:
        """The device the model loaded on, or None before the first rerank
        (the worker loads lazily on its first job)."""
        return self._resolved_device

    def _default_load(self) -> Any:
        # Lazy import: allowed only here and in core/embedder.py (CLAUDE.md
        # hard rule 1 as amended by ADR-33; CI greps for this). Local model,
        # no network service — ADR-25.
        from sentence_transformers import CrossEncoder

        from .compute import resolve_device

        # Explicit device resolution, not ST's device=None auto-detect, which
        # was seen running this cross-encoder on CPU with a T4 idle (lesson 4).
        self._resolved_device = resolve_device(self._device)
        return CrossEncoder(self._model_id, device=self._resolved_device)

    def _worker_loop(self) -> None:
        model: Any = None
        loaded_generation = -1
        while True:
            _kind, item = self._queue.get()
            if item is None:  # shutdown sentinel
                return
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                generation = self._generation
                if model is None or loaded_generation != generation:
                    model = None  # drop the old model before loading the new
                    model = self._load_model()
                    loaded_generation = generation
                future.set_result(fn(model))
            except BaseException as exc:  # noqa: BLE001 — propagate to caller,
                future.set_exception(exc)  # never kill the worker thread.

    def set_device(self, device: str | None) -> None:
        """Retarget the model's device (dashboard setting, ADR-40); None
        re-enables auto-detect. Same generation mechanism as the embedder —
        the single worker thread owns the model, so the swap is race-free."""
        with self._lock:
            if device == self._device:
                return
            self._device = device
            self._generation += 1
            self._resolved_device = None  # unknown until the reload happens

    def _submit(self, fn: Callable[[Any], Any]) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalCrossEncoderReranker is closed")
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._worker_loop, name="noesis-reranker", daemon=True
                )
                self._worker.start()
            self._queue.put((_JOB, (fn, future)))
        return future

    def _count_truncated(self, model: Any, query: str, texts: list[str]) -> int:
        """Count pairs exceeding the model's max length. Best-effort: a model
        without a tokenizer/max_length surface (test fakes) counts nothing."""
        tokenizer = getattr(model, "tokenizer", None)
        max_length = getattr(model, "max_length", None)
        if tokenizer is None or not max_length:
            return 0
        truncated = 0
        for text in texts:
            ids = tokenizer(query, text, truncation=False)["input_ids"]
            if len(ids) > max_length:
                truncated += 1
        return truncated

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        texts = list(texts)
        batch_size = self._batch_size

        def job(model: Any) -> list[float]:
            truncated = self._count_truncated(model, query, texts)
            if truncated:
                logger.warning(
                    "reranker truncated %d/%d pairs to the model max length",
                    truncated,
                    len(texts),
                )
            pairs = [(query, text) for text in texts]
            scores = model.predict(pairs, batch_size=batch_size)
            return [float(s) for s in scores]

        return await asyncio.wrap_future(self._submit(job))

    async def preload(self) -> None:
        """Load the model now (``reranker.preload=true``) instead of on the
        first reranked request — a no-op job forces the lazy load."""
        await asyncio.wrap_future(self._submit(lambda model: None))

    def close(self) -> None:
        """Drain queued jobs, then stop the worker. Idempotent; optional —
        the worker is a daemon thread, so never closing is also safe."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            if worker is None:
                return
            self._queue.put((_SHUTDOWN, None))
        worker.join()
