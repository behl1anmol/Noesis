"""Embedding boundary for the code indexer.

All embedding calls in this codebase go through the ``Embedder`` Protocol
defined here (CLAUDE.md hard rule 1). Implementations are local-only —
remote/hosted embedding was rejected (ADR-25) and the Protocol deliberately
exposes no credentials or transport surface.

``model_id`` is the versioning key (§3.4 rule 2): it is written to the Qdrant
payload and SQLite ``projects.embedding_model``, and changing implementations
triggers a full re-embed. Nothing outside this boundary may know the vector
size except by reading ``dim`` (§3.4 rule 1). Query-instruction quirks (e.g.
CodeRankEmbed's query prefix) live inside implementations' ``embed_query``
(§3.4 rule 3) — callers never apply prefixes themselves.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import itertools
import logging
import queue
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Async embedding interface. Local implementations only (ADR-25)."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """Deterministic in-memory Embedder for tests (M1 deliverable).

    Vectors are derived from sha256 of the input text, so equal texts embed
    identically and the suite needs no model download. ``embed_query`` hashes
    a ``"query:"``-prefixed variant of the text, mirroring the instruction-
    prefix seam of real implementations: a query embedding of some text is
    never equal to its document embedding, so tests can catch prefix misuse.
    """

    def __init__(self, dim: int = 8, model_id: str = "fake-embedder-v1") -> None:
        self._dim = dim
        self._model_id = model_id
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        # Stream sha256 blocks keyed by text + block index until dim bytes
        # are available; map each byte to [-1, 1].
        raw = bytearray()
        block = 0
        while len(raw) < self._dim:
            raw.extend(hashlib.sha256(f"{block}:{text}".encode()).digest())
            block += 1
        return [b / 255.0 * 2.0 - 1.0 for b in raw[: self._dim]]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(f"query:{text}")


# Priority classes for the LocalSTEmbedder worker queue (§3.3): queries
# preempt indexing batches; freshness lags under query load and recovers.
_HIGH = 0  # embed_query — interactive path
_LOW = 1  # embed_documents — indexing path
_SHUTDOWN = 2  # close() sentinel — drains queued jobs first


class LocalSTEmbedder:
    """Default local Embedder: CodeRankEmbed via sentence-transformers (M2).

    Concurrency model (§3.3): one dedicated single worker thread owns the
    model — forward passes are never concurrent. Jobs go through a
    ``queue.PriorityQueue`` of ``(priority, seq, job)``: HIGH for
    ``embed_query``, LOW for ``embed_documents``, so an interactive query
    preempts queued indexing batches. ``seq`` is a monotonic counter that
    keeps FIFO order within a priority class and keeps tuple comparison
    away from job objects.

    The worker thread starts lazily on the first embed call (daemon, so
    never calling ``close()`` is safe) and the model loads inside the
    worker on its first job. ``sentence_transformers`` is imported lazily
    in the loader — never at module top level — so FakeEmbedder users
    don't pay the torch import; this module is the only place allowed to
    import it (CLAUDE.md hard rule 1).

    CodeRankEmbed's query instruction prefix is applied inside
    ``embed_query`` only (§3.4 rule 3) — callers never see it.
    """

    _QUERY_PREFIX = "Represent this query for searching relevant code: "

    def __init__(
        self,
        model_id: str = "nomic-ai/CodeRankEmbed",
        dim: int = 768,
        device: str | None = None,
        batch_size: int = 32,
        _load_model: Callable[[], Any] | None = None,
    ) -> None:
        self._model_id = model_id
        self._dim = dim
        self._device = device
        self._resolved_device: str | None = None  # set at model load
        self._batch_size = batch_size
        self._load_model = _load_model or self._default_load
        self._queue: queue.PriorityQueue[
            tuple[
                int, int, tuple[Callable[[Any], Any], concurrent.futures.Future] | None
            ]
        ] = queue.PriorityQueue()
        self._seq = itertools.count()
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
    def dim(self) -> int:
        return self._dim

    @property
    def resolved_device(self) -> str | None:
        """The device the model loaded on, or None before the first embed
        (the worker loads lazily on its first job)."""
        return self._resolved_device

    def _default_load(self) -> Any:
        # Lazy import: one of the two sentence_transformers import sites in the
        # codebase (CLAUDE.md hard rule 1, ADR-33; CI greps for this). Local
        # model, no network service — ADR-25.
        from sentence_transformers import SentenceTransformer

        from .compute import resolve_device

        # Resolve the device explicitly rather than passing None and trusting
        # ST's auto-detect, which was seen returning CPU on a cuda box
        # (lesson 4). The resolved value is recorded for benchmark provenance.
        self._resolved_device = resolve_device(self._device)
        # Frame the load: on a cold cache this blocks for minutes downloading
        # weights with no other output (the single silent stall M-users read as
        # a hang). model_id + device only — no code or query text (ADR-25).
        logger.info(
            "loading embedding model %s on %s "
            "(first run may download weights; can take minutes)",
            self._model_id,
            self._resolved_device,
        )
        started = time.perf_counter()
        model = SentenceTransformer(
            self._model_id, trust_remote_code=True, device=self._resolved_device
        )
        logger.info(
            "embedding model %s ready on %s took=%.1fs",
            self._model_id,
            self._resolved_device,
            time.perf_counter() - started,
        )
        return model

    def _worker_loop(self) -> None:
        model: Any = None
        loaded_generation = -1
        while True:
            _priority, _seq, item = self._queue.get()
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

    def _submit(
        self, priority: int, fn: Callable[[Any], Any]
    ) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalSTEmbedder is closed")
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._worker_loop, name="noesis-embedder", daemon=True
                )
                self._worker.start()
            self._queue.put((priority, next(self._seq), (fn, future)))
        return future

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        texts = list(texts)
        batch_size = self._batch_size

        def job(model: Any) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                vectors.extend(v.tolist() for v in model.encode(batch))
            return vectors

        return await asyncio.wrap_future(self._submit(_LOW, job))

    async def embed_query(self, text: str) -> list[float]:
        prefixed = self._QUERY_PREFIX + text

        def job(model: Any) -> list[float]:
            return model.encode([prefixed])[0].tolist()

        return await asyncio.wrap_future(self._submit(_HIGH, job))

    def set_device(self, device: str | None) -> None:
        """Retarget the model's device (dashboard setting, ADR-40); None
        re-enables auto-detect. Takes effect on the worker's next job via a
        generation bump — the single worker thread owns the model, so the
        swap is race-free by construction. In-flight jobs finish on the old
        device."""
        with self._lock:
            if device == self._device:
                return
            self._device = device
            self._generation += 1
            self._resolved_device = None  # unknown until the reload happens

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
            self._queue.put((_SHUTDOWN, next(self._seq), None))
        worker.join()
