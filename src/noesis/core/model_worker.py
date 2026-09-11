"""Shared `set_device`/generation race-fix for the two model-loading worker
threads (ADR-40, issue #52, issue #61).

`LocalSTEmbedder` (embedder.py) and `LocalCrossEncoderReranker`
(reranker.py) each run one dedicated worker thread that lazily loads a
model and must reload it when `set_device` retargets the device mid-flight,
without either publishing a stale device or paying for a duplicate load.
Issue #52's round-3 review found the race this guards: a `set_device`
landing between the worker loop's generation read and a second read inside
the loader made the loader publish under the NEW generation while the
worker loop still recorded the OLD one, so a freshly loaded, already-correct
model was thrown away and reloaded on the very next job, with `/healthz`
reporting ready throughout. Both classes carried the fix as near-identical
duplicate code — same `_generation`/`_load_target` state, same locked
snapshot-then-publish-if-current-generation logic, same explanatory
comments — flagged as a non-blocking maintainability note in issue #52's
round-12 review and filed separately as issue #61 rather than risked inside
that already-hardened PR.

This module owns exactly that state and logic, and nothing else: it builds
no model and imports no ML library, so CLAUDE.md hard rule 1 (ADR-33 — only
embedder.py and reranker.py may import `sentence_transformers`) is
unaffected by its existence (ADR-94). A subclass supplies its own `_lock`
(shared with whatever else it guards — the job queue, `_closed`, `_worker`),
its own loader and worker loop, and its own model construction; this mixin
supplies the `_device`/`_generation`/`_load_target`/`_resolved_device`
fields and the three operations that touch them.
"""

from __future__ import annotations

import threading


class _DeviceGenerationTracker:
    """Mixin: the `set_device`-during-load race fix, factored out of
    `LocalSTEmbedder` and `LocalCrossEncoderReranker` (issue #61).

    A subclass must:
    - create `self._lock: threading.Lock` itself, before calling
      `_init_device_generation` — it guards more than this mixin's state
      (the job queue, `_closed`, `_worker`), so it stays owned by the
      subclass rather than created here.
    - call `_init_device_generation(device)` at the end of its `__init__`.
    - in its worker loop, call `_snapshot_load_target()` immediately before
      deciding whether to (re)run the loader, and use the generation it
      returns as the loop's `loaded_generation` comparison value; the
      loader reads the device half of the same snapshot off
      `self._load_target`.
    - in its loader, immediately after the model constructor returns, call
      `_publish_resolved_device(generation, resolved)` with the generation
      half of the `_load_target` pair that load was for.
    """

    _lock: threading.Lock
    _device: str | None
    _resolved_device: str | None
    _generation: int
    _load_target: tuple[int, str | None]

    def _init_device_generation(self, device: str | None) -> None:
        self._device = device
        self._resolved_device = None  # set at model load
        # Bumped by set_device (ADR-40): the worker reloads the model when
        # its loaded generation falls behind.
        self._generation = 0
        # (generation, device) for the load currently in flight — written by
        # the worker loop before it calls the loader, read by the loader.
        # Both happen on the worker thread, so the pair cannot be torn.
        self._load_target = (0, device)

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

    def _snapshot_load_target(self) -> int:
        """Call once per job, from the worker loop, before deciding whether
        to (re)run the loader. Snapshots the generation AND the device it
        belongs to in one locked read, publishing both to `_load_target` for
        the loader to consume instead of reading `_generation`/`_device`
        again itself: a `set_device` landing between the worker's read and a
        second, separate read inside the loader made the loader publish
        under the NEW generation while this loop recorded the OLD one — so
        the freshly loaded, already-correct model was discarded and reloaded
        on the next job, with health reporting ready throughout (issue #52
        review round 3). Returns the generation, to compare against the
        loop's own `loaded_generation` local."""
        with self._lock:
            generation = self._generation
            self._load_target = (generation, self._device)
        return generation

    def _publish_resolved_device(self, generation: int, resolved: str) -> None:
        """Call from the loader immediately after its model constructor
        returns: publish *resolved* as `resolved_device` iff *generation* is
        still current. A `set_device` (ADR-40) landing mid-load must not
        have the `None` it just wrote overwritten by this now-superseded
        load, or `/healthz` reports ready for a model the worker is about to
        drop and reload."""
        with self._lock:
            if self._generation == generation:
                self._resolved_device = resolved
