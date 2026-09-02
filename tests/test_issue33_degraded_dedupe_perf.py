"""Issue #33: `_record_degraded`'s dedupe must be O(1) per call, not O(N).

`_record_degraded` used to dedupe a directory key by rescanning the whole
bucket list on every call (``any(existing == key for existing, _ in
bucket)``), so recording N distinct degraded directories in one walk cost
O(N^2) — measured in the PR #31 round-2 review at 0.02s/1000 dirs,
0.30s/5000, 1.31s/10000. The fix threads a companion ``set`` (``seen``)
through every call site that shares one bucket for the life of a walk, so
the membership check is O(1) and the whole sequence of N calls is O(N).

These tests exercise `_record_degraded` directly rather than through
`discover_files` + a real filesystem walk, so the measurement isolates the
dedupe cost itself from unrelated I/O overhead (mkdir, stat, os.walk).
"""

from __future__ import annotations

import time

from noesis.core.discovery import _record_degraded


def _record_n_distinct(n: int) -> tuple[list[tuple[str, str]], float]:
    bucket: list[tuple[str, str]] = []
    seen: set[str] = set()
    exc = OSError(13, "Permission denied", "/root/dir")
    start = time.perf_counter()
    for i in range(n):
        _record_degraded(bucket, f"dir{i}", f"/root/dir{i}", exc, "summary", seen=seen)
    elapsed = time.perf_counter() - start
    return bucket, elapsed


def test_record_degraded_still_dedupes_one_row_per_key():
    """The O(1) rewrite must preserve the original dedupe contract exactly:
    a second fault on an already-recorded key adds nothing."""
    bucket: list[tuple[str, str]] = []
    seen: set[str] = set()
    exc = OSError(13, "Permission denied", "/root/pkg")

    _record_degraded(bucket, "pkg", "/root/pkg", exc, "first", seen=seen)
    _record_degraded(bucket, "pkg", "/root/pkg", exc, "second", seen=seen)
    _record_degraded(bucket, "other", "/root/other", exc, "third", seen=seen)

    assert [key for key, _ in bucket] == ["pkg", "other"]
    # The first fault recorded on a key wins; a later one on the same key is
    # dropped entirely, never merged or replaced.
    assert bucket[0][1].startswith("first")


def test_record_degraded_dedupe_is_not_quadratic():
    """Recording 8x as many distinct keys must not cost ~64x as long.

    Pure O(N^2) rescanning (the pre-fix behavior) grows quadratically: 8x the
    keys costs roughly 8^2 = 64x the time. O(1)-per-call dedupe grows
    linearly: 8x the keys costs roughly 8x the time. The threshold below
    (20x) sits with a wide margin above the ~8x a linear implementation
    should show and a wide margin below the ~64x a quadratic one would show,
    so it tolerates real timing noise without masking a regression back to
    O(N^2).
    """
    small_n, large_n = 2_000, 16_000  # 8x

    # Warm up interpreter/JIT-ish effects (attribute lookups, etc.) so the
    # first measurement isn't penalized relative to the second.
    _record_n_distinct(200)

    _, small_elapsed = _record_n_distinct(small_n)
    _, large_elapsed = _record_n_distinct(large_n)

    assert large_elapsed < small_elapsed * 20, (
        f"{small_n} keys took {small_elapsed:.4f}s, {large_n} keys "
        f"({large_n // small_n}x as many) took {large_elapsed:.4f}s "
        f"({large_elapsed / small_elapsed:.1f}x) — dedupe looks quadratic again"
    )
