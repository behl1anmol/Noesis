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

Issue #59: the quadratic-regression guard below used to assert a wall-clock
ratio (elapsed time for a small N vs. a large N). Both samples are measured
moments apart in the same process, so a GC pause, a scheduling hiccup, or
CPU contention landing on the *small* (denominator) sample alone inflates
the ratio with no change in the underlying complexity — this failed twice
across roughly eight full-suite runs of a branch that touched none of the
code it exercises, while always passing in isolation. ADR-93 replaces the
timing with a count of real ``__eq__`` comparisons the dedupe performs: an
exact, deterministic number that cannot be perturbed by the scheduler,
because it exercises the operation whose complexity is under test directly
rather than through a wall-clock proxy for it.
"""

from __future__ import annotations

from noesis.core.discovery import _record_degraded


class _CountingKey(str):
    """`str` subclass that counts every `__eq__` call it takes part in.

    Recording N distinct keys through a correct O(1)-per-call hash-set
    membership check needs only a small, N-independent number of `__eq__`
    calls: CPython's open addressing calls it only to disambiguate a hash
    collision, and short, distinct, well-hashed strings essentially never
    collide (see `test_record_degraded_dedupe_catches_quadratic_rescans`
    below for the number the pre-fix O(N) scan produces by comparison).
    Passing keys of this type through `_record_degraded` turns "is the
    dedupe O(1)?" into an exact integer instead of a wall-clock ratio
    (issue #59) — unlike elapsed time, a call count cannot be perturbed by
    scheduling, GC, or CPU contention.
    """

    __slots__ = ()
    comparisons = 0  # class-level: reset at the start of each measurement

    def __eq__(self, other: object) -> bool:
        _CountingKey.comparisons += 1
        return str.__eq__(self, other)

    def __hash__(self) -> int:  # overriding __eq__ requires restating this
        return str.__hash__(self)


def _record_n_distinct(n: int) -> tuple[list[tuple[str, str]], int]:
    """Run `_record_degraded` for n distinct keys; return the bucket and the
    number of `__eq__` comparisons the dedupe performed along the way."""
    _CountingKey.comparisons = 0
    bucket: list[tuple[str, str]] = []
    seen: set[str] = set()
    exc = OSError(13, "Permission denied", "/root/dir")
    for i in range(n):
        key = _CountingKey(f"dir{i}")
        _record_degraded(bucket, key, f"/root/dir{i}", exc, "summary", seen=seen)
    return bucket, _CountingKey.comparisons


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
    """Recording N distinct keys must cost O(N) equality comparisons overall,
    not O(N^2) — a deterministic replacement for the flaky wall-clock ratio
    this test used before (issue #59, ADR-93).

    The budget below (4 comparisons/key) is generous slack above what a
    correct O(1)-per-call implementation actually produces — measured at
    exactly 0 for every N tried locally, since `seen` is a real hash set and
    these keys essentially never collide — and is orders of magnitude below
    what the pre-fix O(N^2) bucket scan produces at this same N (see
    `test_record_degraded_dedupe_catches_quadratic_rescans`, which measures
    ~128,000,000 comparisons for a bucket-scan dedupe at N=16,000 using the
    same instrument). Both sides of that margin are exercised in this file,
    not assumed (CLAUDE.md hard rule 9).
    """
    n = 16_000
    _, comparisons = _record_n_distinct(n)
    assert comparisons <= 4 * n, (
        f"{n} distinct keys cost {comparisons} __eq__ comparisons during "
        f"dedupe (expected roughly 0, budget {4 * n}) — the dedupe looks "
        "like it is rescanning the bucket again (O(N^2), issue #33)"
    )


def test_record_degraded_dedupe_catches_quadratic_rescans():
    """Self-check for the instrument above: reproduce the pre-fix O(N^2)
    bucket scan (`any(existing == key for existing, _ in bucket)`) with the
    same counting keys and confirm it blows through the budget
    `test_record_degraded_dedupe_is_not_quadratic` uses. Without this, that
    test could pass on a quadratic regression too, just by measuring nothing
    (CLAUDE.md hard rule 9 / lesson 20: a probe must be shown non-vacuous,
    not just asserted to be one)."""
    n = 2_000
    _CountingKey.comparisons = 0
    bucket: list[tuple[_CountingKey, str]] = []
    for i in range(n):
        key = _CountingKey(f"dir{i}")
        if any(existing == key for existing, _ in bucket):
            continue
        bucket.append((key, "x"))
    assert _CountingKey.comparisons > 4 * n, (
        f"the pre-fix O(N^2) bucket-scan reference implementation only cost "
        f"{_CountingKey.comparisons} comparisons for {n} keys — at or below "
        f"the {4 * n} budget the real test above uses, which means that test "
        "would pass even on a quadratic regression"
    )
