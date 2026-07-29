"""Regression tests for the PR #24 round-7 review.

Round 7 raised four items; only one was a code defect. `discovery.py` still
held a silent `except OSError` — the `os.stat` that seeds the `follow_symlinks`
cycle guard — in the one module ADR-51 rewrote specifically so that no
filesystem fault vanishes unrecorded. A second site of the same class (the
per-child stat in the `dirnames` loop) was found while fixing the first.

Two things are pinned here, and they pull in opposite directions on purpose:

1. The failure is *reported*. Both sites now go through `_cycle_guard_identity`,
   which logs the path and the errno at DEBUG.
2. The failure is *not recorded as evidence*. It must never reach
   `DiscoveryErrors.dirs`: `os.walk` already scandir'd that directory
   successfully and its files are in the result, so routing it there would
   suppress deletion and orphan pruning for a fully-walked subtree (ADR-51,
   narrowed by ADR-54) on a question the stat never answered. An unstattable
   child directory is still walked, for the same reason — skipping it would
   hide files from the deletion path, which is purge-on-doubt.

Test discipline (lesson #14): the two `*_is_reported` tests fail against
`27daf5f`, where neither site logged anything. The two guards below them are
declared over-correction guards and pass in both directions by construction —
they exist to catch a fix that "reports" the failure by promoting it to a
`dirs` entry, which would trade a cosmetic guard degradation for a deletion
freeze.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from noesis.core.discovery import DiscoveryConfig, DiscoveryErrors, discover_files

DISCOVERY_LOGGER = "noesis.core.discovery"


def _fail_stat_on(monkeypatch, target: Path) -> None:
    """Make `os.stat` raise EACCES for exactly one path, delegating the rest.

    A blanket monkeypatch is not usable here: `os` is a shared module object,
    so `Path.stat()` in discovery's file-screening loop resolves to the same
    attribute, and breaking it would fail the walk for reasons unrelated to the
    cycle guard.

    `follow_symlinks=False` calls are delegated for the same reason. That is
    `Path.lstat()`, which reaches `os.stat` through a different door —
    `Path.is_symlink()` at `discovery.py:300` — and CPython's `is_symlink`
    re-raises anything `_ignore_error` does not cover, EACCES included. Failing
    it here would make these tests assert on a pre-existing crash path that has
    nothing to do with the cycle guard. The guard itself calls plain
    `os.stat(path)`, so targeting only that is both narrower and truer.
    """
    resolved = target.resolve()
    real_stat = os.stat

    def flaky(path, *args, **kwargs):
        if kwargs.get("follow_symlinks", True) is False:
            return real_stat(path, *args, **kwargs)
        try:
            same = Path(path) == resolved
        except TypeError:  # an fd or a bytes path — never our target
            same = False
        if same:
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "mod.py").write_text("y = 2\n")
    return root


# --- the failure is reported (fails against 27daf5f) ---------------------------


def test_walk_dir_cycle_guard_failure_is_reported(tmp_path, monkeypatch, caplog):
    """The walk-directory site. Failing the stat on the root reaches only this
    site: the root is never a child in the `dirnames` loop, so nothing else can
    account for the record."""
    root = _tree(tmp_path)
    _fail_stat_on(monkeypatch, root)

    with caplog.at_level(logging.DEBUG, logger=DISCOVERY_LOGGER):
        found = discover_files(root, DiscoveryConfig(follow_symlinks=True))

    reported = [
        rec
        for rec in caplog.records
        if rec.name == DISCOVERY_LOGGER and str(root) in rec.getMessage()
    ]
    assert reported, "the cycle-guard stat failure went unreported"
    # The errno is the part an operator can act on, so assert it survived
    # rather than asserting the sentence it arrived in.
    assert any("Permission denied" in rec.getMessage() for rec in reported)
    # Still a working walk: the guard degraded, the scan did not.
    assert "top.py" in found


def test_child_dir_cycle_guard_failure_is_reported(tmp_path, monkeypatch, caplog):
    """The per-child site in the `dirnames` loop — the second one, missed by
    the review and fixed in the same change."""
    root = _tree(tmp_path)
    _fail_stat_on(monkeypatch, root / "pkg")

    with caplog.at_level(logging.DEBUG, logger=DISCOVERY_LOGGER):
        found = discover_files(root, DiscoveryConfig(follow_symlinks=True))

    reported = [
        rec
        for rec in caplog.records
        if rec.name == DISCOVERY_LOGGER and str(root / "pkg") in rec.getMessage()
    ]
    assert reported, "the child cycle-guard stat failure went unreported"
    assert any("Permission denied" in rec.getMessage() for rec in reported)
    assert "pkg/mod.py" in found


# --- over-correction guards (pass both ways, by construction) ------------------


def test_cycle_guard_failure_is_never_deletion_evidence(tmp_path, monkeypatch):
    """The whole point of the carve-out. A `dirs` entry here would suppress
    deletion and orphan pruning for a subtree `os.walk` read perfectly well,
    and block the git anchor — ADR-51's machinery aimed at the wrong question.
    """
    root = _tree(tmp_path)
    _fail_stat_on(monkeypatch, root / "pkg")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(follow_symlinks=True), errors=errors)

    assert errors.dirs == [], "a cycle-guard stat failure became a directory error"
    assert errors.files == []
    # Both files remain provable, so a genuine deletion elsewhere stays provable.
    assert sorted(found) == ["pkg/mod.py", "top.py"]
    assert errors.entries_seen == 2


def test_unstattable_child_directory_is_still_walked(tmp_path, monkeypatch):
    """Skipping it would hide `pkg/mod.py` from discovery, and everything
    downstream reads absence as deletion. Walking it unguarded is the
    fail-safe direction (ADR-51)."""
    root = _tree(tmp_path)
    _fail_stat_on(monkeypatch, root / "pkg")

    found = discover_files(root, DiscoveryConfig(follow_symlinks=True))

    assert "pkg/mod.py" in found


def test_cycle_guard_is_inert_without_follow_symlinks(tmp_path, monkeypatch):
    """Neither site runs when `follow_symlinks` is False (the default), so a
    stat failure that only the guard would have seen stays entirely absent —
    no log, no error, no behavior change for the common configuration."""
    root = _tree(tmp_path)
    _fail_stat_on(monkeypatch, root / "pkg")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(), errors=errors)

    assert sorted(found) == ["pkg/mod.py", "top.py"]
    assert errors.dirs == [] and errors.files == []
