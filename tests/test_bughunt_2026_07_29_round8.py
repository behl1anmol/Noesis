"""Regression tests for the PR #24 round-8 review.

Two defects, both about a failure erasing information rather than data:

1. ``state.finish_run`` wrote every column unconditionally. ``execute_run``
   calls it with ``"done"`` and a full set of counters at ``indexer.py:753``,
   then does its remaining bookkeeping — ``add_dirty_paths`` (``BEGIN
   IMMEDIATE``, which raises ``OperationalError`` past the shared connection's
   5s ``busy_timeout``), the anchor write, a ``git status`` subprocess. Any of
   those raising reaches ``except BaseException`` at ``:851``, which calls
   ``finish_run`` again with a status and an error and nothing else — blanking
   every counter the first call had committed. A run whose content was fully
   indexed showed up as failed with no numbers at all.

2. ``_IgnoreStack.push`` swallowed its ``read_text`` failure with a bare
   ``return``. Round 7 claimed ``discovery.py`` had two silent ``except
   OSError`` sites and had fixed both; there were three, and this was the one
   that emitted no trace at all.

Two of the four tests are declared over-correction guards — they pass in both
directions by construction, and exist to catch fixes that overshoot:
``test_finish_run_still_clears_a_stale_error`` (a COALESCE applied to ``error``
too, which would strand a stale message) and
``test_unreadable_gitignore_under_ignores_rather_than_deleting`` (reporting the
``.gitignore`` failure by promoting it to a ``DiscoveryErrors`` entry, which
would convert an under-ignoring problem into a deletion-suppressing one).

The other two fail against ``2c57c5e`` with the symptoms described above.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from noesis.core import state
from noesis.core.discovery import DiscoveryConfig, DiscoveryErrors, discover_files

DISCOVERY_LOGGER = "noesis.core.discovery"


@pytest.fixture()
def conn(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    yield conn
    conn.close()


# --- 1: a second finish_run must not erase the first one's counters -----------


def test_second_finish_run_preserves_counters(conn, tmp_path):
    """The exact shape execute_run produces: a completed run, then a raise
    from the bookkeeping that follows it."""
    project_id = state.register_project(conn, tmp_path, "fake-model")
    run_id = state.start_run(conn, project_id)

    state.finish_run(
        conn,
        run_id,
        "done",
        files_total=12,
        files_changed=3,
        chunks_written=41,
        fast_path_used=True,
        candidate_count=5,
        files_failed=0,
    )
    # add_dirty_paths lost its BEGIN IMMEDIATE race; execute_run's
    # except BaseException handler closes the run with status + error only.
    state.finish_run(conn, run_id, "failed", error="database is locked")

    row = conn.execute("SELECT * FROM index_runs WHERE id=?", (run_id,)).fetchone()
    # The failure is reported...
    assert row["status"] == "failed"
    assert row["error"] == "database is locked"
    # ...without destroying the record of what the run actually did.
    assert row["files_total"] == 12
    assert row["files_changed"] == 3
    assert row["chunks_written"] == 41
    assert row["fast_path_used"] == 1
    assert row["candidate_count"] == 5
    assert row["files_failed"] == 0


def test_finish_run_still_clears_a_stale_error(conn, tmp_path):
    """OVER-CORRECTION GUARD — passes before and after the fix by design.

    `error` is deliberately NOT coalesced: it has to be able to move back to
    NULL, or a run row would carry a previous attempt's message forever. This
    catches an over-broad fix that wraps every column in COALESCE.
    """
    project_id = state.register_project(conn, tmp_path, "fake-model")
    run_id = state.start_run(conn, project_id)

    state.finish_run(conn, run_id, "failed", error="transient")
    state.finish_run(conn, run_id, "done", files_total=1)

    row = conn.execute("SELECT * FROM index_runs WHERE id=?", (run_id,)).fetchone()
    assert row["status"] == "done"
    assert row["error"] is None
    assert row["files_total"] == 1


# --- 2: an unreadable .gitignore must not vanish silently --------------------


def _unreadable_gitignore(monkeypatch, target: Path) -> None:
    """Fail read_text for exactly one path, delegating every other read."""
    resolved = target.resolve()
    real_read_text = Path.read_text

    def flaky(self, *args, **kwargs):
        if self.resolve() == resolved:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".gitignore").write_text("secret_notes.py\n")
    (root / "keep.py").write_text("x = 1\n")
    (root / "secret_notes.py").write_text("y = 2\n")
    return root


def test_unreadable_gitignore_is_reported(tmp_path, monkeypatch, caplog):
    root = _tree(tmp_path)
    _unreadable_gitignore(monkeypatch, root / ".gitignore")

    with caplog.at_level(logging.DEBUG, logger=DISCOVERY_LOGGER):
        discover_files(root, DiscoveryConfig())

    reported = [
        rec
        for rec in caplog.records
        if rec.name == DISCOVERY_LOGGER and ".gitignore" in rec.getMessage()
    ]
    assert reported, "an unreadable .gitignore was skipped with no trace"
    # The errno is the actionable half; assert it survived rather than
    # asserting the sentence it arrived in (lesson #14).
    assert any("Permission denied" in rec.getMessage() for rec in reported)


def test_unreadable_gitignore_under_ignores_rather_than_deleting(tmp_path, monkeypatch):
    """OVER-CORRECTION GUARD — passes before and after the fix by design.

    Documents the consequence that makes this different from every other
    error in the module, and pins the decision not to escalate it: the
    failure must stay out of DiscoveryErrors, or a lost ignore file would
    start suppressing deletions for a directory that was read perfectly well.
    """
    root = _tree(tmp_path)
    _unreadable_gitignore(monkeypatch, root / ".gitignore")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(), errors=errors)

    # Never deletion evidence.
    assert errors.dirs == [], "a lost .gitignore became a directory error"
    assert errors.files == []
    # The actual cost: the ignore rule did not apply, so the file the user
    # excluded is now in the index.
    assert "secret_notes.py" in found
    assert "keep.py" in found
