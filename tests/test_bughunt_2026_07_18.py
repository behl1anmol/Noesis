"""Regression tests for the 2026-07-18 bug hunt.

Findings closed here (severity order):

1. ``try_start_run`` returned early on the first alive ``running`` row and
   skipped the dead-owner cleanup, so a crashed co-process's stale row sat
   ``running`` forever whenever a live run coexisted with it — exactly the
   jam ``fail_orphaned_runs`` exists to clear (state.py). Dead rows are now
   failed in the same transaction even when an alive run is returned.
2. ``prepare_run`` opened its run row with ``state.start_run`` directly,
   bypassing the ``try_start_run`` concurrency guard every other launcher
   goes through — a second caller could race a concurrent run onto the same
   collection (indexer.py). It now goes through the guard and raises when a
   live run already holds the project.
"""

from __future__ import annotations

import pytest

from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import prepare_run


def _state_conn(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    return conn


# --- 1. try_start_run fails dead rows even when an alive run exists ----------


def test_try_start_run_fails_dead_rows_even_with_alive_run(tmp_path):
    conn = _state_conn(tmp_path)
    pid = state.register_project(conn, tmp_path, "model")
    alive = state.start_run(conn, pid)  # owner is this live process
    stale = state.start_run(conn, pid)
    # Rewrite the owner to a different boot token: unambiguously dead.
    conn.execute(
        "UPDATE index_runs SET owner = 'not-this-boot:1:1' WHERE id = ?", (stale,)
    )
    conn.commit()

    run_id, created = state.try_start_run(conn, pid)
    # The live run is returned, not a fresh one...
    assert not created and run_id == alive
    # ...and the dead row was still cleaned up, not skipped by the early
    # return — pre-fix it stayed 'running' until the next process restart.
    stale_row = conn.execute(
        "SELECT status, error, finished_at FROM index_runs WHERE id = ?", (stale,)
    ).fetchone()
    assert stale_row["status"] == "failed"
    assert stale_row["error"] == "interrupted"
    assert stale_row["finished_at"] is not None
    alive_row = conn.execute(
        "SELECT status FROM index_runs WHERE id = ?", (alive,)
    ).fetchone()
    assert alive_row["status"] == "running"
    conn.close()


# --- 2. prepare_run goes through the concurrency guard ------------------------


def test_prepare_run_refuses_live_run(tmp_path):
    conn = _state_conn(tmp_path)
    embedder = FakeEmbedder(dim=8)
    # Same model_id as prepare_run will use, so register_project's
    # mixed-model guard cannot fire before the run guard under test.
    pid = state.register_project(conn, str(tmp_path), embedder.model_id)
    state.start_run(conn, pid)  # owner is this live process

    with pytest.raises(RuntimeError, match="already running"):
        prepare_run(conn, embedder, str(tmp_path))
    conn.close()


def test_prepare_run_clean_project_opens_run(tmp_path):
    conn = _state_conn(tmp_path)
    embedder = FakeEmbedder(dim=8)

    project_id, run_id = prepare_run(conn, embedder, str(tmp_path))
    row = conn.execute(
        "SELECT project_id, status FROM index_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    assert row["project_id"] == project_id
    assert row["status"] == "running"
    conn.close()
