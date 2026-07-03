"""Tests for the SQLite state store (M1 spine)."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from code_index.core import state


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "state.sqlite"


@pytest.fixture()
def conn(db_path):
    conn = state.connect(db_path)
    state.init_db(conn)
    yield conn
    conn.close()


def test_connect_enables_wal(conn):
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_wal_concurrent_read_during_write(db_path, tmp_path):
    """Doc-mandated M1 test: reads proceed while another connection writes."""
    setup = state.connect(db_path)
    state.init_db(setup)
    project_id = state.register_project(setup, tmp_path, "fake-model")
    setup.close()

    n_rows = 200
    writer_started = threading.Event()
    errors: list[Exception] = []

    def writer():
        wconn = state.connect(db_path)
        try:
            for i in range(n_rows):
                state.upsert_file(wconn, project_id, f"src/f{i}.py", f"hash{i}")
                writer_started.set()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
            writer_started.set()
        finally:
            wconn.close()

    t = threading.Thread(target=writer)
    t.start()
    assert writer_started.wait(timeout=10)

    rconn = state.connect(db_path)
    try:
        while t.is_alive():
            rows = rconn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            assert rows >= 0
    finally:
        t.join(timeout=30)
        assert not errors, f"writer failed: {errors}"
        final = rconn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        assert final == n_rows
        rconn.close()


def test_register_project_idempotent(conn, tmp_path):
    first = state.register_project(conn, tmp_path, "fake-model")
    second = state.register_project(conn, tmp_path, "fake-model")
    assert first == second
    assert len(state.list_projects(conn)) == 1


def test_register_project_model_mismatch_raises(conn, tmp_path):
    state.register_project(conn, tmp_path, "fake-model")
    with pytest.raises(ValueError, match="mixed-model"):
        state.register_project(conn, tmp_path, "other-model")


def test_get_project_roundtrip(conn, tmp_path):
    project_id = state.register_project(conn, tmp_path, "fake-model")
    row = state.get_project(conn, project_id)
    assert row is not None
    assert row["root_path"] == str(tmp_path.resolve())
    assert row["last_indexed_commit"] is None
    assert state.get_project(conn, "missing") is None


def test_upsert_and_file_states(conn, tmp_path):
    project_id = state.register_project(conn, tmp_path, "fake-model")
    state.upsert_file(conn, project_id, "a.py", "h1", language="python", chunk_count=3)
    assert state.get_file_states(conn, project_id) == {"a.py": "h1"}

    state.upsert_file(conn, project_id, "a.py", "h2", language="python", chunk_count=5)
    assert state.get_file_states(conn, project_id) == {"a.py": "h2"}
    row = conn.execute("SELECT * FROM files WHERE path='a.py'").fetchone()
    assert row["chunk_count"] == 5
    assert row["last_indexed_at"] is not None
    # update, not duplicate insert
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert count == 1


def test_delete_files(conn, tmp_path):
    project_id = state.register_project(conn, tmp_path, "fake-model")
    state.upsert_file(conn, project_id, "a.py", "h1")
    state.upsert_file(conn, project_id, "b.py", "h2")
    state.delete_files(conn, project_id, ["a.py", "missing.py"])
    assert state.get_file_states(conn, project_id) == {"b.py": "h2"}


def test_run_lifecycle(conn, tmp_path):
    project_id = state.register_project(conn, tmp_path, "fake-model")
    run_id = state.start_run(conn, project_id)
    row = conn.execute("SELECT * FROM index_runs WHERE id=?", (run_id,)).fetchone()
    assert row["status"] == "running"
    assert row["started_at"] is not None
    assert row["finished_at"] is None

    state.finish_run(
        conn, run_id, "done", files_total=10, files_changed=2, chunks_written=7
    )
    row = conn.execute("SELECT * FROM index_runs WHERE id=?", (run_id,)).fetchone()
    assert row["status"] == "done"
    assert row["files_total"] == 10
    assert row["files_changed"] == 2
    assert row["chunks_written"] == 7
    assert row["finished_at"] is not None
    assert row["error"] is None


def test_run_status_check_constraint(conn, tmp_path):
    project_id = state.register_project(conn, tmp_path, "fake-model")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO index_runs (id, project_id, status) VALUES ('x', ?, 'bogus')",
            (project_id,),
        )
