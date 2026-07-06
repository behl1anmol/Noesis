"""Regression tests for the 2026-07-06 bug hunt fixes.

Deterministic, model-free coverage (FakeEmbedder + in-memory Qdrant) for the
trickiest fixes: H1 (git fast-path revert), H7 (transient-error vs deletion),
M7 (owner-gated crash recovery). See dev/bug-hunt-2026-07-06.md.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.core import hashdiff, state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import index_project
from noesis.core.vectorstore import VectorStore

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not on PATH"
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        (
            "git",
            "-c", "user.name=test",
            "-c", "user.email=test@test",
            "-c", "commit.gpgsign=false",
            "-C", str(root),
            *args,
        ),
        check=True,
        capture_output=True,
    )


def _env(tmp_path: Path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return conn, store, embedder


# --- H1: reverted-away dirty content is re-examined on the next fast path -----


@requires_git
async def test_h1_reverted_dirty_file_reindexed_on_fast_path(tmp_path: Path) -> None:
    committed = "def alpha():\n    return 1\n"
    dirty = "def alpha():\n    return 999\n"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "alpha.py").write_text(committed)
    (repo / "beta.py").write_text("def beta():\n    return 2\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    conn, store, embedder = _env(tmp_path)

    # Run 1: full walk, anchor recorded, nothing dirty.
    pid = (await index_project(conn, store, embedder, str(repo))).project_id
    committed_hash = hashlib.sha256(committed.encode()).hexdigest()
    assert state.get_file_states(conn, pid)["alpha.py"] == committed_hash

    # Run 2: alpha edited (uncommitted) — status makes it a candidate, so it is
    # indexed with the dirty hash and persisted to the dirty set.
    (repo / "alpha.py").write_text(dirty)
    await index_project(conn, store, embedder, str(repo))
    dirty_hash = hashlib.sha256(dirty.encode()).hexdigest()
    assert state.get_file_states(conn, pid)["alpha.py"] == dirty_hash
    assert "alpha.py" in state.get_dirty_paths(conn, pid)

    # Revert alpha to HEAD content: neither diff nor status now names it.
    (repo / "alpha.py").write_text(committed)

    # Run 3: without the H1 fix the stale dirty hash carries forward forever;
    # with it, the persisted dirty set re-admits alpha, which is re-hashed and
    # detected as changed back to the committed content.
    await index_project(conn, store, embedder, str(repo))
    assert state.get_file_states(conn, pid)["alpha.py"] == committed_hash
    # And it rotates out of the dirty set once clean again.
    assert "alpha.py" not in state.get_dirty_paths(conn, pid)


# --- H7: an unreadable-but-present file is not a deletion --------------------


def test_h7_transient_oserror_preserves_stored_hash(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "live.py").write_text("x\n")
    (tmp_path / "ok.py").write_text("y\n")
    stored = {"live.py": "STORED_HASH", "ok.py": "OLD_HASH"}
    real = hashdiff.hash_file

    def flaky(path):
        if str(path).endswith("live.py"):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(hashdiff, "hash_file", flaky)
    res = hashdiff.partition(tmp_path, ["live.py", "ok.py"], stored)

    # live.py could not be hashed but still exists → never deleted, hash kept.
    assert "live.py" not in res.deleted
    assert "live.py" in res.unchanged
    assert res.hashes["live.py"] == "STORED_HASH"
    # ok.py hashed normally and differs from stored → changed.
    assert "ok.py" in res.changed


def test_h7_missing_file_is_still_a_deletion(tmp_path: Path) -> None:
    stored = {"gone.py": "H"}
    # gone.py is "discovered" but not on disk → FileNotFoundError → deleted.
    res = hashdiff.partition(tmp_path, ["gone.py"], stored)
    assert res.deleted == ("gone.py",)
    assert "gone.py" not in res.hashes


# --- M7: crash recovery only fails runs whose owning process is dead ---------


def _state_conn(tmp_path: Path):
    conn = state.connect(tmp_path / "s.sqlite")
    state.init_db(conn)
    return conn


def test_m7_live_owner_run_is_spared(tmp_path: Path) -> None:
    conn = _state_conn(tmp_path)
    pid = state.register_project(conn, tmp_path, "m")
    run = state.start_run(conn, pid)  # owner is this live process
    assert state.fail_orphaned_runs(conn) == 0
    row = conn.execute("SELECT status FROM index_runs WHERE id=?", (run,)).fetchone()
    assert row["status"] == "running"


def test_m7_dead_owner_run_is_failed(tmp_path: Path) -> None:
    conn = _state_conn(tmp_path)
    pid = state.register_project(conn, tmp_path, "m")
    run = state.start_run(conn, pid)
    conn.execute(
        "UPDATE index_runs SET owner=? WHERE id=?", ("dead-boot:1", run)
    )
    conn.commit()
    assert state.fail_orphaned_runs(conn) == 1
    row = conn.execute(
        "SELECT status, error FROM index_runs WHERE id=?", (run,)
    ).fetchone()
    assert row["status"] == "failed" and row["error"] == "interrupted"


def test_m7_null_owner_row_is_treated_as_dead(tmp_path: Path) -> None:
    conn = _state_conn(tmp_path)
    pid = state.register_project(conn, tmp_path, "m")
    run = state.start_run(conn, pid)
    conn.execute("UPDATE index_runs SET owner=NULL WHERE id=?", (run,))
    conn.commit()
    assert state.fail_orphaned_runs(conn) == 1
