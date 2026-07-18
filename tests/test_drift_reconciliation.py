"""Drift self-heal: SQLite state vs Qdrant reality (2026-07-17 RCA).

Root cause reproduced here: when the Qdrant collection is externally
dropped/recreated (points gone) while the state DB still records every file
as indexed, incremental reindex compared content hashes, saw no change, wrote
nothing, and search returned empty for those files *permanently*. execute_run
now compares an exact project-scoped point count against the stored chunk
total; on mismatch a full run re-embeds the drifted files (idempotent via
deterministic point ids) and prunes within-project orphans. Scoped (watcher)
runs only warn.

Harness mirrors tests/test_bughunt_2026_07_17.py: in-memory Qdrant +
FakeEmbedder + execute_run driven directly on the test loop.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from noesis.core import jobs, state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import execute_run
from noesis.core.vectorstore import VectorStore


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@test",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(root),
            *args,
        ),
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "auth.py").write_text(
        "def validate_token(token):\n"
        '    """Check JWT expiry before trusting claims."""\n'
        "    return token.expiry > now()\n"
    )
    (r / "db.py").write_text("def connect(dsn):\n    return Driver(dsn)\n")
    (r / "util.py").write_text("def slug(s):\n    return s.strip().lower()\n")
    return r


@pytest.fixture()
def ctx(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)

    class _Ctx:
        pass

    c = _Ctx()
    c.conn = conn
    c.store = store
    c.embedder = embedder
    return c


async def _index(ctx, repo, *, paths=None):
    project_id = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)
    run_id, _ = state.try_start_run(ctx.conn, project_id)
    result = await execute_run(
        ctx.conn,
        ctx.store,
        ctx.embedder,
        str(repo),
        project_id,
        run_id,
        paths=paths,
    )
    return project_id, result


async def _reindex(ctx, repo, project_id, *, paths=None):
    run_id, _ = state.try_start_run(ctx.conn, project_id)
    return await execute_run(
        ctx.conn,
        ctx.store,
        ctx.embedder,
        str(repo),
        project_id,
        run_id,
        paths=paths,
    )


# --- 1. The incident: full collection wipe self-heals -----------------------


async def test_drift_self_heal_restores_chunks(ctx, repo):
    project_id, _ = await _index(ctx, repo)
    expected = state.expected_chunk_total(ctx.conn, project_id)
    assert expected > 0
    assert ctx.store.count_project_points(project_id) == expected

    # Simulate the external wipe: every point for the project vanishes while
    # the state DB still reports all files indexed.
    ctx.store.delete_project_points(project_id)
    assert ctx.store.count_project_points(project_id) == 0
    assert state.expected_chunk_total(ctx.conn, project_id) == expected

    # Incremental reindex with zero file changes must now detect the drift
    # and re-embed everything.
    result = await _reindex(ctx, repo, project_id)
    assert ctx.store.count_project_points(project_id) == expected
    assert result.chunks_written == expected


# --- 2. Partial drift touches only the missing file -------------------------


async def test_partial_drift_restores_only_missing_file(ctx, repo):
    project_id, _ = await _index(ctx, repo)
    expected = state.expected_chunk_total(ctx.conn, project_id)

    # Drop one file's points only.
    ctx.store.delete_file_chunks(project_id, ["util.py"])
    missing = state.get_file_chunk_counts(ctx.conn, project_id)["util.py"]
    assert ctx.store.count_project_points(project_id) == expected - missing

    result = await _reindex(ctx, repo, project_id)
    assert ctx.store.count_project_points(project_id) == expected
    # Only the drifted file was re-embedded — no content changed.
    assert result.files_indexed == 1
    assert result.chunks_written == missing


# --- 3. No drift keeps the incremental fast path -----------------------------


async def test_no_drift_keeps_fast_path(ctx, repo, caplog):
    project_id, _ = await _index(ctx, repo)
    with caplog.at_level(logging.WARNING):
        result = await _reindex(ctx, repo, project_id)
    assert result.files_indexed == 0
    assert result.chunks_written == 0
    assert "drift detected" not in caplog.text


# --- 4. Scoped (watcher) run warns but does not full-reindex -----------------


async def test_scoped_run_warns_but_no_escalation(ctx, repo, caplog):
    project_id, _ = await _index(ctx, repo)
    ctx.store.delete_project_points(project_id)

    with caplog.at_level(logging.WARNING):
        result = await _reindex(ctx, repo, project_id, paths=["auth.py"])
    assert "drift detected" in caplog.text
    assert "scoped run" in caplog.text
    # Only the scoped file was processed; the rest stay drifted until a full
    # reindex.
    assert result.files_indexed <= 1


# --- 5. Never-indexed project raises no false drift --------------------------


async def test_never_indexed_no_false_drift(ctx, tmp_path, caplog):
    empty = tmp_path / "empty"
    empty.mkdir()
    with caplog.at_level(logging.WARNING):
        project_id, result = await _index(ctx, empty)
    assert "drift detected" not in caplog.text
    assert state.expected_chunk_total(ctx.conn, project_id) == 0
    assert ctx.store.count_project_points(project_id) == 0


# --- 6. Disk deletion is not misread as drift --------------------------------


async def test_disk_deletion_no_false_drift(ctx, repo, caplog):
    project_id, _ = await _index(ctx, repo)
    (repo / "util.py").unlink()
    with caplog.at_level(logging.WARNING):
        result = await _reindex(ctx, repo, project_id)
    # Counts matched before the delete-branch pruned; no drift warning fired.
    assert "drift detected" not in caplog.text
    assert result.files_deleted == 1
    assert ctx.store.count_project_points(project_id) == state.expected_chunk_total(
        ctx.conn, project_id
    )


# --- 7. Within-project orphan points are pruned ------------------------------


async def test_orphan_points_pruned(ctx, repo):
    project_id, _ = await _index(ctx, repo)
    expected = state.expected_chunk_total(ctx.conn, project_id)

    # Inject a point for a file_path that is neither tracked in state nor on
    # disk — an orphan no other prune path covers.
    ctx.store._client.upsert(
        collection_name=ctx.store._collection,
        points=[
            models.PointStruct(
                id="00000000-0000-0000-0000-0000000000ff",
                vector={
                    "dense": [0.0] * 8,
                    "bm25": models.Document(text="ghost", model="Qdrant/bm25"),
                },
                payload={
                    "project_id": project_id,
                    "file_path": "ghost.py",
                    "start_line": 1,
                    "file_hash": "deadbeef",
                    "text": "ghost",
                },
            )
        ],
        wait=True,
    )
    assert ctx.store.count_project_points(project_id) == expected + 1

    await _reindex(ctx, repo, project_id)
    assert ctx.store.count_project_points(project_id) == expected
    assert "ghost.py" not in ctx.store.per_file_point_counts(project_id)


# --- 8. index_status exposes drift -------------------------------------------


async def test_index_status_exposes_drift(ctx, repo):
    project_id, _ = await _index(ctx, repo)
    status = jobs.index_status(ctx, project_id)
    assert status["drift"] is False
    assert status["vector_count"] == status["expected_chunks"]

    ctx.store.delete_project_points(project_id)
    status = jobs.index_status(ctx, project_id)
    assert status["drift"] is True
    assert status["vector_count"] == 0
    assert status["expected_chunks"] > 0


# --- 9. Drift heal under a live git fast path (PR #20 review finding #1) ------


async def test_drift_self_heal_with_git_fast_path(ctx, repo):
    """The `candidates is not None and drifted` union branch only runs on a
    full run with an active git fast path — every other drift test uses a
    non-git fixture where candidates stays None. Commit the repo so run 2
    fast-paths off the anchor, then wipe points and confirm the drifted files
    are still re-embedded through that branch."""
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    project_id, _ = await _index(ctx, repo)
    expected = state.expected_chunk_total(ctx.conn, project_id)
    assert expected > 0
    # First run recorded the anchor, so the next run takes the git fast path.
    assert state.get_project(ctx.conn, project_id)["last_indexed_commit"]

    ctx.store.delete_project_points(project_id)

    result = await _reindex(ctx, repo, project_id)
    # fast_path_used True proves candidates was not None on this run — i.e. the
    # union branch was reachable — and the restored count proves it re-embedded.
    assert result.fast_path_used is True
    assert ctx.store.count_project_points(project_id) == expected
    assert result.chunks_written == expected
