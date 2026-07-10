"""Regression tests for the 2026-07-10 bug hunt.

One test (or pair) per confirmed finding, in severity order:

1. (high) Transient hash OSError under the git fast path must not strand a
   committed file with stale index content: partition surfaces the error,
   the run refuses to advance the anchor, and the path re-enters the retry
   loop via failed_paths.
2. (medium) Hybrid RRF prefetch depth must stay `candidates` deep when no
   reranker runs — not collapse to top_k.
3. (medium) The already-running launch guard is atomic (BEGIN IMMEDIATE)
   and owner-gated: a live run dedups, a dead process's stale row is failed
   and replaced instead of jamming the guard.
4. (low) `[embedder] batch_size` reaches execute_run from the context.
5. (low) ensure_collection survives losing a concurrent first create.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from noesis.core import hashdiff, jobs, state
from noesis.core.embedder import FakeEmbedder
from noesis.core.hashdiff import partition
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.retriever import search_code
from noesis.core.vectorstore import VectorStore
from noesis.runtime import AppContext

from tests.test_gitfast import (
    anchor_of,
    build_git_repo,
    git,
    git_head,
    make_env,
    requires_git,
)


# --- 1. hash-error carry-forward must not strand a file ----------------------


def test_partition_reports_hash_errors_and_still_carries_forward(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    real_hash = hashdiff.hash_file

    def flaky(path):
        if Path(path).name == "a.py":
            raise PermissionError("transient EACCES")
        return real_hash(path)

    hashdiff.hash_file = flaky
    try:
        diff = partition(
            tmp_path,
            ["a.py", "b.py"],
            {"a.py": "old-hash", "b.py": "old-hash-b"},
            candidates={"a.py", "b.py"},
        )
    finally:
        hashdiff.hash_file = real_hash

    # H7 carry-forward still holds: never treated as deleted, old hash serves.
    assert "a.py" not in diff.deleted
    assert diff.hashes["a.py"] == "old-hash"
    assert "a.py" in diff.unchanged
    # New: the failure is visible to the caller instead of silent.
    assert [p for p, _ in diff.errored] == ["a.py"]
    assert "EACCES" in dict(diff.errored)["a.py"]
    # b.py hashed normally (changed, since content differs from stored hash).
    assert "b.py" in diff.changed


def test_hash_error_new_file_is_skipped_but_reported(tmp_path):
    (tmp_path / "new.py").write_text("z = 3\n")
    real_hash = hashdiff.hash_file

    def broken(path):
        raise OSError("EIO")

    hashdiff.hash_file = broken
    try:
        diff = partition(tmp_path, ["new.py"], {})
    finally:
        hashdiff.hash_file = real_hash

    assert diff.new == () and diff.deleted == ()
    assert [p for p, _ in diff.errored] == ["new.py"]


@requires_git
def test_hash_error_blocks_anchor_advance_and_repends(tmp_path, monkeypatch):
    """The full stranding loop from the finding, now closed.

    A file modified in a commit is a fast-path candidate exactly once; if
    its hash errors transiently in that run, the anchor must stay put (else
    no future run ever re-nominates it and its stale chunks serve forever).
    """
    root = tmp_path / "repo"
    build_git_repo(root)
    conn, store, embedder = make_env(tmp_path)

    async def run(run_paths=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(conn, store, embedder, str(root), project_id, run_id)

    result = asyncio.run(run())
    pid = result.project_id
    c0 = git_head(root)
    assert anchor_of(conn, pid) == c0

    # Commit a change to alpha.py: it is a fast-path candidate next run.
    (root / "alpha.py").write_text("def alpha():\n    return 100\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change alpha")
    c1 = git_head(root)

    real_hash = hashdiff.hash_file

    def flaky(path):
        if Path(path).name == "alpha.py":
            raise OSError("transient EIO")
        return real_hash(path)

    monkeypatch.setattr(hashdiff, "hash_file", flaky)
    result = asyncio.run(run())
    monkeypatch.undo()

    # The run completes (old chunks keep serving) but is honest about the
    # failure, keeps the anchor at c0, and re-queues the path.
    assert result.files_failed == 1
    assert result.failed_paths == ("alpha.py",)
    assert anchor_of(conn, pid) == c0
    run_row = state.get_latest_run(conn, pid)
    assert run_row["status"] == "done"
    assert run_row["files_failed"] == 1

    # Next run (hash healthy again): alpha.py is still a candidate because
    # the anchor never passed c1 — it re-hashes, re-indexes, and the anchor
    # advances. Pre-fix, this run carried the stale hash forward silently.
    result = asyncio.run(run())
    assert result.files_failed == 0
    assert result.files_indexed == 1  # alpha.py re-indexed
    assert anchor_of(conn, pid) == c1


# --- 2. hybrid prefetch depth without a reranker ------------------------------


def test_retriever_prefetch_depth_without_reranker():
    captured: dict = {}

    class SpyStore:
        def search(self, project_id, **kw):
            captured.update(kw)
            return []

    async def scenario():
        return await search_code(
            SpyStore(), FakeEmbedder(dim=8), "query", "proj", top_k=10
        )

    result = asyncio.run(scenario())
    assert result == {"hits": [], "reranked": False}
    assert captured["top_k"] == 10  # result depth unchanged
    assert captured["prefetch_limit"] == 50  # channel depth stays wide


# --- 3. atomic, owner-gated launch guard --------------------------------------


def test_try_start_run_dedups_against_live_run(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, tmp_path, "model")

    run_id, created = state.try_start_run(conn, pid)
    assert created
    # Second caller — even over a second connection (second process) — sees
    # the live run instead of opening a concurrent one.
    conn2 = state.connect(tmp_path / "state.sqlite")
    dup_id, dup_created = state.try_start_run(conn2, pid)
    assert not dup_created and dup_id == run_id

    state.finish_run(conn, run_id, "done")
    next_id, next_created = state.try_start_run(conn2, pid)
    assert next_created and next_id != run_id
    conn2.close()
    conn.close()


def test_try_start_run_fails_dead_owner_rows(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, tmp_path, "model")
    stale = state.start_run(conn, pid)
    # Rewrite the owner to a different boot token: unambiguously dead.
    conn.execute(
        "UPDATE index_runs SET owner = 'not-this-boot:1:1' WHERE id = ?", (stale,)
    )
    conn.commit()

    run_id, created = state.try_start_run(conn, pid)
    assert created and run_id != stale
    stale_row = conn.execute(
        "SELECT status, error FROM index_runs WHERE id = ?", (stale,)
    ).fetchone()
    assert stale_row["status"] == "failed"
    assert stale_row["error"] == "interrupted"
    conn.close()


# --- 4. configured batch size reaches the indexer ------------------------------


def test_launch_index_run_passes_context_batch_size(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    ctx = AppContext(conn=conn, store=store, embedder=embedder, embed_batch_size=128)

    seen: dict = {}

    async def fake_execute_run(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(failed_paths=())

    monkeypatch.setattr(jobs, "execute_run", fake_execute_run)

    async def scenario():
        body = jobs.launch_index_run(ctx, str(root))
        assert body["status"] == "accepted"
        await asyncio.gather(*ctx.jobs.values())

    asyncio.run(scenario())
    assert seen["batch_size"] == 128
    conn.close()


# --- 5. ensure_collection concurrent-create race -------------------------------


def test_ensure_collection_survives_losing_create_race(monkeypatch):
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder(dim=8)
    winner = VectorStore(client, collection_name="race")
    winner.ensure_collection(embedder)

    # Loser: its exists-check raced before the winner's create. First probe
    # lies False (the pre-create snapshot); the retry after the conflict
    # sees the truth.
    loser = VectorStore(client, collection_name="race")
    real_exists = client.collection_exists
    calls = {"n": 0}

    def stale_then_real(name):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real_exists(name)

    monkeypatch.setattr(client, "collection_exists", stale_then_real)
    loser.ensure_collection(embedder)  # must verify, not crash

    # Shape verification still bites on a real mismatch after the race.
    calls["n"] = 0
    with pytest.raises(ValueError, match="refusing mixed-model"):
        loser.ensure_collection(FakeEmbedder(dim=16))
