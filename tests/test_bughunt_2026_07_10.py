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

PR #14 review (automated) added two more, appended at the end of this file:

6. A whole-tree hash outage (every candidate errors) must mark the run
   failed, not done — the original fix (finding 1 above) only guarded the
   chunk/embed failure path, not a total hash-stage outage.
7. The PostToolUse format hook must read the edited path from stdin JSON,
   not the undocumented $CLAUDE_FILE_PATH env var.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
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


# --- 6. whole-tree hash outage must fail the run, not report done ------------


def test_total_hash_outage_marks_run_failed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = asyncio.run(run())
    assert first.files_indexed == 2  # clean baseline run

    def always_broken(path):
        raise OSError("simulated permission/network outage")

    monkeypatch.setattr(hashdiff, "hash_file", always_broken)
    second = asyncio.run(run())

    # Pre-fix: to_index was empty (nothing hashed successfully) and the
    # all_failed check only looked at file_errors, so this reported "done"
    # despite indexing nothing — the outage was invisible to callers like
    # register_project.py --wait.
    assert second.files_failed == 2
    assert set(second.failed_paths) == {"a.py", "b.py"}
    run_row = state.get_latest_run(conn, second.project_id)
    assert run_row["status"] == "failed"
    assert run_row["error"] == "all 2 files failed"


def test_partial_hash_outage_does_not_false_positive(tmp_path, monkeypatch):
    """A real success (b.py's legitimate change) must not be masked by an
    unrelated hash error elsewhere — the all_failed check must not fire on
    partial failure, only total failure."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = asyncio.run(run())
    pid = first.project_id

    real_hash = hashdiff.hash_file

    def flaky(path):
        if Path(path).name == "a.py":
            raise OSError("transient")
        return real_hash(path)

    monkeypatch.setattr(hashdiff, "hash_file", flaky)
    (root / "b.py").write_text("y = 3\n")
    second = asyncio.run(run())

    assert second.files_failed == 1
    assert second.files_indexed == 1  # b.py's real change still landed
    run_row = state.get_latest_run(conn, pid)
    assert run_row["status"] == "done"


# --- 7. format hook reads stdin JSON, not $CLAUDE_FILE_PATH -------------------


def _load_format_hook():
    """Import format_edited_file.py by path (it's repo tooling, not part of
    the noesis package)."""
    import importlib.util

    hook_path = (
        Path(__file__).resolve().parents[1]
        / ".claude"
        / "hooks"
        / "format_edited_file.py"
    )
    spec = importlib.util.spec_from_file_location("format_edited_file", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_hook_reads_path_from_stdin_json(tmp_path, monkeypatch):
    """The hook must resolve its target from the stdin JSON payload
    (tool_input.file_path), not $CLAUDE_FILE_PATH — that env var isn't a
    documented PostToolUse variable and evaluated empty on at least one
    harness, making the old guard a permanent no-op (PR #14 review).

    Runs the hook in-process with subprocess.run mocked, rather than
    shelling out and checking real formatting output: `ruff` isn't a
    declared project dependency (only formatter-availability, not this
    hook's stdin-parsing logic, would be under test), so an end-to-end
    assertion on reformatted content is only as reliable as the CI
    runner's incidental global tool installs — it passed locally where
    ruff happens to be on PATH and failed in CI where it isn't.
    """
    hook = _load_format_hook()
    target = tmp_path / "messy.py"
    target.write_text("x=1\n")
    payload = json.dumps({"tool_input": {"file_path": str(target)}})

    monkeypatch.delenv("CLAUDE_FILE_PATH", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    calls: list[list[str]] = []
    monkeypatch.setattr(hook.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    hook.main()

    assert len(calls) == 1
    assert calls[0][-1] == str(target)  # resolved from stdin JSON, not env
    assert calls[0][:4] == ["uv", "run", "ruff", "format"]


def test_format_hook_skips_non_python_files(monkeypatch):
    hook = _load_format_hook()
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"tool_input": {"file_path": "notes.md"}}))
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(hook.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    hook.main()

    assert calls == []
