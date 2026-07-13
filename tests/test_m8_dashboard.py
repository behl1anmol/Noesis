"""M8 tests — state extensions, scoped runs, per-file errors, progress,
dashboard routes, telemetry. Watcher behavior lives in test_watcher.py.

Same offline harness as test_api.py: FakeEmbedder + in-memory Qdrant.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import dashboard as core_dashboard
from noesis.core import indexer, jobs, state
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore


@pytest.fixture()
def project_dir(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "auth.py").write_text("def validate_token(token):\n    return token\n")
    (src / "db.py").write_text("def connect(dsn):\n    return dsn\n")
    return src


def make_ctx(tmp_path) -> AppContext:
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


@pytest.fixture()
def ctx(tmp_path):
    return make_ctx(tmp_path)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(ctx=make_ctx(tmp_path))) as tc:
        yield tc


async def _wait_done(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"run {run_id} still {body['status']}")
        await asyncio.sleep(0.02)


# -- state layer --------------------------------------------------------------


def test_init_db_idempotent_with_migrations(tmp_path):
    conn = state.connect(tmp_path / "s.sqlite")
    state.init_db(conn)
    state.init_db(conn)  # re-running must not fail or duplicate columns
    cols = {r[1] for r in conn.execute("PRAGMA table_info(projects)")}
    assert {"watch_enabled", "auto_reindex"} <= cols
    run_cols = {r[1] for r in conn.execute("PRAGMA table_info(index_runs)")}
    assert {"triggered_by", "files_failed"} <= run_cols


def test_pending_changes_upsert_and_scoped_clear(ctx, project_dir):
    pid = state.register_project(ctx.conn, project_dir, "fake")
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])
    cut = ctx.conn.execute("SELECT detected_at FROM pending_changes").fetchone()[0]
    # Re-dirty after the cut: the clear at `cut` must NOT remove the row.
    state.upsert_pending_changes(
        ctx.conn, pid, [("a.py", "deleted"), ("b.py", "created")]
    )
    state.clear_pending_changes(ctx.conn, pid, paths=["a.py"], before=cut)
    remaining = {r["path"] for r in state.list_pending_changes(ctx.conn, pid)}
    assert remaining == {"a.py", "b.py"}  # a.py survived (re-dirtied later)
    state.clear_pending_changes(ctx.conn, pid)  # full clear
    assert state.list_pending_changes(ctx.conn, pid) == []


def test_settings_roundtrip(ctx):
    assert state.get_setting(ctx.conn, "compute_device") is None
    state.set_setting(ctx.conn, "compute_device", "cpu")
    state.set_setting(ctx.conn, "compute_device", "auto")
    assert state.get_setting(ctx.conn, "compute_device") == "auto"


def test_project_flags(ctx, project_dir):
    pid = state.register_project(ctx.conn, project_dir, "fake")
    state.set_project_flags(ctx.conn, pid, watch_enabled=True, auto_reindex=True)
    row = state.get_project(ctx.conn, pid)
    assert row["watch_enabled"] == 1 and row["auto_reindex"] == 1
    assert [r["id"] for r in state.watched_projects(ctx.conn)] == [pid]


# -- scoped runs / per-file errors / progress ---------------------------------


def test_scoped_run_hashes_only_candidates(ctx, project_dir):
    result = asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    assert result.files_indexed == 2
    # Change both files, but scope the run to auth.py only.
    (project_dir / "auth.py").write_text("def validate_token(t):\n    return not t\n")
    (project_dir / "db.py").write_text("def connect(d):\n    return None\n")
    pid, rid = indexer.prepare_run(ctx.conn, ctx.embedder, str(project_dir))
    scoped = asyncio.run(
        indexer.execute_run(
            ctx.conn,
            ctx.store,
            ctx.embedder,
            str(project_dir),
            pid,
            rid,
            paths=["auth.py"],
        )
    )
    assert scoped.files_indexed == 1  # db.py carried forward unhashed
    assert scoped.fast_path_used is False
    # db.py is stale by design until the next unscoped run examines it.
    follow = asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    assert follow.files_indexed == 1  # exactly db.py


def test_scoped_run_detects_deletions_and_never_advances_anchor(
    ctx, project_dir, tmp_path
):
    import os
    import subprocess

    def git(*args):
        subprocess.run(
            ["git", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@t",
            },
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "init")
    asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    pid = state.register_project(ctx.conn, project_dir, ctx.embedder.model_id)
    anchor = state.get_project(ctx.conn, pid)["last_indexed_commit"]
    assert anchor  # full run on a git repo records the anchor

    (project_dir / "db.py").unlink()
    (project_dir / "auth.py").write_text("changed = 1\n")
    git("add", "-A")
    git("commit", "-qm", "change")
    _, rid = indexer.prepare_run(ctx.conn, ctx.embedder, str(project_dir))
    scoped = asyncio.run(
        indexer.execute_run(
            ctx.conn,
            ctx.store,
            ctx.embedder,
            str(project_dir),
            pid,
            rid,
            paths=["auth.py"],
        )
    )
    assert scoped.files_deleted == 1  # deletion caught discovery-wide
    # The anchor must not move: HEAD advanced, but the scoped candidate set
    # was watcher-derived, not git-derived (§3.2 rule 1 / ADR-40).
    assert state.get_project(ctx.conn, pid)["last_indexed_commit"] == anchor


def test_per_file_error_contained(ctx, project_dir, monkeypatch):
    real_chunk_file = indexer.chunk_file

    def flaky(text, *, language=None, file_path=None, file_hash=None):
        if file_path == "auth.py":
            raise RuntimeError("boom on auth.py")
        return real_chunk_file(
            text, language=language, file_path=file_path, file_hash=file_hash
        )

    monkeypatch.setattr(indexer, "chunk_file", flaky)
    result = asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    assert result.files_failed == 1
    assert result.files_indexed == 1  # db.py still made it
    pid = result.project_id
    run = state.get_latest_run(ctx.conn, pid)
    assert run["status"] == "done" and run["files_failed"] == 1
    errors = state.list_file_errors(ctx.conn, result.run_id)
    assert [(e["path"], e["error"]) for e in errors] == [("auth.py", "boom on auth.py")]
    # The failed file's state row was never written: the next run retries it.
    monkeypatch.setattr(indexer, "chunk_file", real_chunk_file)
    retry = asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    assert retry.files_indexed == 1 and retry.files_failed == 0


def test_all_files_failed_marks_run_failed(ctx, project_dir, monkeypatch):
    monkeypatch.setattr(
        indexer,
        "chunk_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("total loss")),
    )
    result = asyncio.run(
        indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(project_dir))
    )
    run = state.get_latest_run(ctx.conn, result.project_id)
    assert run["status"] == "failed"
    assert "failed" in run["error"]


def test_progress_callback_and_run_progress(ctx, project_dir):
    seen: list[tuple[int, int, int]] = []
    pid, rid = indexer.prepare_run(ctx.conn, ctx.embedder, str(project_dir))
    asyncio.run(
        indexer.execute_run(
            ctx.conn,
            ctx.store,
            ctx.embedder,
            str(project_dir),
            pid,
            rid,
            on_progress=lambda d, t, c: seen.append((d, t, c)),
        )
    )
    assert [s[:2] for s in seen] == [(1, 2), (2, 2)]  # one tick per file

    import time as _time

    ctx.progress["r1"] = {
        "files_done": 5,
        "files_to_index": 20,
        "chunks_written": 40,
        "monotonic_start": _time.monotonic() - 10.0,
    }
    prog = jobs.run_progress(ctx, "r1")
    assert prog["percent"] == 25.0
    assert prog["eta_s"] == pytest.approx(30.0, rel=0.2)
    assert jobs.run_progress(ctx, "missing") is None


# -- dashboard routes ----------------------------------------------------------


def test_dashboard_pages_render(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    asyncio.run(_wait_done(client, body["run_id"]))
    home = client.get("/")
    assert home.status_code == 200
    assert "repo" in home.text  # project name
    assert client.get("/usage").status_code == 200
    view = client.get(f"/projects/{body['project_id']}/view")
    assert view.status_code == 200
    assert client.get("/projects/nope/view").status_code == 404


def test_api_state_shape_and_flags_toggle(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    asyncio.run(_wait_done(client, body["run_id"]))
    pid = body["project_id"]
    overview = client.get("/api/state").json()
    assert overview["totals"]["projects"] == 1
    proj = overview["projects"][0]
    assert proj["file_count"] == 2 and proj["chunk_count"] > 0
    assert proj["watch_enabled"] is False and proj["auto_reindex"] is False
    assert proj["watch_mode"] is None  # not watched yet

    resp = client.post(f"/api/projects/{pid}/flags", json={"watch_enabled": True})
    assert resp.status_code == 200
    assert resp.json()["watch_enabled"] is True
    assert resp.json()["watching"] is True  # live watch scheduled
    # pytest tmp dirs sit on a native filesystem — no polling fallback here.
    assert resp.json()["watch_mode"] == "native"
    resp = client.post(f"/api/projects/{pid}/flags", json={"watch_enabled": False})
    assert resp.json()["watching"] is False
    assert resp.json()["watch_mode"] is None
    assert client.post("/api/projects/nope/flags", json={}).status_code == 404


def test_device_endpoint(client):
    info = client.get("/api/state").json()["device"]
    assert info["setting"] == "auto"
    assert "cpu" in info["available"]
    resp = client.post("/api/settings/device", json={"device": "cpu"})
    assert resp.status_code == 200
    assert resp.json()["setting"] == "cpu"
    assert (
        client.post("/api/settings/device", json={"device": "tpu"}).status_code == 400
    )


def test_device_config_pin_blocks_dashboard(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.config_device_pin = "cuda"
    with pytest.raises(ValueError, match="pinned"):
        core_dashboard.set_compute_device(ctx, "cpu")


def test_reindex_pending_endpoint(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    asyncio.run(_wait_done(client, body["run_id"]))
    pid = body["project_id"]
    ctx = client.app.state.ctx
    (project_dir / "auth.py").write_text("def validate_token(t):\n    return 2\n")
    state.upsert_pending_changes(ctx.conn, pid, [("auth.py", "modified")])
    resp = client.post(f"/api/projects/{pid}/reindex-pending")
    assert resp.status_code == 202
    run = asyncio.run(_wait_done(client, resp.json()["run_id"]))
    assert run["files_changed"] == 1 and run["triggered_by"] == "manual"
    assert state.list_pending_changes(ctx.conn, pid) == []  # cleared on success


def test_run_status_carries_live_progress_field(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    done = asyncio.run(_wait_done(client, body["run_id"]))
    assert "progress" not in done  # only surfaced while running


# -- telemetry -----------------------------------------------------------------


def test_search_logs_metadata_only(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    asyncio.run(_wait_done(client, body["run_id"]))
    client.post(
        "/search",
        json={"query": "SECRET_QUERY_TEXT", "project_id": body["project_id"]},
    )
    ctx = client.app.state.ctx
    rows = ctx.conn.execute("SELECT * FROM query_log").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["interface"] == "rest" and row["kind"] == "search"
    assert row["channel"] == "hybrid" and row["latency_ms"] is not None
    # Metadata only (ADR-40): the query text appears nowhere in the table.
    assert "SECRET_QUERY_TEXT" not in str(row)
    cols = {r[1] for r in ctx.conn.execute("PRAGMA table_info(query_log)")}
    assert "query" not in cols and "text" not in cols


def test_usage_aggregation(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    asyncio.run(_wait_done(client, body["run_id"]))
    client.post("/search", json={"query": "q", "project_id": body["project_id"]})
    usage = client.get("/api/usage").json()
    assert usage["index_activity"]["total_runs"] == 1
    assert usage["search_usage"]["total_queries"] == 1
    assert usage["search_usage"]["latency_p50_ms"] is not None
    assert usage["index_health"][0]["file_count"] == 2
