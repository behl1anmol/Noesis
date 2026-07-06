"""ADR-42 register surface: register-only vs add+index, per-project index
config honored across runs, pre-flight preview, folder browse, languages.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import dashboard as core_dashboard
from noesis.core import indexer, state
from noesis.core.discovery import DiscoveryConfig, discover_files
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "a.py").write_text("x = 1\n")
    (r / "src" / "d.py").write_text("y = 2\n")
    (r / "b.go").write_text("package main\n")
    (r / "c.md").write_text("# doc\n")
    (r / "big.py").write_text("z = 0\n" * 5000)
    return r


def make_ctx(tmp_path) -> AppContext:
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(ctx=make_ctx(tmp_path))) as tc:
        yield tc


async def _wait_done(client, run_id, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("run stuck")
        await asyncio.sleep(0.02)


# -- discovery filter ---------------------------------------------------------


def test_discovery_language_filter(repo):
    all_files = discover_files(repo)
    assert set(all_files) == {"a.py", "src/d.py", "b.go", "c.md", "big.py"}
    py_only = discover_files(repo, DiscoveryConfig(include_languages=frozenset(["python"])))
    assert set(py_only) == {"a.py", "src/d.py", "big.py"}


def test_discovery_extra_ignores_and_size(repo):
    filtered = discover_files(
        repo, DiscoveryConfig(extra_ignore_patterns=("src/",), max_file_bytes=100)
    )
    # src/ ignored, big.py over the 100-byte cap → dropped
    assert set(filtered) == {"a.py", "b.go", "c.md"}


# -- supported languages / browse / preview -----------------------------------


def test_language_chips_server_rendered(client):
    """The register modal must not depend on a runtime fetch: chips are in
    the HTML itself, and pages are no-store so a stale document can't wire
    stale expectations to fresh assets."""
    resp = client.get("/")
    assert resp.headers["cache-control"] == "no-store"
    html = resp.text
    assert html.count('class="lang-chip"') >= 20  # all languages present
    assert 'value="python"' in html and "loading languages" not in html


def test_supported_languages(client):
    data = client.get("/api/languages").json()["languages"]
    names = {l["language"] for l in data}
    assert {"python", "go", "markdown", "toml"} <= names
    py = next(l for l in data if l["language"] == "python")
    assert ".py" in py["extensions"] and py["structural"] is True
    toml = next(l for l in data if l["language"] == "toml")
    assert toml["structural"] is False  # no ast-grep support (LANGUAGE_MAP)


def test_browse_lists_directories_only(client, repo, tmp_path):
    (repo / "a.py").write_text("x=1\n")  # a file, must not appear
    resp = client.get("/api/browse", params={"path": str(repo)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == str(repo.resolve())
    assert body["parent"] == str(repo.parent.resolve())
    assert [e["name"] for e in body["entries"]] == ["src"]  # only the dir
    # non-existent path → 400
    assert client.get("/api/browse", params={"path": str(tmp_path / "nope")}).status_code == 400


def test_preview_scan_and_filter(client, repo):
    full = client.post("/api/register/preview", json={"root_path": str(repo)}).json()
    assert full["total_files"] == 5
    langs = {b["language"]: b["files"] for b in full["by_language"]}
    assert langs == {"python": 3, "go": 1, "markdown": 1}

    py = client.post(
        "/api/register/preview",
        json={"root_path": str(repo), "index_languages": ["python"]},
    ).json()
    assert py["total_files"] == 3
    assert [b["language"] for b in py["by_language"]] == ["python"]

    assert client.post(
        "/api/register/preview", json={"root_path": "/no/such/dir"}
    ).status_code == 400


# -- register: add-only vs add+index, config persistence ----------------------


def test_register_add_only(client, repo):
    resp = client.post(
        "/api/register",
        json={"root_path": str(repo), "index_now": False, "watch": True},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["run"] is None                    # not indexed
    assert body["project"]["file_count"] == 0
    assert body["project"]["watch_enabled"] is True
    assert body["project"]["watching"] is True    # watch scheduled live


def test_register_add_and_index_honors_language_filter(client, repo):
    resp = client.post(
        "/api/register",
        json={
            "root_path": str(repo),
            "index_now": True,
            "index_languages": ["python"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    run = asyncio.run(_wait_done(client, body["run"]["run_id"]))
    assert run["status"] == "done"
    pid = body["project"]["id"]
    # Only python files got indexed (a.py, src/d.py, big.py) — filter persisted
    # and threaded into the run via discovery_config_for_project.
    files = client.app.state.ctx.conn.execute(
        "SELECT path FROM files WHERE project_id=?", (pid,)
    ).fetchall()
    assert {r["path"] for r in files} == {"a.py", "src/d.py", "big.py"}


def test_register_config_survives_later_reindex(client, repo):
    body = client.post(
        "/api/register",
        json={"root_path": str(repo), "index_now": True, "index_languages": ["go"]},
    ).json()
    pid = body["project"]["id"]
    asyncio.run(_wait_done(client, body["run"]["run_id"]))
    # A plain reindex (no config passed) must still honor the go-only scope.
    r2 = client.post(f"/projects/{pid}/reindex").json()
    run = asyncio.run(_wait_done(client, r2["run_id"]))
    assert run["files_total"] == 1  # only b.go discovered under the filter


def test_register_missing_dir_400(client):
    assert client.post(
        "/api/register", json={"root_path": "/no/such/dir"}
    ).status_code == 400


# -- project deletion (ADR-43) ------------------------------------------------


def test_delete_project_full_cleanup(client, repo):
    body = client.post(
        "/api/register",
        json={"root_path": str(repo), "index_now": True, "watch": True},
    ).json()
    pid = body["project"]["id"]
    asyncio.run(_wait_done(client, body["run"]["run_id"]))
    ctx = client.app.state.ctx
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])
    state.bump_watcher_stats(ctx.conn, pid, events_seen=3)
    assert ctx.watcher.watching(pid)
    points_before = ctx.store._client.count(ctx.store.collection_name).count
    assert points_before > 0

    resp = client.delete(f"/api/projects/{pid}")
    assert resp.status_code == 200 and resp.json()["deleted"] is True

    # Source tree untouched; every index-state surface gone.
    assert (repo / "a.py").exists()
    assert state.get_project(ctx.conn, pid) is None
    for table in ("files", "index_runs", "pending_changes", "watcher_stats"):
        n = ctx.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE project_id = ?", (pid,)
        ).fetchone()[0]
        assert n == 0, table
    assert ctx.store._client.count(ctx.store.collection_name).count == 0
    assert not ctx.watcher.watching(pid)
    # Idempotent-ish: second delete is a clean 404.
    assert client.delete(f"/api/projects/{pid}").status_code == 404
    # Re-registering the same folder works (fresh id, fresh index).
    again = client.post(
        "/api/register", json={"root_path": str(repo), "index_now": False}
    )
    assert again.status_code == 201


def test_delete_unknown_project_404(client):
    assert client.delete("/api/projects/nope").status_code == 404


# -- PR #10 review fixes --------------------------------------------------------


def test_reregister_watch_off_unschedules(client, repo):
    body = client.post(
        "/api/register", json={"root_path": str(repo), "watch": True}
    ).json()
    pid = body["project"]["id"]
    assert body["project"]["watching"] is True
    again = client.post(
        "/api/register", json={"root_path": str(repo), "watch": False}
    ).json()
    assert again["project"]["id"] == pid
    assert again["project"]["watch_enabled"] is False
    assert again["project"]["watching"] is False  # live watch unscheduled


def test_failed_file_stays_pending_after_scoped_run(client, repo, monkeypatch):
    from noesis.core import indexer as indexer_mod

    body = client.post(
        "/api/register", json={"root_path": str(repo), "index_now": True}
    ).json()
    pid = body["project"]["id"]
    asyncio.run(_wait_done(client, body["run"]["run_id"]))
    ctx = client.app.state.ctx

    real_chunk_file = indexer_mod.chunk_file

    def flaky(text, *, language=None, file_path=None, file_hash=None):
        if file_path == "a.py":
            raise RuntimeError("boom")
        return real_chunk_file(
            text, language=language, file_path=file_path, file_hash=file_hash
        )

    monkeypatch.setattr(indexer_mod, "chunk_file", flaky)
    (repo / "a.py").write_text("x = 99\n")
    (repo / "b.go").write_text("package main // v2\n")
    state.upsert_pending_changes(
        ctx.conn, pid, [("a.py", "modified"), ("b.go", "modified")]
    )
    # Scoped run over exactly the pending set (the watcher's launch path).
    launched = client.post(f"/api/projects/{pid}/reindex-pending").json()
    run = asyncio.run(_wait_done(client, launched["run_id"]))
    assert run["files_failed"] == 1
    # b.go cleared (indexed); a.py re-pended so the watcher retries it —
    # without this, only a manual full run would ever pick it up (ADR-41).
    remaining = {r["path"] for r in state.list_pending_changes(ctx.conn, pid)}
    assert remaining == {"a.py"}


def test_reranker_only_pin_blocks_device_change(tmp_path):
    from noesis.core import dashboard as core_dashboard

    ctx = make_ctx(tmp_path)
    ctx.config_reranker_device_pin = "cuda"
    import pytest as _pytest

    with _pytest.raises(ValueError, match="pinned"):
        core_dashboard.set_compute_device(ctx, "cpu")


def test_orphaned_running_runs_failed_and_unblock_launch(client, repo):
    ctx = client.app.state.ctx
    pid = state.register_project(ctx.conn, repo, ctx.embedder.model_id)
    dead_run = state.start_run(ctx.conn, pid)  # simulates a crash leftover
    # Owner-gated recovery (M7) only fails runs whose process is dead; stamp
    # this row with an owner from a different boot so it reads as a genuine
    # crash leftover rather than this live test process's run.
    ctx.conn.execute(
        "UPDATE index_runs SET owner = ? WHERE id = ?", ("dead-boot:1", dead_run)
    )
    ctx.conn.commit()
    assert state.fail_orphaned_runs(ctx.conn) == 1
    row = ctx.conn.execute(
        "SELECT status, error FROM index_runs WHERE id = ?", (dead_run,)
    ).fetchone()
    assert row["status"] == "failed" and row["error"] == "interrupted"
    # Launch guard no longer returns the dead run id.
    resp = client.post(f"/projects/{pid}/reindex")
    assert resp.status_code == 202
    assert resp.json()["run_id"] != dead_run


def test_pending_created_survives_later_modified(client, repo):
    ctx = client.app.state.ctx
    pid = state.register_project(ctx.conn, repo, ctx.embedder.model_id)
    state.upsert_pending_changes(ctx.conn, pid, [("new.py", "created")])
    state.upsert_pending_changes(ctx.conn, pid, [("new.py", "modified")])
    rows = state.list_pending_changes(ctx.conn, pid)
    assert [(r["path"], r["event_type"]) for r in rows] == [("new.py", "created")]
    state.upsert_pending_changes(ctx.conn, pid, [("new.py", "deleted")])
    assert state.list_pending_changes(ctx.conn, pid)[0]["event_type"] == "deleted"


def test_foreign_host_rejected(client, repo):
    # DNS-rebinding guard: same socket, attacker-controlled Host header.
    resp = client.get("/api/state", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400
    resp = client.delete("/api/projects/whatever", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400


def test_discovery_config_for_project_defaults(client, repo):
    body = client.post(
        "/api/register", json={"root_path": str(repo), "index_now": False}
    ).json()
    pid = body["project"]["id"]
    conn = client.app.state.ctx.conn
    # No filter set → None-ish config equal to the default walk.
    cfg = indexer.discovery_config_for_project(conn, pid)
    assert cfg.include_languages is None
    assert cfg.max_file_bytes == DiscoveryConfig().max_file_bytes
    assert cfg.follow_symlinks is False
