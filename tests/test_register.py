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
