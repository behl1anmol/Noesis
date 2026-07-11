"""End-to-end REST tests: register → poll run → search, on fakes.

FakeEmbedder + in-memory Qdrant keep the suite offline (no model download,
no Docker) while exercising the same code paths production uses; the real
model is covered by the opt-in ``-m integration`` test.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.reranker import FakeReranker
from noesis.core.vectorstore import VectorStore


@pytest.fixture()
def project_dir(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "auth.py").write_text(
        "def validate_token(token):\n"
        '    """Check JWT expiry before trusting claims."""\n'
        "    return token.expiry > now()\n"
    )
    (src / "db.py").write_text("def connect(dsn):\n    return Driver(dsn)\n")
    return src


def make_client(tmp_path, reranker=None):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    ctx = AppContext(conn=conn, store=store, embedder=embedder, reranker=reranker)
    app = create_app(ctx=ctx)
    return TestClient(app)


@pytest.fixture()
def client(tmp_path):
    with make_client(tmp_path) as tc:
        yield tc


@pytest.fixture()
def client_with_reranker(tmp_path):
    with make_client(tmp_path, reranker=FakeReranker()) as tc:
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


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_register_index_search_roundtrip(client, project_dir):
    resp = client.post("/projects", json={"root_path": str(project_dir)})
    assert resp.status_code == 202
    body = resp.json()
    run = asyncio.run(_wait_done(client, body["run_id"]))
    assert run["status"] == "done"
    assert run["chunks_written"] > 0

    resp = client.post(
        "/search",
        json={"query": "validate token", "project_id": body["project_id"], "top_k": 5},
    )
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits, "expected at least one hit"
    top = hits[0]
    for key in ("file_path", "start_line", "end_line", "score", "snippet"):
        assert key in top
    assert top["file_path"] in ("auth.py", "db.py")


def test_project_status_and_reindex_roundtrip(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    project_id = body["project_id"]
    asyncio.run(_wait_done(client, body["run_id"]))

    status = client.get(f"/projects/{project_id}/status").json()
    assert status["project_id"] == project_id
    assert status["run_id"] == body["run_id"]
    assert status["status"] == "done"
    assert status["chunks_written"] > 0

    resp = client.post(f"/projects/{project_id}/reindex")
    assert resp.status_code == 202
    again = resp.json()
    assert again["project_id"] == project_id
    assert again["run_id"] != body["run_id"]
    run = asyncio.run(_wait_done(client, again["run_id"]))
    # Incremental: nothing changed between the two runs.
    assert run["status"] == "done"
    assert run["files_changed"] == 0
    assert (
        client.get(f"/projects/{project_id}/status").json()["run_id"] == again["run_id"]
    )


def test_project_status_unknown_404(client):
    assert client.get("/projects/nope/status").status_code == 404
    assert client.post("/projects/nope/reindex").status_code == 404


def test_search_unknown_project_404(client):
    resp = client.post("/search", json={"query": "x", "project_id": "nope"})
    assert resp.status_code == 404


def test_run_status_unknown_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_search_channel_param(client, project_dir):
    resp = client.post("/projects", json={"root_path": str(project_dir)})
    body = resp.json()
    run = asyncio.run(_wait_done(client, body["run_id"]))
    assert run["status"] == "done"

    # Default is hybrid and the response says so.
    resp = client.post(
        "/search", json={"query": "validate_token", "project_id": body["project_id"]}
    )
    assert resp.status_code == 200
    assert resp.json()["channel"] == "hybrid"

    # Sparse-only surfaces the exact-symbol file without any dense help.
    resp = client.post(
        "/search",
        json={
            "query": "validate_token",
            "project_id": body["project_id"],
            "channel": "sparse",
        },
    )
    assert resp.status_code == 200
    body_sparse = resp.json()
    assert body_sparse["channel"] == "sparse"
    assert body_sparse["hits"][0]["file_path"] == "auth.py"

    # Unknown channel is rejected by validation.
    resp = client.post(
        "/search",
        json={"query": "x", "project_id": body["project_id"], "channel": "psychic"},
    )
    assert resp.status_code == 422


def test_search_without_reranker_states_not_reranked(client, project_dir):
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    assert asyncio.run(_wait_done(client, body["run_id"]))["status"] == "done"

    # No reranker wired: default off, and rerank=true is not an error —
    # the response just states reranking was not applied (§3.3 contract).
    for payload in ({}, {"rerank": True}):
        resp = client.post(
            "/search",
            json={
                "query": "validate token",
                "project_id": body["project_id"],
                **payload,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reranked"] is False
        assert all("rerank_score" not in h for h in resp.json()["hits"])


def test_search_with_reranker_defaults_on_and_opts_out(
    client_with_reranker, project_dir
):
    client = client_with_reranker
    body = client.post("/projects", json={"root_path": str(project_dir)}).json()
    assert asyncio.run(_wait_done(client, body["run_id"]))["status"] == "done"

    # rerank omitted → defaults to reranker availability (config enabled).
    resp = client.post(
        "/search",
        json={"query": "validate token expiry", "project_id": body["project_id"]},
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["reranked"] is True
    assert out["hits"]
    assert all("rerank_score" in h and "text" not in h for h in out["hits"])
    # FakeReranker is lexical-overlap: the token-validating chunk wins.
    assert out["hits"][0]["file_path"] == "auth.py"

    # Per-request opt-out.
    resp = client.post(
        "/search",
        json={
            "query": "validate token expiry",
            "project_id": body["project_id"],
            "rerank": False,
        },
    )
    assert resp.json()["reranked"] is False
    assert all("rerank_score" not in h for h in resp.json()["hits"])
