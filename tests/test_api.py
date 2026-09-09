"""End-to-end REST tests: register → poll run → search, on fakes.

FakeEmbedder + in-memory Qdrant keep the suite offline (no model download,
no Docker) while exercising the same code paths production uses; the real
model is covered by the opt-in ``-m integration`` test.
"""

from __future__ import annotations

import asyncio
import threading

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
        try:
            yield tc
        finally:
            # This context owns its telemetry writer (ADR-59); production
            # closes it in close_runtime_context, a test fixture here.
            tc.app.state.ctx.telemetry.close(timeout=5.0)


@pytest.fixture()
def client_with_reranker(tmp_path):
    with make_client(tmp_path, reranker=FakeReranker()) as tc:
        try:
            yield tc
        finally:
            # This context owns its telemetry writer (ADR-59); production
            # closes it in close_runtime_context, a test fixture here.
            tc.app.state.ctx.telemetry.close(timeout=5.0)


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
    # ADR-77/78 (issue #47 finding 4): assets/embedder_ready ride alongside
    # status. FakeEmbedder has no model on the HF hub and no resolved_device
    # attribute at all, so both read their "not applicable to this embedder"
    # values rather than a real ready/missing verdict — a real LocalSTEmbedder
    # is covered by the opt-in -m integration test instead.
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["assets"] == "missing"
    assert body["embedder_ready"] == "n/a"


async def test_healthz_checks_assets_off_the_event_loop_thread():
    """PR #50 round-3 review: embedder_assets_ready() does blocking
    filesystem stat calls and must run via asyncio.to_thread — same
    convention runtime.py already uses for delete_orphan_points — instead
    of synchronously inside the event loop, which would stall every other
    concurrent request for however long the stat calls take."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from noesis.api.routes import healthz

    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_ready(model_id: str) -> bool:
        seen["thread"] = threading.get_ident()
        return True

    ctx = SimpleNamespace(embedder=FakeEmbedder(dim=8))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)))

    with patch("noesis.prefetch.embedder_assets_ready", fake_ready):
        await healthz(request)

    assert seen["thread"] != loop_thread, (
        "embedder_assets_ready ran on the event-loop thread, not a worker "
        "thread — asyncio.to_thread isn't wrapping it"
    )


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


def test_register_at_index_capacity_still_reports_the_registered_project(
    client, project_dir, tmp_path
):
    """POST /projects registers BEFORE it launches, so a capacity refusal
    must not be reported as a bare 429.

    Every other capacity refusal answers 429 (ADR-84/85), and that is right
    where nothing happened. Here the registration has already committed by
    the time the cap is hit, so a 429 would tell the caller nothing happened
    while leaving behind a project whose id it never learned — findable only
    by listing every project. The registration is real, so it is reported
    (202, which is what actually occurred) with the refusal in the body,
    exactly as ``already_running`` already does, and the same way
    ``core.dashboard.register_project`` resolves the same conflict.

    Watched failing against the 429 version: ``assert 429 == 202``, and the
    response body carried no ``project_id`` at all.
    """
    from noesis.core.config import IndexingSettings

    ctx = client.app.state.ctx
    ctx.indexing = IndexingSettings(max_concurrent_index_runs=1)

    # Occupy the single slot with a DIFFERENT project's run, so the cap - not
    # the per-project guard - is what refuses the registration below.
    other = tmp_path / "other_project"
    other.mkdir()
    other_id = state.register_project(ctx.conn, other, "fake-embedder-v1")
    state.try_start_run(ctx.conn, other_id)

    resp = client.post("/projects", json={"root_path": str(project_dir)})

    assert resp.status_code == 202, (
        f"registration committed, so it must be reported; got {resp.status_code}"
    )
    body = resp.json()
    assert body["status"] == "capacity_reached"
    assert body["project_id"], "the caller must learn the id of what was registered"
    assert not body["run_id"], "no run started, so no run id"

    # And the project really is registered, not a phantom id.
    listed = {p["id"] for p in client.get("/projects").json()}
    assert body["project_id"] in listed
