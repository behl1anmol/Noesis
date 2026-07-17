"""Regression tests for the 2026-07-17 bug hunt.

Findings closed here (severity order):

1. A NaN rerank score reached the response payload. The 2026-07-11 hunt
   fixed the *sort key* (NaN → -inf, so the ranking stays a total order)
   but left the raw score in the returned hit. On the MCP surface — the
   primary one — that serialized to a bare ``NaN`` token, which is not
   valid JSON (RFC 8259): a strict client parser rejects the entire search
   result, not just the field. REST happened to escape because its route
   is annotated ``-> dict[str, Any]``, so FastAPI serializes through
   Pydantic, which maps NaN to null by default; that is FastAPI's
   accident, not this code's doing, and it does not save MCP. The score is
   now nulled in core/retriever.py, which is where both adapters share it.
   NaN sorts last, so it only surfaces once the NaN hit is inside the
   returned slice — i.e. when top_k >= the hit count.
2. Only the two reindex POSTs carried the ``verify_local_origin`` CSRF
   guard (L5). FastAPI parses a POST with no Content-Type header as JSON,
   and a cross-site ``fetch(..., {mode: "no-cors"})`` with an untyped Blob
   body sends exactly that with no CORS preflight — so every JSON-body
   dashboard mutation was drivable from a hostile page open in a browser
   on this machine. All mutating dashboard routes now carry the guard.
3. ``delete_project`` cancelled a running index task and immediately wiped
   the project's points, but cancelling a task parked on
   ``await asyncio.to_thread(store.upsert_chunks, ...)`` returns at once
   while the worker thread runs on — so the abandoned write landed after
   the wipe and orphaned those points under a dead project_id, which no
   prune path (all scoped to live projects) would ever reclaim. Two halves:
   execute_run now shields the upsert and waits out the in-flight write on
   cancel, and delete_project awaits the cancelled task. runtime.py also
   sweeps orphans at startup, covering what no wipe could prevent (a
   killed process). Storage-only: project ids are uuid4, so a re-registered
   folder gets a fresh id and can never see stale points.
4. ``query`` accepted an empty/whitespace string on both adapters: an
   empty BM25 document yields an empty sparse vector and the dense embed
   of "" is a wasted worker job. No legitimate empty query exists.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import dashboard as core_dashboard
from noesis.core import jobs, state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import execute_run
from noesis.core.vectorstore import VectorStore
from noesis.mcp.server import build_mcp


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
    return r


def make_client(tmp_path, reranker=None):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    ctx = AppContext(conn=conn, store=store, embedder=embedder, reranker=reranker)
    return TestClient(create_app(ctx=ctx))


async def _wait_done(client, run_id, timeout=10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"run {run_id} still {body['status']}")
        await asyncio.sleep(0.02)


# --- 1. NaN rerank score must not 500 the search response --------------------


class NaNReranker:
    """Reranker whose first candidate scores NaN — the fp16-overflow /
    degenerate-text case retriever.py's own comment admits is possible.
    Remaining candidates score descending so the ordering stays checkable."""

    model_id = "nan-reranker-v1"

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float("nan")] + [float(len(texts) - i) for i in range(1, len(texts))]


def make_ctx(tmp_path, reranker=None):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder, reranker=reranker)


async def _index(ctx, repo) -> str:
    """Register + index synchronously on this loop (no TestClient portal, so
    the MCP tests own their event loop)."""
    project_id = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)
    run_id, _ = state.try_start_run(ctx.conn, project_id)
    await execute_run(ctx.conn, ctx.store, ctx.embedder, str(repo), project_id, run_id)
    return project_id


@pytest.fixture()
def nan_client(tmp_path):
    with make_client(tmp_path, reranker=NaNReranker()) as tc:
        yield tc


@pytest.fixture()
def mcp_nan(tmp_path):
    ctx = make_ctx(tmp_path, reranker=NaNReranker())
    return build_mcp(lambda: ctx), ctx


def _strict_loads(raw: str):
    """json.loads that refuses the bare NaN/Infinity tokens Python's parser
    accepts by extension. Stands in for a strict client parser — which is
    what an MCP client on another runtime is."""

    def _reject(token: str):
        raise AssertionError(f"payload is not valid JSON: bare {token} token")

    return json.loads(raw, parse_constant=_reject)


async def test_mcp_nan_rerank_score_is_valid_json(mcp_nan, repo):
    """The bug that mattered: a NaN score serialized to a bare ``NaN`` token
    on the MCP surface, which RFC 8259 does not allow. A strict client parser
    rejects the whole search result, so one degenerate score takes out every
    hit in the response."""
    mcp, ctx = mcp_nan
    project_id = await _index(ctx, repo)
    async with Client(mcp) as client:
        # top_k above the hit count: the NaN hit sorts last but still lands in
        # the returned slice — exactly when the bad token used to appear.
        result = await client.call_tool(
            "search_code",
            {
                "query": "validate token",
                "project_id": project_id,
                "top_k": 50,
                "rerank": True,
            },
        )
    raw = result.content[0].text
    assert "NaN" not in raw, "bare NaN token in MCP payload"
    body = _strict_loads(raw)
    hits = body["hits"]
    assert hits and body["reranked"] is True

    nulls = [h for h in hits if h["rerank_score"] is None]
    assert len(nulls) == 1, "exactly the NaN-scored pair reports a null score"
    # NaN sorts as -inf, so the unscoreable hit is last and the real scores
    # stay in descending order — the ranking is unharmed by the nulling.
    assert hits[-1]["rerank_score"] is None
    scored = [h["rerank_score"] for h in hits[:-1]]
    assert scored == sorted(scored, reverse=True)
    assert not any(isinstance(s, float) and math.isnan(s) for s in scored)


def test_rest_nan_rerank_score_serializes_as_null(nan_client, repo):
    """REST's half of the same contract. This passed even before the fix —
    FastAPI's response_model path nulls NaN via Pydantic — so it is a guard
    against that accident being lost (e.g. someone returning a bare Response
    or dropping the dict[str, Any] annotation), not proof of the fix."""
    body = nan_client.post("/projects", json={"root_path": str(repo)}).json()
    assert asyncio.run(_wait_done(nan_client, body["run_id"]))["status"] == "done"
    resp = nan_client.post(
        "/search",
        json={
            "query": "validate token",
            "project_id": body["project_id"],
            "top_k": 50,
            "rerank": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert b"NaN" not in resp.content
    hits = _strict_loads(resp.text)["hits"]
    assert hits and len([h for h in hits if h["rerank_score"] is None]) == 1


# --- 2. every mutating dashboard route carries the CSRF guard ----------------


@pytest.fixture()
def client(tmp_path):
    with make_client(tmp_path) as tc:
        yield tc


def _mutations(repo, project_id):
    """(method, url, json) for every state-changing dashboard route."""
    return [
        ("POST", "/api/register", {"root_path": str(repo)}),
        ("POST", "/api/register/preview", {"root_path": str(repo)}),
        ("POST", f"/api/projects/{project_id}/flags", {"watch_enabled": False}),
        ("POST", "/api/settings/device", {"device": "cpu"}),
        ("POST", f"/api/projects/{project_id}/reindex-pending", None),
        ("DELETE", f"/api/projects/{project_id}", None),
    ]


def test_cross_origin_mutations_rejected(client, repo):
    """A page on evil.com must not be able to drive any dashboard mutation
    from the browser on this machine — including via a no-Content-Type POST,
    which FastAPI happily parses as JSON and which needs no CORS preflight."""
    body = client.post("/api/register", json={"root_path": str(repo)}).json()
    pid = body["project"]["id"]

    for method, url, payload in _mutations(repo, pid):
        resp = client.request(
            method, url, json=payload, headers={"Origin": "http://evil.com"}
        )
        assert resp.status_code == 403, f"{method} {url} allowed cross-origin"
        assert resp.json()["detail"] == "cross-origin request rejected"

    # Referer-only (no Origin) is checked too — same class of drive-by.
    resp = client.post(
        "/api/register",
        json={"root_path": str(repo)},
        headers={"Referer": "http://evil.com/page"},
    )
    assert resp.status_code == 403


def test_same_origin_and_headerless_mutations_still_work(client, repo):
    """The guard must not break the two legitimate callers: the same-origin
    dashboard JS (sends Origin: 127.0.0.1) and non-browser clients like curl
    or an agent (send no Origin at all)."""
    body = client.post("/api/register", json={"root_path": str(repo)}).json()
    pid = body["project"]["id"]

    for method, url, payload in _mutations(repo, pid):
        # Same-origin browser.
        resp = client.request(
            method, url, json=payload, headers={"Origin": "http://127.0.0.1:8000"}
        )
        assert resp.status_code != 403, f"{method} {url} blocked same-origin"
        # Header-less client (curl / agent).
        if method != "DELETE":  # the project is gone after the DELETE above
            resp = client.request(method, url, json=payload)
            assert resp.status_code != 403, f"{method} {url} blocked header-less"


# --- 3. delete_project must not orphan points of an in-flight run -----------


class ParkingStore:
    """VectorStore wrapper whose first upsert parks its worker thread until
    released — the exact window delete_project used to wipe through."""

    def __init__(self, inner):
        self._inner = inner
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self._parked = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def upsert_chunks(self, *args, **kwargs):
        if self._parked:
            return self._inner.upsert_chunks(*args, **kwargs)
        self._parked = True
        self.entered.set()
        self.release.wait(timeout=10)
        try:
            return self._inner.upsert_chunks(*args, **kwargs)
        finally:
            self.finished.set()


async def test_delete_project_waits_out_inflight_upsert(tmp_path, repo):
    """Delete a project while a run is mid-upsert: the write in flight must
    land BEFORE the wipe, not after it, so no points survive under the dead
    project_id."""
    ctx = make_ctx(tmp_path)
    inner = ctx.store
    ctx.store = ParkingStore(inner)

    launch = jobs.launch_index_run(ctx, str(repo))
    project_id = launch["project_id"]
    deadline = asyncio.get_event_loop().time() + 10
    while not ctx.store.entered.is_set():
        assert asyncio.get_event_loop().time() < deadline, "upsert never parked"
        await asyncio.sleep(0.01)

    # Release shortly after the delete begins, so the worker thread's write
    # is still pending exactly when the wipe would otherwise run.
    asyncio.get_event_loop().call_later(0.05, ctx.store.release.set)
    assert await core_dashboard.delete_project(ctx, project_id) is True

    # Counting straight after the delete proves nothing: unfixed, the wipe
    # runs at once and the abandoned write lands a moment LATER. Wait for the
    # parked write to actually complete, then look — the fix is that it landed
    # before the wipe, so there is nothing left behind either way it raced.
    deadline = asyncio.get_event_loop().time() + 10
    while not ctx.store.finished.is_set():
        assert asyncio.get_event_loop().time() < deadline, "parked upsert never ran"
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)  # settle: the write is synchronous once released

    remaining = inner._client.count(inner.collection_name).count
    assert remaining == 0, f"{remaining} orphaned point(s) survived the delete"
    assert state.get_project(ctx.conn, project_id) is None


class SlowStore:
    """VectorStore wrapper whose upserts take a beat, so a run launched into
    the delete's await window is reliably still in flight when the wipe runs."""

    def __init__(self, inner, delay=0.15):
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def upsert_chunks(self, *args, **kwargs):
        time.sleep(self._delay)
        return self._inner.upsert_chunks(*args, **kwargs)


async def test_delete_project_drains_a_run_launched_during_the_await(tmp_path, repo):
    """PR #18 review (P2): awaiting the cancelled run hands the event loop back
    to every other launcher. The watcher's _maybe_auto_reindex gates on
    auto_reindex alone and fires the moment the cancelled run stops being
    'running'; a concurrent REST/MCP reindex can win the same window. A
    cancel-once-then-wipe sails straight past that second run, which then
    writes points after the wipe."""
    ctx = make_ctx(tmp_path)
    inner = ctx.store
    ctx.store = SlowStore(inner)

    launch = jobs.launch_index_run(ctx, str(repo))
    project_id, run_a = launch["project_id"], launch["run_id"]
    task_a = ctx.jobs[run_a]

    relaunched: dict = {}

    async def racer():
        # Exactly what the watcher's retry does: wait for the cancelled run to
        # stop being 'running', then launch a fresh one for the same project.
        await asyncio.gather(task_a, return_exceptions=True)
        relaunched.update(jobs.launch_index_run(ctx, str(repo)))

    racing = asyncio.create_task(racer())
    await asyncio.sleep(0)  # racer registers on task_a before delete_project does

    assert await core_dashboard.delete_project(ctx, project_id) is True
    await racing
    assert relaunched.get("status") == "accepted", "the racing relaunch never started"

    # Let anything still running finish, then look: nothing may have survived.
    for task in list(ctx.jobs.values()):
        await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.3)

    remaining = inner._client.count(inner.collection_name).count
    assert remaining == 0, f"{remaining} point(s) written by a run started mid-delete"
    assert state.get_project(ctx.conn, project_id) is None


def test_orphan_sweep_removes_dead_projects_points(tmp_path, repo):
    """The startup backstop: points whose project vanished from SQLite (a
    killed process, a delete that raced a write) are swept; live projects'
    points are untouched."""
    ctx = make_ctx(tmp_path)
    live = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)
    run_id, _ = state.try_start_run(ctx.conn, live)
    asyncio.run(
        execute_run(ctx.conn, ctx.store, ctx.embedder, str(repo), live, run_id)
    )
    live_points = ctx.store._client.count(ctx.store.collection_name).count
    assert live_points > 0

    # Forge an orphan: points tagged with a project_id SQLite has never heard
    # of — indistinguishable from what a killed mid-run process leaves.
    dead = "dead0000000000000000000000000000"
    run2, _ = state.try_start_run(ctx.conn, live)
    asyncio.run(execute_run(ctx.conn, ctx.store, ctx.embedder, str(repo), dead, run2))
    assert ctx.store._client.count(ctx.store.collection_name).count > live_points

    swept = ctx.store.delete_orphan_points([r["id"] for r in state.list_projects(ctx.conn)])
    assert swept > 0
    assert ctx.store._client.count(ctx.store.collection_name).count == live_points


def test_orphan_sweep_refuses_to_run_on_an_empty_project_table(tmp_path, repo):
    """Destructive-op guard: an empty project table is indistinguishable from
    a state DB opened at the wrong path (the 2026-07-11 cwd-relative db_path
    bug). Sweeping there would delete the entire index, so it must no-op."""
    ctx = make_ctx(tmp_path)
    pid = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)
    run_id, _ = state.try_start_run(ctx.conn, pid)
    asyncio.run(execute_run(ctx.conn, ctx.store, ctx.embedder, str(repo), pid, run_id))
    before = ctx.store._client.count(ctx.store.collection_name).count
    assert before > 0

    # Drop the row, keep the points: a populated collection with an empty
    # project table is exactly what a wrong-path state DB looks like.
    state.delete_project(ctx.conn, pid)
    assert state.list_projects(ctx.conn) == []

    swept = ctx.store.delete_orphan_points(
        [r["id"] for r in state.list_projects(ctx.conn)]
    )
    assert swept == 0, "sweep must refuse an empty project table"
    assert ctx.store._client.count(ctx.store.collection_name).count == before


# --- 4. empty query rejected at the validation boundary ---------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rest_rejects_blank_query(client, repo, query):
    body = client.post("/projects", json={"root_path": str(repo)}).json()
    resp = client.post(
        "/search", json={"query": query, "project_id": body["project_id"]}
    )
    assert resp.status_code == 422, f"{query!r} accepted"


def test_rest_accepts_nonblank_query(client, repo):
    """The min_length guard must not reject a legitimate one-character query."""
    body = client.post("/projects", json={"root_path": str(repo)}).json()
    resp = client.post("/search", json={"query": "x", "project_id": body["project_id"]})
    assert resp.status_code == 200, resp.text


@pytest.fixture()
def mcp_server(tmp_path):
    ctx = make_ctx(tmp_path)
    return build_mcp(lambda: ctx), ctx


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_mcp_rejects_blank_query(mcp_server, repo, query):
    """The MCP twin must refuse what REST refuses (§ M6 surface parity)."""
    mcp, ctx = mcp_server
    project_id = state.register_project(ctx.conn, str(repo), ctx.embedder.model_id)
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "search_code", {"query": query, "project_id": project_id}
            )


async def test_mcp_project_id_still_required(mcp_server):
    """``project_id: str = Field()`` is what keeps the signature legal once
    ``query`` carries a Field default. Field() supplies no default value, so
    project_id must stay REQUIRED — if it ever silently acquired one, the
    tool would receive a FieldInfo object instead of an id."""
    mcp, _ = mcp_server
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        schema = tools["search_code"].inputSchema
        assert set(schema["required"]) == {"query", "project_id"}
        assert schema["properties"]["query"]["minLength"] == 1

        with pytest.raises(ToolError):
            await client.call_tool("search_code", {"query": "x"})
