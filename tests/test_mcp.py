"""M6 MCP adapter tests — the six tools, response identity with REST, and
the mounted-transport lifespan.

Identity is the milestone's core assertion: MCP and REST are thin wrappers
over the same core functions, so for the same query the two surfaces must
return byte-identical bodies. The in-memory fastmcp Client talks to a
server built over the *same* AppContext the TestClient's app uses — one
core, two adapters, compared directly.

The uvicorn test exists to catch the mounted-transport failure mode the
milestone names explicitly: serving ``/mcp`` without running the MCP
app's lifespan raises "Task group is not initialized". It boots the real
combined app over real HTTP and completes an MCP round-trip.
"""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest
import uvicorn
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore
from noesis.mcp import build_mcp

EXPECTED_TOOLS = {
    "search_code",
    "structural_search",
    "list_projects",
    "get_index_status",
    "get_chunk",
    "reindex",
}


@pytest.fixture()
def project_dir(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "auth.py").write_text(
        "def validate_token(token):\n"
        '    """Check JWT expiry before trusting claims."""\n'
        "    return token.expiry > now()\n"
    )
    (src / "db.py").write_text(
        "def connect(dsn):\n"
        "    return Driver(dsn)\n"
    )
    return src


@pytest.fixture()
def ctx(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


@pytest.fixture()
def rest(ctx):
    """REST surface over the shared context."""
    with TestClient(create_app(ctx=ctx)) as tc:
        yield tc


@pytest.fixture()
def mcp(ctx):
    """MCP surface over the same shared context as ``rest``."""
    return build_mcp(lambda: ctx)


async def _wait_done(rest: TestClient, project_id: str, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        body = rest.get(f"/projects/{project_id}/status").json()
        if body["status"] in ("done", "failed"):
            assert body["status"] == "done", body
            return
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"run still {body['status']}")
        await asyncio.sleep(0.02)


async def _indexed_project(rest: TestClient, project_dir) -> str:
    resp = rest.post("/projects", json={"root_path": str(project_dir)})
    assert resp.status_code == 202
    project_id = resp.json()["project_id"]
    await _wait_done(rest, project_id)
    return project_id


async def test_lists_exactly_the_six_m6_tools(mcp):
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS


async def test_search_code_identical_to_rest(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    request = {"query": "jwt expiry check", "project_id": project_id, "top_k": 5}
    rest_body = rest.post("/search", json=request).json()
    async with Client(mcp) as client:
        result = await client.call_tool("search_code", request)
    assert result.structured_content == rest_body
    assert rest_body["hits"], "identity test must compare non-empty hits"
    assert all(hit["chunk_id"] for hit in rest_body["hits"])


async def test_structural_search_identical_to_rest(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    request = {
        "pattern": "def $NAME($$$ARGS): $$$BODY",
        "language": "python",
        "project_id": project_id,
    }
    rest_body = rest.post("/structural-search", json=request).json()
    async with Client(mcp) as client:
        result = await client.call_tool("structural_search", request)
    assert result.structured_content == rest_body
    assert rest_body["matches"], "identity test must compare non-empty matches"


async def test_list_projects_identical_to_rest(rest, mcp, project_dir):
    await _indexed_project(rest, project_dir)
    rest_body = rest.get("/projects").json()
    async with Client(mcp) as client:
        result = await client.call_tool("list_projects")
    assert result.data == rest_body
    assert rest_body, "identity test must compare a non-empty listing"


async def test_get_index_status_identical_to_rest(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    rest_body = rest.get(f"/projects/{project_id}/status").json()
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_index_status", {"project_id": project_id}
        )
    assert result.structured_content == rest_body
    assert rest_body["status"] == "done"


async def test_reindex_matches_rest_contract(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    rest_body = rest.post(f"/projects/{project_id}/reindex").json()
    await _wait_done(rest, project_id)
    async with Client(mcp) as client:
        result = await client.call_tool("reindex", {"project_id": project_id})
    body = result.structured_content
    # run_id is fresh per launch, so assert contract equality, not bytes.
    assert body.keys() == rest_body.keys()
    assert body["project_id"] == rest_body["project_id"] == project_id
    assert body["status"] == rest_body["status"] == "accepted"
    assert body["run_id"] != rest_body["run_id"]
    await _wait_done(rest, project_id)


async def test_get_chunk_roundtrip_from_search_hit(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    hit = rest.post(
        "/search", json={"query": "connect driver", "project_id": project_id}
    ).json()["hits"][0]
    async with Client(mcp) as client:
        result = await client.call_tool("get_chunk", {"chunk_id": hit["chunk_id"]})
    chunk = result.structured_content
    assert chunk["chunk_id"] == hit["chunk_id"]
    assert chunk["project_id"] == project_id
    assert chunk["file_path"] == hit["file_path"]
    assert chunk["start_line"] == hit["start_line"]
    assert chunk["end_line"] == hit["end_line"]
    # The stored content is the full chunk text the snippet was cut from.
    assert chunk["content"].startswith(hit["snippet"])


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("search_code", {"query": "q", "project_id": "nope"}, "unknown project_id"),
        ("get_index_status", {"project_id": "nope"}, "unknown project_id"),
        ("reindex", {"project_id": "nope"}, "unknown project_id"),
        ("get_chunk", {"chunk_id": "not-a-point-id"}, "unknown chunk_id"),
    ],
)
async def test_unknown_ids_raise_tool_errors(mcp, tool, args, message):
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match=message):
            await client.call_tool(tool, args)


async def test_structural_errors_carry_type_and_diagnostic(rest, mcp, project_dir):
    project_id = await _indexed_project(rest, project_dir)
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="unsupported_language"):
            await client.call_tool(
                "structural_search",
                {"pattern": "x", "language": "sql", "project_id": project_id},
            )


async def test_mounted_http_transport_lifespan(ctx, project_dir):
    """Boot the combined app under real uvicorn and complete an MCP
    round-trip over streamable HTTP. Fails with "Task group is not
    initialized" if the MCP app's lifespan is not part of the FastAPI
    lifespan (the exact wiring mistake M6 names)."""
    app = create_app(ctx=ctx)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = asyncio.get_event_loop().time() + 10
        while not server.started:
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("uvicorn did not start")
            await asyncio.sleep(0.05)
        async with Client(f"http://127.0.0.1:{port}/mcp/") as client:
            tools = await client.list_tools()
            assert {t.name for t in tools} == EXPECTED_TOOLS
            result = await client.call_tool("list_projects")
            assert result.data == []
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
