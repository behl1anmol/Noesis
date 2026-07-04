"""Structural search tests (§3.5): exact expected match sets on a fixture
repo, the skip-list negative from the M5 exit criterion, structured errors,
and the REST mirror. No index, no model, no Docker — structural search reads
the live filesystem, so a registry row is all the state it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import state
from noesis.core.config import StructuralSettings
from noesis.core.embedder import FakeEmbedder
from noesis.core.structural import StructuralSearchError, structural_search
from noesis.core.vectorstore import VectorStore

# Fixture repo. Line numbers in the exact-set assertions below are 1-based
# and tied to this content — keep them in sync when editing.
FIXTURE_FILES = {
    "app/db.py": (
        "import sqlite3\n"  # line 1
        "\n"
        "def query_users(db):\n"  # line 3
        "    return db.Exec('SELECT * FROM users')\n"  # line 4
        "\n"
        "def query_orders(db, ctx):\n"  # line 6
        "    return db.Exec('SELECT * FROM orders', ctx)\n"  # line 7
    ),
    "app/views.py": (
        "from app.db import query_users\n"  # line 1
        "\n"
        "def render_users(db):\n"  # line 3
        "    print('rendering')\n"  # line 4
        "    return query_users(db)\n"  # line 5
    ),
    "util.js": "function ping() { return db.Exec('SELECT 1'); }\n",
    # Secret-skip-listed: contains text that WOULD match db.Exec($$$ARGS) —
    # the exit-criterion negative asserts it never appears in results.
    ".env": "TOKEN = db.Exec('leak me')\n",
    "keys/service.pem": "db.Exec('also leak me')\n",
    # .gitignore'd file with a matching call — discovery reuse must drop it.
    ".gitignore": "generated.py\n",
    "generated.py": "x = db.Exec('ignored file')\n",
}


def make_repo(root: Path) -> None:
    for rel, content in FIXTURE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    make_repo(root)
    return root


@pytest.fixture()
def conn(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    yield conn
    conn.close()


@pytest.fixture()
def project_id(conn, repo):
    return state.register_project(conn, repo, "fake-model")


# --- exact expected match sets: 5 known patterns -------------------------


async def test_call_pattern_exact_set(conn, project_id):
    result = await structural_search(conn, project_id, "db.Exec($$$ARGS)", "python")
    got = {(m["file_path"], m["start_line"]) for m in result["matches"]}
    assert got == {("app/db.py", 4), ("app/db.py", 7)}
    assert result["truncated"] is False
    assert result["timed_out"] is False


async def test_function_def_pattern_exact_set(conn, project_id):
    result = await structural_search(
        conn, project_id, "def $NAME($$$PARAMS): $$$BODY", "python"
    )
    got = {(m["file_path"], m["meta_vars"]["NAME"]) for m in result["matches"]}
    assert got == {
        ("app/db.py", "query_users"),
        ("app/db.py", "query_orders"),
        ("app/views.py", "render_users"),
    }


async def test_import_pattern_exact_set(conn, project_id):
    result = await structural_search(conn, project_id, "import $MOD", "python")
    got = {(m["file_path"], m["start_line"], m["meta_vars"]["MOD"]) for m in result["matches"]}
    assert got == {("app/db.py", 1, "sqlite3")}


async def test_print_pattern_exact_set(conn, project_id):
    result = await structural_search(conn, project_id, "print($MSG)", "python")
    assert [(m["file_path"], m["start_line"], m["meta_vars"]["MSG"]) for m in result["matches"]] == [
        ("app/views.py", 4, "'rendering'")
    ]


async def test_javascript_pattern_scans_only_that_language(conn, project_id):
    result = await structural_search(conn, project_id, "db.Exec($$$ARGS)", "javascript")
    got = {(m["file_path"], m["start_line"]) for m in result["matches"]}
    assert got == {("util.js", 1)}
    assert result["scanned_files"] == 1  # only the .js file was read


# --- exit-criterion negative: skip-listed files never match ---------------


async def test_skip_listed_and_gitignored_files_never_match(conn, project_id):
    result = await structural_search(conn, project_id, "db.Exec($$$ARGS)", "python")
    files = {m["file_path"] for m in result["matches"]}
    assert ".env" not in files
    assert "keys/service.pem" not in files
    assert "generated.py" not in files  # .gitignore'd
    assert files == {"app/db.py"}


# --- multi-metavar capture -------------------------------------------------


async def test_multi_metavar_captures_named_nodes(conn, project_id):
    result = await structural_search(conn, project_id, "db.Exec($$$ARGS)", "python")
    by_line = {m["start_line"]: m["meta_vars"]["ARGS"] for m in result["matches"]}
    assert by_line[4] == ["'SELECT * FROM users'"]
    assert by_line[7] == ["'SELECT * FROM orders'", "ctx"]


# --- structured errors -----------------------------------------------------


async def test_unknown_project_error(conn):
    with pytest.raises(StructuralSearchError) as exc:
        await structural_search(conn, "nope", "print($X)", "python")
    assert exc.value.error_type == "unknown_project"


async def test_unsupported_language_error(conn, project_id):
    with pytest.raises(StructuralSearchError) as exc:
        await structural_search(conn, project_id, "anything", "toml")
    assert exc.value.error_type == "unsupported_language"
    assert "python" in exc.value.message  # message lists the supported set


async def test_pattern_error_carries_diagnostic(conn, project_id):
    with pytest.raises(StructuralSearchError) as exc:
        await structural_search(conn, project_id, "", "python")
    assert exc.value.error_type == "pattern_error"
    assert exc.value.message  # ast-grep's own diagnostic, non-empty


async def test_escaping_paths_rejected(conn, project_id):
    for bad in ["../outside", "/etc"]:
        with pytest.raises(StructuralSearchError) as exc:
            await structural_search(
                conn, project_id, "print($X)", "python", paths=[bad]
            )
        assert exc.value.error_type == "invalid_path"


# --- caps and budget --------------------------------------------------------


async def test_max_results_truncates(conn, project_id):
    result = await structural_search(
        conn, project_id, "def $NAME($$$PARAMS): $$$BODY", "python", max_results=1
    )
    assert len(result["matches"]) == 1
    assert result["truncated"] is True


async def test_request_cannot_raise_configured_cap(conn, project_id):
    settings = StructuralSettings(max_results=2)
    result = await structural_search(
        conn,
        project_id,
        "def $NAME($$$PARAMS): $$$BODY",
        "python",
        max_results=50,
        settings=settings,
    )
    assert len(result["matches"]) == 2
    assert result["truncated"] is True


async def test_paths_restriction(conn, project_id):
    result = await structural_search(
        conn,
        project_id,
        "def $NAME($$$PARAMS): $$$BODY",
        "python",
        paths=["app/views.py"],
    )
    assert {m["file_path"] for m in result["matches"]} == {"app/views.py"}


async def test_timeout_returns_partial_with_flag(conn, project_id):
    settings = StructuralSettings(timeout_s=0.0)
    result = await structural_search(
        conn, project_id, "print($X)", "python", settings=settings
    )
    assert result["timed_out"] is True
    assert result["matches"] == []  # budget expired before any file was read


# --- REST mirror -------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, repo):
    conn = state.connect(tmp_path / "api-state.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, repo, "fake-model")
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    ctx = AppContext(conn=conn, store=store, embedder=embedder)
    app = create_app(ctx=ctx)
    with TestClient(app) as tc:
        tc.project_id = pid
        yield tc
    conn.close()


def test_rest_structural_search_mirrors_core(client):
    resp = client.post(
        "/structural-search",
        json={
            "pattern": "db.Exec($$$ARGS)",
            "language": "python",
            "project_id": client.project_id,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    got = {(m["file_path"], m["start_line"]) for m in body["matches"]}
    assert got == {("app/db.py", 4), ("app/db.py", 7)}
    assert body["truncated"] is False
    assert body["timed_out"] is False


def test_rest_unknown_project_is_404(client):
    resp = client.post(
        "/structural-search",
        json={"pattern": "print($X)", "language": "python", "project_id": "nope"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["type"] == "unknown_project"


def test_rest_unsupported_language_is_400(client):
    resp = client.post(
        "/structural-search",
        json={"pattern": "x", "language": "sql", "project_id": client.project_id},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["type"] == "unsupported_language"


def test_rest_pattern_error_is_400_with_diagnostic(client):
    resp = client.post(
        "/structural-search",
        json={"pattern": "", "language": "python", "project_id": client.project_id},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["type"] == "pattern_error"
    assert detail["message"]
