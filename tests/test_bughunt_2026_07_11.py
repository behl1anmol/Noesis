"""Regression tests for the 2026-07-11 bug hunt.

Findings closed here (severity order):

1. `execute_run` ran its heavy synchronous spans — discovery, hash
   partition, file read/chunk, Qdrant upsert/delete, gitfast subprocess
   calls — directly on the event loop, so every concurrent search /
   dashboard / MCP request stalled while a project indexed, contradicting
   §3.8 ("live queries preempt indexing"). Spans now run via
   ``asyncio.to_thread``.
2. `chunk_embed_all_failed` (indexer.py) lacked the remainder guard the
   sibling hash check got in PR #14: a single transient embed error on a
   narrow candidate set marked the whole run "failed" even though the
   rest of the tree was known-healthy via verified/skipped.
3. Default ``db_path``/``config.toml`` were cwd-relative, so the stdio
   MCP server (spawned with the agent host's cwd) silently created a
   fresh empty DB instead of the one the dashboard uses. Both are now
   anchored (XDG), mirroring prefetch.py's fastembed-cache fix.
4. `init_db`'s check-then-act migration loop raced across processes:
   concurrent first startups could crash one with "duplicate column
   name". The loop now runs single-writer (BEGIN IMMEDIATE) with a
   duplicate-column guard.
5. A NaN reranker score broke the rerank sort's total order and could
   silently drop the best hit from top-k. Sort key now maps NaN to -inf.
6. The dashboard cache-buster ``asset_ver`` was frozen at import, so
   in-place asset swaps kept serving stale scripts. Now computed per
   render.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.vectorstore import VectorStore


def make_env(tmp_path: Path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return conn, store, embedder


# --- 1. index runs must not starve the event loop ----------------------------


class SlowStore:
    """VectorStore wrapper whose upsert blocks the calling thread, standing
    in for a slow Qdrant round-trip (or any heavy sync span in the run)."""

    def __init__(self, inner: VectorStore, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    def upsert_chunks(self, *args, **kwargs):
        time.sleep(self._delay)
        return self._inner.upsert_chunks(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.asyncio
async def test_execute_run_does_not_starve_the_event_loop(tmp_path):
    """While a run sits in a blocking store call, other coroutines must
    keep getting scheduled — the longest observed gap between heartbeat
    ticks stays well under the blocking span's duration."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n")
    (root / "b.py").write_text("def b():\n    return 2\n")

    conn, store, embedder = make_env(tmp_path)
    slow = SlowStore(store, delay=0.5)
    project_id, run_id = prepare_run(conn, embedder, str(root))

    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    try:
        result = await execute_run(
            conn, slow, embedder, str(root), project_id, run_id, git_fast_path=False
        )
    finally:
        stop.set()
        await hb

    assert result.files_indexed == 2  # the run itself still works
    # Two 0.5s blocking upserts happened; had they run on the loop thread
    # the heartbeat would show a ~0.5s gap (or, with a never-yielding
    # embedder, no tick at all). Generous margin for CI jitter.
    assert gaps, "event loop starved: heartbeat never ticked during the run"
    assert max(gaps) < 0.4, f"event loop starved: max heartbeat gap {max(gaps):.3f}s"


# --- 2. one contained embed error must not mark a healthy run "failed" -------


class ExplodingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed outage")


def _index(conn, store, embedder, root, **kwargs):
    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, **kwargs
        )

    return asyncio.run(run())


def test_scoped_single_file_embed_failure_stays_done(tmp_path):
    """Watcher-scoped rerun of one edited file: a transient embed error on
    that file is a contained per-file failure — the other file is healthy
    via `skipped`, so the run must finish "done", mirroring the hash-path
    guard from PR #14 (indexer.py's own stated principle)."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    first = _index(conn, store, embedder, root, git_fast_path=False)
    assert first.files_indexed == 2

    (root / "a.py").write_text("x = 99\n")
    second = _index(conn, store, ExplodingEmbedder(dim=8), root, paths=["a.py"])

    assert second.files_failed == 1
    assert second.failed_paths == ("a.py",)
    run_row = state.get_latest_run(conn, second.project_id)
    assert run_row["status"] == "done"  # pre-fix: "failed" / "all 1 files failed"
    assert run_row["error"] is None


def test_full_walk_single_file_embed_failure_stays_done(tmp_path):
    """Same containment on a full walk: one changed file failing to embed
    among N-1 verified-unchanged files is partial failure (ADR-41), not a
    run-wide outage."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    first = _index(conn, store, embedder, root, git_fast_path=False)
    assert first.files_indexed == 2

    (root / "a.py").write_text("x = 99\n")
    second = _index(conn, store, ExplodingEmbedder(dim=8), root, git_fast_path=False)

    assert second.files_failed == 1
    run_row = state.get_latest_run(conn, second.project_id)
    assert run_row["status"] == "done"


def test_total_embed_outage_still_marks_run_failed(tmp_path):
    """The real-outage shape is preserved: a first index where every file
    fails to embed has no known-good remainder and must stay "failed"."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, _ = make_env(tmp_path)

    result = _index(conn, store, ExplodingEmbedder(dim=8), root, git_fast_path=False)

    assert result.files_failed == 2
    run_row = state.get_latest_run(conn, result.project_id)
    assert run_row["status"] == "failed"
    assert run_row["error"] == "all 2 files failed"


# --- 3. db_path/config resolution must not depend on the process cwd ---------


def test_default_db_path_is_anchored_and_cwd_independent(tmp_path, monkeypatch):
    """Two processes launched from different cwds (uvicorn from the repo,
    stdio MCP from the agent host's cwd) must resolve the SAME state DB."""
    from noesis.core.config import load_settings

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("NOESIS_CONFIG", raising=False)

    cwd_a = tmp_path / "repo-checkout"
    cwd_b = tmp_path / "agent-host-cwd"
    cwd_a.mkdir()
    cwd_b.mkdir()

    monkeypatch.chdir(cwd_a)
    db_a = load_settings().db_path
    monkeypatch.chdir(cwd_b)
    db_b = load_settings().db_path

    assert db_a == db_b == tmp_path / "xdg-data" / "noesis" / "noesis.sqlite"
    assert db_a.is_absolute()


def test_noesis_config_env_points_at_the_config_file(tmp_path, monkeypatch):
    """$NOESIS_CONFIG lets a host that can't control its cwd pin the config;
    a relative db_path inside resolves against the config file's directory,
    not the cwd."""
    from noesis.core.config import load_settings

    cfg_dir = tmp_path / "custom"
    cfg_dir.mkdir()
    (cfg_dir / "my.toml").write_text('db_path = "state/db.sqlite"\n')
    monkeypatch.setenv("NOESIS_CONFIG", str(cfg_dir / "my.toml"))
    monkeypatch.chdir(tmp_path)  # cwd is elsewhere on purpose

    settings = load_settings()

    assert settings.db_path == cfg_dir / "state" / "db.sqlite"


def test_cwd_config_toml_remains_a_dev_override(tmp_path, monkeypatch):
    """Running from a checkout that carries a config.toml keeps working —
    the cwd file is a deliberate override, looked up before the XDG path."""
    from noesis.core.config import load_settings

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("NOESIS_CONFIG", raising=False)
    (tmp_path / "config.toml").write_text('[qdrant]\ncollection = "dev_chunks"\n')
    monkeypatch.chdir(tmp_path)

    assert load_settings().qdrant.collection == "dev_chunks"


# --- 5. a NaN rerank score must not corrupt the returned ordering ------------


def test_nan_rerank_score_ranks_last_not_random(tmp_path):
    """A cross-encoder can emit NaN (fp16 overflow, degenerate text). NaN
    breaks the sort's total order — every comparison is False — so pre-fix
    the 'sorted' order was arbitrary and could slice away the genuinely
    best hit. NaN must rank last; finite scores keep exact order."""
    import math

    from noesis.core.retriever import search_code

    hits = [
        {
            "file_path": f"f{i}.py",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "symbol_name": None,
            "score": 0.5,
            "snippet": f"t{i}",
            "text": f"t{i}",
        }
        for i in range(3)
    ]

    class StubStore:
        def search(self, project_id, **kwargs):
            return [dict(h) for h in hits]

    class NaNReranker:
        model_id = "nan-stub"

        async def rerank(self, query, texts):
            return [0.9, float("nan"), 0.95]

    result = asyncio.run(
        search_code(
            StubStore(), FakeEmbedder(), "q", "p1", reranker=NaNReranker(), top_k=2
        )
    )

    scores = [h["rerank_score"] for h in result["hits"]]
    assert scores[0] == 0.95, f"best finite hit lost: {scores}"
    assert scores[1] == 0.9
    assert not any(math.isnan(s) for s in scores)


# --- 4. concurrent init_db must not crash on the migration race --------------


def test_concurrent_init_db_survives_migration_race(tmp_path):
    """Dual-transport startup (HTTP + stdio MCP) can run init_db on the same
    pre-migration DB simultaneously. Simulate the loser's view: writer A has
    ALTERed a migration column but not committed while B runs init_db — B's
    table_info read (WAL snapshot) says the column is absent, and pre-fix
    B's own ALTER then hit 'duplicate column name' (a schema error
    busy_timeout can't retry) and aborted B's startup."""
    import threading

    db = tmp_path / "state.sqlite"
    setup = state.connect(db)
    setup.executescript(state._SCHEMA)  # pre-migration: no watch_enabled etc.
    setup.commit()
    setup.close()

    writer = state.connect(db)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "ALTER TABLE projects ADD COLUMN watch_enabled INTEGER NOT NULL DEFAULT 0"
    )

    errors: list[BaseException] = []

    def loser() -> None:
        conn_b = state.connect(db)
        try:
            state.init_db(conn_b)  # blocks on writer's lock, then must not crash
        except BaseException as exc:  # noqa: BLE001 — recorded for the assert
            errors.append(exc)
        finally:
            conn_b.close()

    thread = threading.Thread(target=loser)
    thread.start()
    time.sleep(1.0)  # let B pass its schema read and block on the lock
    writer.commit()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert not errors, f"concurrent init_db crashed: {errors!r}"

    # Whoever won, the schema must be complete.
    check = state.connect(db)
    cols = {row[1] for row in check.execute("PRAGMA table_info(projects)")}
    assert {"watch_enabled", "auto_reindex"} <= cols
    check.close()
    writer.close()
