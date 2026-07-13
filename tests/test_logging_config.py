"""Tests for the central logging setup and the new runtime log narration.

Covers noesis.logging_config (idempotency, level/format env knobs, stderr
target) and the ADR-25 content rule: no query text or file contents may reach
any log line. Log-capture assertions follow the existing repo pattern
(tests/test_reranker.py): caplog.at_level(LEVEL, logger="noesis...").
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from noesis import logging_config as lc
from noesis.core import state
from noesis.core.embedder import FakeEmbedder
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.vectorstore import VectorStore


# --- logging_config unit tests ----------------------------------------------


def _noesis_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, "_noesis_tag", None)]


def test_configure_logging_attaches_single_stderr_handler(monkeypatch):
    monkeypatch.delenv(lc.LEVEL_ENV, raising=False)
    monkeypatch.delenv(lc.FORMAT_ENV, raising=False)
    log = lc.configure_logging()
    assert log.name == "noesis"
    assert log.level == logging.INFO
    handlers = _noesis_handlers(log)
    assert len(handlers) == 1
    # stderr only — stdout carries the stdio MCP JSON-RPC stream (rule 2).
    assert handlers[0].stream is sys.stderr
    # propagate must stay True so pytest's root-level caplog still captures.
    assert log.propagate is True


def test_propagate_defaults_true_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv(lc.LEVEL_ENV, raising=False)
    # Default keeps propagation so pytest caplog (root handler) still captures.
    log = lc.configure_logging()
    assert log.propagate is True
    # stdio MCP opts out: an already-configured logger must still flip, so a
    # root handler bound to stdout never receives noesis records (P2 review).
    log = lc.configure_logging(propagate=False)
    assert log.propagate is False
    # ...and the stderr handler is not duplicated by the second call.
    assert len(_noesis_handlers(log)) == 1
    # restore for other tests that rely on the default
    lc.configure_logging(propagate=True)


def test_configure_logging_is_idempotent(monkeypatch):
    monkeypatch.delenv(lc.LEVEL_ENV, raising=False)
    log = logging.getLogger("noesis")
    for _ in range(5):
        lc.configure_logging()
    assert len(_noesis_handlers(log)) == 1


def test_level_env_is_respected(monkeypatch):
    monkeypatch.setenv(lc.LEVEL_ENV, "debug")
    log = lc.configure_logging()
    assert log.level == logging.DEBUG


def test_bad_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv(lc.LEVEL_ENV, "NONSENSE")
    log = lc.configure_logging()
    assert log.level == logging.INFO


def test_json_format_emits_parseable_lines(monkeypatch):
    monkeypatch.setenv(lc.FORMAT_ENV, "json")
    log = lc.configure_logging()
    handler = _noesis_handlers(log)[0]
    rec = logging.LogRecord(
        "noesis.core.x", logging.INFO, "x.py", 1, "hello %s", ("world",), None
    )
    obj = json.loads(handler.formatter.format(rec))
    assert obj["msg"] == "hello world"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "noesis.core.x"


# --- index-run narration + ADR-25 content rule ------------------------------


def _make_env(tmp_path: Path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return conn, store, embedder


def _index(conn, store, embedder, root: Path):
    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    return asyncio.run(run())


def test_index_run_logs_start_and_finish_milestones(tmp_path, caplog):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n")
    (root / "b.py").write_text("def b():\n    return 2\n")
    conn, store, embedder = _make_env(tmp_path)

    with caplog.at_level(logging.INFO, logger="noesis.core.indexer"):
        result = _index(conn, store, embedder, root)

    assert result.files_indexed == 2
    text = caplog.text
    # Start milestone carries the correct to_index count...
    assert "to_index=2" in text
    assert "discovered=2" in text
    # ...and the run announces completion with the chunk count and status.
    assert "finished: status=done" in text
    assert "files_indexed=2" in text


def test_logs_never_leak_file_contents_or_paths_at_info(tmp_path, caplog):
    """ADR-25: file contents never appear in logs, and file paths stay at
    DEBUG (absent from the INFO stream)."""
    sentinel_code = "SUPER_SECRET_PROPRIETARY_TOKEN_42"
    sentinel_name = "confidential_module.py"
    root = tmp_path / "repo"
    root.mkdir()
    (root / sentinel_name).write_text(f"value = '{sentinel_code}'\n")
    conn, store, embedder = _make_env(tmp_path)

    with caplog.at_level(logging.INFO, logger="noesis.core.indexer"):
        _index(conn, store, embedder, root)

    assert sentinel_code not in caplog.text  # never, at any level
    assert sentinel_name not in caplog.text  # path is DEBUG-only, not at INFO
