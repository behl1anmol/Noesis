"""PR #50 round-3 review: close_runtime_context's teardown loop
(embedder, reranker, store) had no per-resource exception isolation. If one
resource's close() raised, the exception propagated straight out, skipping
every later resource's close() *and* the telemetry/conn cleanup after the
loop — a leak on top of whatever the original close() failure was."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from noesis.core import state
from noesis.runtime import AppContext, close_runtime_context


def _ctx(tmp_path, **overrides) -> AppContext:
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    base = dict(conn=conn, store=Mock(), embedder=Mock())
    base.update(overrides)
    return AppContext(**base)


async def test_close_runtime_context_isolates_resource_close_failures(tmp_path):
    embedder = Mock()
    embedder.close.side_effect = RuntimeError("embedder close boom")
    reranker = Mock()
    store = Mock()

    ctx = _ctx(tmp_path, embedder=embedder, reranker=reranker, store=store)
    conn = ctx.conn

    await close_runtime_context(ctx)  # must not raise despite embedder.close() failing

    embedder.close.assert_called_once()
    reranker.close.assert_called_once()
    store.close.assert_called_once()
    # conn.close() ran despite the earlier failure — a closed sqlite
    # connection raises ProgrammingError on use.
    with pytest.raises(Exception):
        conn.execute("SELECT 1")
