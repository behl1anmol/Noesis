"""Usage telemetry — metadata-only query logging (ADR-40).

Feeds the dashboard's usage page. Records *that* a query happened and how
it performed (interface, kind, channel, latency, result count) — never the
query text: queries routinely quote proprietary code, and a local DB is
still a file that gets backed up, synced, and pasted into bug reports
(ADR-25 spirit). Logging is fire-and-forget: a telemetry failure must never
fail the search that triggered it.

Writes go through a dedicated best-effort connection, not the shared
ctx.conn: in the dual-transport deployment (HTTP server + stdio MCP sharing
one DB file) the other process's index-run writes hold sqlite's writer
lock, and an INSERT on ctx.conn (busy_timeout=5000) would stall the event
loop up to 5s per search. The dedicated connection waits at most 250ms and
then drops the row — telemetry may lose a row, never slow a query — and the
INSERT itself runs in a worker thread so the loop is never blocked at all.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from typing import Any

from . import state

logger = logging.getLogger(__name__)

# Process-lifetime by design: telemetry has no teardown hook, and one sqlite
# handle per DB path is cheap. Keyed by path so tests opening fresh DBs in
# one process each get their own connection. The lock serializes both lazy
# connection creation and the INSERT (one connection, many loop tasks
# offloading here concurrently).
_conn_lock = threading.Lock()
_conns: dict[str, sqlite3.Connection] = {}


def _dedicated_conn(db_path: str) -> sqlite3.Connection:
    conn = _conns.get(db_path)
    if conn is None:
        # Same sanctioned constructor as the shared conn (state.connect is
        # the one place sqlite3.connect appears) — only the busy_timeout
        # diverges: telemetry waits at most 250ms, then drops the row.
        conn = state.connect(db_path)
        conn.execute("PRAGMA busy_timeout=250")
        _conns[db_path] = conn
    return conn


def _write_locked(db_path: str, fields: dict[str, Any]) -> None:
    with _conn_lock:
        state.log_query(_dedicated_conn(db_path), **fields)


async def record_query(
    conn: sqlite3.Connection,
    *,
    interface: str,
    kind: str,
    project_id: str | None,
    channel: str | None = None,
    reranked: bool | None = None,
    latency_ms: float | None = None,
    result_count: int | None = None,
) -> None:
    fields: dict[str, Any] = dict(
        interface=interface,
        kind=kind,
        project_id=project_id,
        channel=channel,
        reranked=reranked,
        latency_ms=latency_ms,
        result_count=result_count,
    )
    try:
        # Quick read on the caller's loop thread: resolve the DB file behind
        # the shared conn so the dedicated connection opens the same DB.
        row = conn.execute("PRAGMA database_list").fetchone()
        db_path = row[2] if row is not None else ""
        if not db_path:
            # In-memory DB: a second connection cannot reach it — write on
            # the shared conn as before (test-only shape; runtime DBs are
            # always files).
            state.log_query(conn, **fields)
            return
        # Off the event loop: under a contending writer this blocks its
        # worker thread up to the dedicated conn's 250ms busy_timeout —
        # never the loop, never ctx.conn's 5s.
        await asyncio.to_thread(_write_locked, db_path, fields)
    except Exception:  # noqa: BLE001 — telemetry must never break the query path
        logger.debug("query telemetry write failed", exc_info=True)
