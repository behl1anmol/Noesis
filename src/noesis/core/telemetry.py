"""Usage telemetry — metadata-only query logging (ADR-40).

Feeds the dashboard's usage page. Records *that* a query happened and how
it performed (interface, kind, channel, latency, result count) — never the
query text: queries routinely quote proprietary code, and a local DB is
still a file that gets backed up, synced, and pasted into bug reports
(ADR-25 spirit). Logging is fire-and-forget: a telemetry failure must never
fail the search that triggered it.
"""

from __future__ import annotations

import logging
import sqlite3

from . import state

logger = logging.getLogger(__name__)


def record_query(
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
    try:
        state.log_query(
            conn,
            interface=interface,
            kind=kind,
            project_id=project_id,
            channel=channel,
            reranked=reranked,
            latency_ms=latency_ms,
            result_count=result_count,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the query path
        logger.debug("query telemetry write failed", exc_info=True)
