"""Job manager — one place both adapters launch and inspect index runs.

Part of the Core Engine's "Job/State manager" box (§3.1). REST
(`POST /projects`, `POST /projects/{id}/reindex`) and the MCP `reindex`
tool must behave identically (two thin adapters over one core), so this
module owns the single launch path: open a run row, hand back ids
immediately, and index in a background task tracked in ``ctx.jobs`` so the
app lifespan can cancel orphans on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Protocol

from noesis.core import state
from noesis.core.indexer import execute_run

logger = logging.getLogger(__name__)


class _ContextLike(Protocol):
    """The AppContext attributes this module needs (duck-typed to avoid an
    import cycle: app.py builds the MCP server, which uses this module)."""

    conn: Any
    store: Any
    embedder: Any
    jobs: dict[str, asyncio.Task]
    git_fast_path: bool


def launch_index_run(ctx: _ContextLike, root_path: str) -> dict[str, str]:
    """Register (or re-open) the project and start indexing in the
    background. Returns the 202-style acceptance body shared verbatim by
    REST and MCP.

    Fails fast with ValueError before touching state when *root_path* is
    missing or not a directory (an agent gets the answer now instead of
    polling a background failure), and on the mixed-model guard
    (state.register_project). If a run is already ``running`` for this
    project, returns that run's id rather than launching a second
    concurrent index racing on the same collection and state rows."""
    if not os.path.isdir(root_path):
        raise ValueError(f"root_path is not an existing directory: {root_path!r}")
    project_id = state.register_project(ctx.conn, root_path, ctx.embedder.model_id)
    latest = state.get_latest_run(ctx.conn, project_id)
    if latest is not None and latest["status"] == "running":
        return {
            "project_id": project_id,
            "run_id": latest["id"],
            "status": "accepted",
        }
    run_id = state.start_run(ctx.conn, project_id)

    async def _run() -> None:
        try:
            await execute_run(
                ctx.conn,
                ctx.store,
                ctx.embedder,
                root_path,
                project_id,
                run_id,
                git_fast_path=ctx.git_fast_path,
            )
        except Exception:
            # execute_run already marked the run failed; log for the operator.
            logger.exception("index run %s failed", run_id)

    task = asyncio.create_task(_run())
    ctx.jobs[run_id] = task
    task.add_done_callback(lambda _t: ctx.jobs.pop(run_id, None))
    return {"project_id": project_id, "run_id": run_id, "status": "accepted"}


def index_status(ctx: _ContextLike, project_id: str) -> dict[str, Any]:
    """Latest run for a project, shaped identically for REST and MCP.

    A registered project with no runs yet reports ``never_indexed`` with
    the run fields nulled — a stable shape beats a shape-shifting one for
    agent consumers."""
    run = state.get_latest_run(ctx.conn, project_id)
    if run is None:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "never_indexed",
            "files_total": None,
            "files_changed": None,
            "chunks_written": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
    return {
        "project_id": project_id,
        "run_id": run["id"],
        "status": run["status"],
        "files_total": run["files_total"],
        "files_changed": run["files_changed"],
        "chunks_written": run["chunks_written"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "error": run["error"],
    }
