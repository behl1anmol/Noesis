"""Job manager — one place both adapters launch and inspect index runs.

Part of the Core Engine's "Job/State manager" box (§3.1). REST
(`POST /projects`, `POST /projects/{id}/reindex`) and the MCP `reindex`
tool must behave identically (two thin adapters over one core), so this
module owns the single launch path: open a run row, hand back ids
immediately, and index in a background task tracked in ``ctx.jobs`` so the
app lifespan can cancel orphans on shutdown.

M8 additions (ADR-40): watcher-scoped launches (``paths`` narrows the
candidate set, ``triggered_by`` records provenance), in-memory live
progress per run (``ctx.progress``, read by the dashboard — deliberately
not persisted: live progress of a dead process is meaningless), and
pending-change clearing once a run has examined the files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

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
    progress: dict[str, dict[str, Any]]
    git_fast_path: bool
    embed_batch_size: int


def launch_index_run(
    ctx: _ContextLike,
    root_path: str,
    *,
    paths: Sequence[str] | None = None,
    triggered_by: str = "manual",
) -> dict[str, str]:
    """Register (or re-open) the project and start indexing in the
    background. Returns the 202-style acceptance body shared verbatim by
    REST and MCP.

    Fails fast with ValueError before touching state when *root_path* is
    missing or not a directory (an agent gets the answer now instead of
    polling a background failure), and on the mixed-model guard
    (state.register_project). If a run is already ``running`` for this
    project, launches nothing (a second concurrent index would race on the
    same collection and state rows) and returns that run's id with
    ``status: "already_running"`` — a distinct status so a scoped/watcher
    caller can tell its launch was skipped and re-arm its retry (H3). A
    real launch returns ``status: "accepted"``."""
    if not os.path.isdir(root_path):
        raise ValueError(f"root_path is not an existing directory: {root_path!r}")
    project_id = state.register_project(ctx.conn, root_path, ctx.embedder.model_id)
    # Atomic check-and-insert (state.try_start_run, BEGIN IMMEDIATE): a plain
    # read-then-insert is race-free on one event loop but not across the
    # dual-transport deployment (HTTP + stdio MCP sharing this DB), where two
    # near-simultaneous launches could both pass the check and race two index
    # runs onto the same collection.
    run_id, created = state.try_start_run(
        ctx.conn, project_id, triggered_by=triggered_by
    )
    if not created:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "already_running",
        }
    # Cut point for the pending clear: rows re-dirtied after this survive.
    started_iso = datetime.now(timezone.utc).isoformat()
    ctx.progress[run_id] = {
        "files_done": None,
        "files_to_index": None,
        "chunks_written": 0,
        "monotonic_start": time.monotonic(),
    }

    def _on_progress(done: int, total: int, chunks: int) -> None:
        entry = ctx.progress.get(run_id)
        if entry is not None:
            entry["files_done"] = done
            entry["files_to_index"] = total
            entry["chunks_written"] = chunks

    async def _run() -> None:
        try:
            result = await execute_run(
                ctx.conn,
                ctx.store,
                ctx.embedder,
                root_path,
                project_id,
                run_id,
                batch_size=ctx.embed_batch_size,
                git_fast_path=ctx.git_fast_path,
                paths=paths,
                on_progress=_on_progress,
            )
        except Exception:
            # execute_run already marked the run failed; log for the operator.
            logger.exception("index run %s failed", run_id)
        else:
            # The run examined these files — clear their pending rows, but
            # only up to the launch timestamp so a file re-dirtied mid-run
            # stays pending (resolved toward re-examination, ADR-40).
            state.clear_pending_changes(
                ctx.conn, project_id, paths=paths, before=started_iso
            )
            # Files that failed per-file containment (ADR-41) go straight
            # back to pending: their state rows are stale, and without a
            # pending row the watcher would never auto-retry them — only a
            # manual full run would (PR #10 review). No hot loop: a retry
            # fires only on the next event/quiet cycle or manual action.
            if result.failed_paths:
                state.upsert_pending_changes(
                    ctx.conn,
                    project_id,
                    [(p, "modified") for p in result.failed_paths],
                )

    task = asyncio.create_task(_run())
    ctx.jobs[run_id] = task

    def _cleanup(_t: asyncio.Task) -> None:
        ctx.jobs.pop(run_id, None)
        ctx.progress.pop(run_id, None)

    task.add_done_callback(_cleanup)
    return {"project_id": project_id, "run_id": run_id, "status": "accepted"}


def run_progress(ctx: _ContextLike, run_id: str) -> dict[str, Any] | None:
    """Live progress for a running run: percent, counts, elapsed, ETA.
    None when the run is not tracked (finished, or another process's).
    ETA is naive linear extrapolation over files processed — honest enough
    for a progress bar, no smoothing pretence."""
    entry = ctx.progress.get(run_id)
    if entry is None:
        return None
    done = entry["files_done"]
    total = entry["files_to_index"]
    elapsed = time.monotonic() - entry["monotonic_start"]
    percent: float | None = None
    eta: float | None = None
    if done is not None and total is not None:
        percent = 100.0 if total == 0 else round(done / total * 100.0, 1)
        if done > 0 and total > done:
            eta = round(elapsed / done * (total - done), 1)
        elif total == done:
            eta = 0.0
    return {
        "files_done": done,
        "files_to_index": total,
        "chunks_written": entry["chunks_written"],
        "percent": percent,
        "elapsed_s": round(elapsed, 1),
        "eta_s": eta,
    }


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
