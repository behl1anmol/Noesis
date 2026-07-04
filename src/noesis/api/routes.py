"""REST routes (Overview §8) — every handler is a thin call into core/.

M2 surface: register+index a project (202 + run_id, status polled), dense
``POST /search``, health. The MCP adapter (M6) wraps the same core
functions; the two surfaces must not drift.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from noesis.core import state
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.retriever import search_code

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterProjectRequest(BaseModel):
    root_path: str


class SearchRequest(BaseModel):
    query: str
    project_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    language: str | None = None


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/projects", status_code=202)
async def register_and_index(req: RegisterProjectRequest, request: Request) -> dict[str, str]:
    ctx = request.app.state.ctx
    try:
        project_id, run_id = prepare_run(ctx.conn, ctx.embedder, req.root_path)
    except ValueError as exc:  # mixed-model guard
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def _run() -> None:
        try:
            await execute_run(
                ctx.conn, ctx.store, ctx.embedder, req.root_path, project_id, run_id
            )
        except Exception:
            # execute_run already marked the run failed; log for the operator.
            logger.exception("index run %s failed", run_id)

    task = asyncio.create_task(_run())
    ctx.jobs[run_id] = task
    task.add_done_callback(lambda _t: ctx.jobs.pop(run_id, None))
    return {"project_id": project_id, "run_id": run_id, "status": "accepted"}


@router.get("/projects")
async def list_projects(request: Request) -> list[dict[str, Any]]:
    ctx = request.app.state.ctx
    return [dict(row) for row in state.list_projects(ctx.conn)]


@router.get("/runs/{run_id}")
async def run_status(run_id: str, request: Request) -> dict[str, Any]:
    ctx = request.app.state.ctx
    row = ctx.conn.execute(
        "SELECT * FROM index_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return dict(row)


@router.post("/search")
async def search(req: SearchRequest, request: Request) -> dict[str, Any]:
    ctx = request.app.state.ctx
    if state.get_project(ctx.conn, req.project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    hits = await search_code(
        ctx.store,
        ctx.embedder,
        req.query,
        req.project_id,
        top_k=req.top_k,
        language=req.language,
    )
    return {"query": req.query, "hits": hits}
