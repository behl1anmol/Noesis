"""REST routes (Overview §8) — every handler is a thin call into core/.

M3 surface: register+index a project (202 + run_id, status polled), hybrid
``POST /search`` (``channel`` selects hybrid/dense/sparse), health. M4 adds
the ``rerank`` flag (None → server default per ADR-34) and the ``reranked``
response field. M6 adds ``GET /projects/{id}/status`` and
``POST /projects/{id}/reindex`` (initial idea §8) via the shared core job
manager. The MCP adapter (noesis.mcp.server) wraps the same core functions;
tests assert the two surfaces return identical bodies.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from noesis.api.security import verify_local_origin
from noesis.core import jobs, state, telemetry
from noesis.core.retriever import search_code
from noesis.core.state import MixedModelError
from noesis.core.structural import StructuralSearchError, structural_search
from noesis.core.vectorstore import SearchChannel

router = APIRouter()


class RegisterProjectRequest(BaseModel):
    root_path: str


class SearchRequest(BaseModel):
    query: str
    project_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    language: str | None = None
    channel: SearchChannel = "hybrid"
    # None → server default (reranker availability, config reranker.enabled).
    rerank: bool | None = None


class StructuralSearchRequest(BaseModel):
    pattern: str
    language: str
    project_id: str
    paths: list[str] | None = None
    # None → config structural.max_results; requests may lower the cap, not raise it.
    max_results: int | None = Field(default=None, ge=1)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/projects", status_code=202)
async def register_and_index(req: RegisterProjectRequest, request: Request) -> dict[str, str]:
    ctx = request.app.state.ctx
    try:
        return jobs.launch_index_run(ctx, req.root_path)
    except ValueError as exc:
        # Typed, not text-matched (M3): the mixed-model guard is a real 409
        # Conflict ("re-index required"), but a missing/non-directory path is
        # a 400 Bad Request — mapping both to 409 sent agents down the wrong
        # recovery path for a typo'd root_path.
        status = 409 if isinstance(exc, MixedModelError) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get("/projects")
async def list_projects(request: Request) -> list[dict[str, Any]]:
    ctx = request.app.state.ctx
    return [dict(row) for row in state.list_projects(ctx.conn)]


@router.get("/projects/{project_id}/status")
async def project_status(project_id: str, request: Request) -> dict[str, Any]:
    ctx = request.app.state.ctx
    if state.get_project(ctx.conn, project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    return jobs.index_status(ctx, project_id)


@router.post(
    "/projects/{project_id}/reindex",
    status_code=202,
    dependencies=[Depends(verify_local_origin)],
)
async def reindex(project_id: str, request: Request) -> dict[str, str]:
    ctx = request.app.state.ctx
    project = state.get_project(ctx.conn, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    try:
        return jobs.launch_index_run(ctx, project["root_path"])
    except ValueError as exc:
        # Mixed-model guard → 409; a vanished root_path → 400 (M3).
        status = 409 if isinstance(exc, MixedModelError) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
async def run_status(run_id: str, request: Request) -> dict[str, Any]:
    ctx = request.app.state.ctx
    row = ctx.conn.execute(
        "SELECT * FROM index_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    body = dict(row)
    # Live progress (percent/ETA) exists only while the run's task is in
    # this process — REST-only surface; the MCP status shape stays frozen.
    if row["status"] == "running":
        body["progress"] = jobs.run_progress(ctx, run_id)
    return body


@router.post("/search")
async def search(req: SearchRequest, request: Request) -> dict[str, Any]:
    ctx = request.app.state.ctx
    if state.get_project(ctx.conn, req.project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    t0 = time.perf_counter()
    result = await search_code(
        ctx.store,
        ctx.embedder,
        req.query,
        req.project_id,
        top_k=req.top_k,
        language=req.language,
        channel=req.channel,
        reranker=ctx.reranker,
        rerank=req.rerank,
        candidates=ctx.rerank_candidates,
    )
    telemetry.record_query(
        ctx.conn,
        interface="rest",
        kind="search",
        project_id=req.project_id,
        channel=req.channel,
        reranked=result["reranked"],
        latency_ms=(time.perf_counter() - t0) * 1000,
        result_count=len(result["hits"]),
    )
    return {
        "query": req.query,
        "channel": req.channel,
        "reranked": result["reranked"],
        "hits": result["hits"],
    }


@router.post("/structural-search")
async def structural_search_route(
    req: StructuralSearchRequest, request: Request
) -> dict[str, Any]:
    ctx = request.app.state.ctx
    t0 = time.perf_counter()
    try:
        result = await structural_search(
            ctx.conn,
            req.project_id,
            req.pattern,
            req.language,
            paths=req.paths,
            max_results=req.max_results,
            settings=ctx.structural,
        )
    except StructuralSearchError as exc:
        status = 404 if exc.error_type == "unknown_project" else 400
        raise HTTPException(
            status_code=status,
            detail={"type": exc.error_type, "message": exc.message},
        ) from exc
    telemetry.record_query(
        ctx.conn,
        interface="rest",
        kind="structural",
        project_id=req.project_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        result_count=len(result.get("matches", [])),
    )
    return {"pattern": req.pattern, "language": req.language, **result}
