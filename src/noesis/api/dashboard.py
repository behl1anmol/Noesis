"""Dashboard adapter — Jinja2 pages + JSON polling + settings actions (M8).

Thin over ``core/dashboard.py`` per the adapter rule: every handler is one
core call plus template/JSON shaping. Pages are server-rendered (Overview
§4.12 — no SPA tooling); the polling endpoints exist so the pages can
refresh run progress and pending counts without full reloads. All assets
are served from ``/static`` — a strict no-CDN surface (local-only, ADR-25
spirit: the dashboard must render with the network cable pulled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from noesis.core import dashboard as core_dashboard

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

dashboard_router = APIRouter()


class ProjectFlagsRequest(BaseModel):
    watch_enabled: bool | None = None
    auto_reindex: bool | None = None


class DeviceRequest(BaseModel):
    device: str


# -- pages -------------------------------------------------------------------


@dashboard_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_home(request: Request) -> Any:
    ctx = request.app.state.ctx
    return templates.TemplateResponse(
        request, "index.html", {"overview": core_dashboard.overview(ctx)}
    )


@dashboard_router.get(
    "/projects/{project_id}/view", response_class=HTMLResponse, include_in_schema=False
)
async def dashboard_project(project_id: str, request: Request) -> Any:
    ctx = request.app.state.ctx
    detail = core_dashboard.project_detail(ctx, project_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    return templates.TemplateResponse(
        request,
        "project.html",
        {"project": detail, "device": core_dashboard.device_info(ctx)},
    )


@dashboard_router.get("/usage", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_usage(request: Request, days: int = 30) -> Any:
    ctx = request.app.state.ctx
    days = max(1, min(days, 365))
    return templates.TemplateResponse(
        request, "usage.html", {"usage": core_dashboard.usage(ctx, days=days)}
    )


# -- polling JSON -------------------------------------------------------------


@dashboard_router.get("/api/state")
async def api_state(request: Request) -> dict[str, Any]:
    return core_dashboard.overview(request.app.state.ctx)


@dashboard_router.get("/api/projects/{project_id}/state")
async def api_project_state(project_id: str, request: Request) -> dict[str, Any]:
    detail = core_dashboard.project_detail(request.app.state.ctx, project_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    return detail


@dashboard_router.get("/api/usage")
async def api_usage(request: Request, days: int = 30) -> dict[str, Any]:
    return core_dashboard.usage(request.app.state.ctx, days=max(1, min(days, 365)))


# -- actions ------------------------------------------------------------------


@dashboard_router.post("/api/projects/{project_id}/flags")
async def api_set_flags(
    project_id: str, req: ProjectFlagsRequest, request: Request
) -> dict[str, Any]:
    summary = core_dashboard.set_project_flags(
        request.app.state.ctx,
        project_id,
        watch_enabled=req.watch_enabled,
        auto_reindex=req.auto_reindex,
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    return summary


@dashboard_router.post("/api/projects/{project_id}/reindex-pending", status_code=202)
async def api_reindex_pending(project_id: str, request: Request) -> dict[str, Any]:
    try:
        result = core_dashboard.reindex_pending(request.app.state.ctx, project_id)
    except ValueError as exc:  # mixed-model / missing root guard
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="unknown project_id")
    return result


@dashboard_router.post("/api/settings/device")
async def api_set_device(req: DeviceRequest, request: Request) -> dict[str, Any]:
    try:
        return core_dashboard.set_compute_device(request.app.state.ctx, req.device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
