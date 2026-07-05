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
from noesis.core.state import MixedModelError

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _asset_version() -> str:
    """Cache-busting token from the static assets' latest mtime. Appended to
    the app.js/style.css URLs so a browser can never serve a stale script
    after a redeploy — the query string changes, forcing a fresh fetch."""
    latest = 0.0
    for name in ("app.js", "style.css"):
        try:
            latest = max(latest, (STATIC_DIR / name).stat().st_mtime)
        except OSError:
            continue
    return str(int(latest))


# Global so every template (base.html and all it extends) sees it without
# each handler threading it through the response context.
templates.env.globals["asset_ver"] = _asset_version()

dashboard_router = APIRouter()


class ProjectFlagsRequest(BaseModel):
    watch_enabled: bool | None = None
    auto_reindex: bool | None = None


class DeviceRequest(BaseModel):
    device: str


class IndexScope(BaseModel):
    index_languages: list[str] | None = None
    max_file_bytes: int | None = None
    follow_symlinks: bool = False
    extra_ignores: list[str] | None = None


class PreviewRequest(IndexScope):
    root_path: str


class RegisterRequest(IndexScope):
    root_path: str
    watch: bool = False
    auto_reindex: bool = False
    index_now: bool = False


# -- pages -------------------------------------------------------------------


# Pages are never cached: a stale HTML document wires stale expectations to
# fresh assets (observed as a permanently stuck register modal). The pages
# are cheap local renders — no-store costs nothing and removes the failure
# class entirely.
_NO_STORE = {"Cache-Control": "no-store"}


@dashboard_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_home(request: Request) -> Any:
    ctx = request.app.state.ctx
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "overview": core_dashboard.overview(ctx),
            # Server-rendered into the register modal: the language list is
            # static per process, so it belongs in the HTML, not behind a
            # runtime fetch (§4.12 server-rendered principle).
            "languages": core_dashboard.supported_languages(),
        },
        headers=_NO_STORE,
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
        headers=_NO_STORE,
    )


@dashboard_router.get("/usage", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_usage(request: Request, days: int = 30) -> Any:
    ctx = request.app.state.ctx
    days = max(1, min(days, 365))
    return templates.TemplateResponse(
        request,
        "usage.html",
        {"usage": core_dashboard.usage(ctx, days=days)},
        headers=_NO_STORE,
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


@dashboard_router.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: str, request: Request) -> dict[str, Any]:
    if not core_dashboard.delete_project(request.app.state.ctx, project_id):
        raise HTTPException(status_code=404, detail="unknown project_id")
    return {"project_id": project_id, "deleted": True}


# -- project registration (ADR-42) -------------------------------------------


@dashboard_router.get("/api/languages")
async def api_languages() -> dict[str, Any]:
    return {"languages": core_dashboard.supported_languages()}


@dashboard_router.get("/api/browse")
async def api_browse(request: Request, path: str | None = None) -> dict[str, Any]:
    try:
        return core_dashboard.browse_dir(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@dashboard_router.post("/api/register/preview")
async def api_register_preview(req: PreviewRequest, request: Request) -> dict[str, Any]:
    try:
        return await core_dashboard.preview_scan(
            request.app.state.ctx,
            req.root_path,
            index_languages=req.index_languages,
            max_file_bytes=req.max_file_bytes,
            follow_symlinks=req.follow_symlinks,
            extra_ignores=req.extra_ignores,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@dashboard_router.post("/api/register", status_code=201)
async def api_register(req: RegisterRequest, request: Request) -> dict[str, Any]:
    try:
        return core_dashboard.register_project(
            request.app.state.ctx,
            req.root_path,
            watch=req.watch,
            auto_reindex=req.auto_reindex,
            index_languages=req.index_languages,
            max_file_bytes=req.max_file_bytes,
            follow_symlinks=req.follow_symlinks,
            extra_ignores=req.extra_ignores,
            index_now=req.index_now,
        )
    except ValueError as exc:
        # Typed, not text-matched (PR #10 review): mixed-model guard → 409,
        # any other validation failure (missing dir, bad values) → 400.
        status = 409 if isinstance(exc, MixedModelError) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
