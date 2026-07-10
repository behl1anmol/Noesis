"""Dashboard read models + settings mutations (M8, ADR-40).

All SQL and aggregation for the human monitoring surface lives here —
``api/dashboard.py`` renders templates and forwards JSON, nothing more
(thin-adapter rule). Everything reads the local SQLite state; nothing here
may perform network I/O (CLAUDE.md rule 2).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from . import jobs, state
from .compute import available_devices
from .discovery import DiscoveryConfig, discover_files
from .languages import EXT_TO_LANGUAGE, LANGUAGE_MAP, detect_language

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def _age_seconds(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds())


def _project_summary(ctx: Any, project: sqlite3.Row) -> dict[str, Any]:
    conn = ctx.conn
    project_id = project["id"]
    counts = conn.execute(
        "SELECT COUNT(*) AS files, COALESCE(SUM(chunk_count), 0) AS chunks"
        " FROM files WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM pending_changes WHERE project_id = ?",
        (project_id,),
    ).fetchone()["n"]
    run = state.get_latest_run(conn, project_id)
    last_run: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    if run is not None:
        last_run = {
            "run_id": run["id"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "files_total": run["files_total"],
            "files_changed": run["files_changed"],
            "files_failed": run["files_failed"],
            "chunks_written": run["chunks_written"],
            "triggered_by": run["triggered_by"],
            "error": run["error"],
        }
        if run["status"] == "running":
            progress = jobs.run_progress(ctx, run["id"])
    watcher = getattr(ctx, "watcher", None)
    last_done = conn.execute(
        "SELECT finished_at FROM index_runs WHERE project_id = ? AND status = 'done'"
        " ORDER BY finished_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    return {
        "id": project_id,
        "name": PurePath(project["root_path"]).name or project["root_path"],
        "root_path": project["root_path"],
        "embedding_model": project["embedding_model"],
        "created_at": project["created_at"],
        "watch_enabled": bool(project["watch_enabled"]),
        "auto_reindex": bool(project["auto_reindex"]),
        "watching": bool(watcher is not None and watcher.watching(project_id)),
        "file_count": counts["files"],
        "chunk_count": counts["chunks"],
        "pending_count": pending,
        "last_indexed_at": None if last_done is None else last_done["finished_at"],
        "index_age_s": None
        if last_done is None
        else _age_seconds(last_done["finished_at"]),
        "last_run": last_run,
        "progress": progress,
    }


def overview(ctx: Any) -> dict[str, Any]:
    """The GET / read model: every project's health at a glance."""
    projects = [_project_summary(ctx, row) for row in state.list_projects(ctx.conn)]
    return {
        "projects": projects,
        "totals": {
            "projects": len(projects),
            "files": sum(p["file_count"] for p in projects),
            "chunks": sum(p["chunk_count"] for p in projects),
            "pending": sum(p["pending_count"] for p in projects),
            "running": sum(
                1
                for p in projects
                if p["last_run"] is not None and p["last_run"]["status"] == "running"
            ),
        },
        "device": device_info(ctx),
    }


def project_detail(ctx: Any, project_id: str) -> dict[str, Any] | None:
    """Per-project drill-down: pending files, recent runs, failed files."""
    project = state.get_project(ctx.conn, project_id)
    if project is None:
        return None
    summary = _project_summary(ctx, project)
    pending = [dict(row) for row in state.list_pending_changes(ctx.conn, project_id)]
    runs = [
        dict(row)
        for row in ctx.conn.execute(
            "SELECT * FROM index_runs WHERE project_id = ?"
            " ORDER BY started_at DESC, rowid DESC LIMIT 20",
            (project_id,),
        ).fetchall()
    ]
    # Failed files of the most recent run that recorded any — the
    # actionable set (older runs' failures were either fixed or recur here).
    failed_files: list[dict[str, Any]] = []
    for run in runs:
        errors = state.list_file_errors(ctx.conn, run["id"])
        if errors:
            failed_files = [
                {"path": e["path"], "error": e["error"], "run_id": run["id"]}
                for e in errors
            ]
            break
    return {
        **summary,
        "pending_files": pending,
        "recent_runs": runs,
        "failed_files": failed_files,
    }


def usage(ctx: Any, days: int = 30) -> dict[str, Any]:
    """The usage-page read model (ADR-40): index activity, index health,
    search usage, watcher activity — everything derived, nothing stored
    beyond what the run/query/watcher tables already hold."""
    conn = ctx.conn
    # Day-granularity cutoff on both sides: stored timestamps are ISO with a
    # 'T' separator + tz offset, SQLite's datetime() emits space-separated —
    # comparing them raw is only lexically right away from the boundary
    # (PR #10 review). substr → date-vs-date is exact.
    cutoff = f"-{days} days"
    _DAY = "substr({col}, 1, 10) >= date('now', ?)"

    runs_per_day = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT substr(started_at, 1, 10) AS day,
                   COUNT(*) AS runs,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                   SUM(CASE WHEN triggered_by = 'watcher' THEN 1 ELSE 0 END) AS watcher_runs,
                   COALESCE(SUM(files_changed), 0) AS files_changed,
                   COALESCE(SUM(chunks_written), 0) AS chunks_written
            FROM index_runs
            WHERE {_DAY.format(col="started_at")}
            GROUP BY day ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
    ]
    run_stats = dict(
        conn.execute(
            f"""
            SELECT COUNT(*) AS total_runs,
                   SUM(CASE WHEN fast_path_used = 1 THEN 1 ELSE 0 END) AS fast_path_runs,
                   AVG(CASE WHEN finished_at IS NOT NULL AND started_at IS NOT NULL
                       THEN (julianday(finished_at) - julianday(started_at)) * 86400.0
                       END) AS avg_duration_s
            FROM index_runs
            WHERE {_DAY.format(col="started_at")}
            """,
            (cutoff,),
        ).fetchone()
    )

    queries_per_day = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT substr(ts, 1, 10) AS day,
                   COUNT(*) AS queries,
                   SUM(CASE WHEN interface = 'mcp' THEN 1 ELSE 0 END) AS mcp,
                   SUM(CASE WHEN interface = 'rest' THEN 1 ELSE 0 END) AS rest,
                   SUM(CASE WHEN kind = 'structural' THEN 1 ELSE 0 END) AS structural,
                   SUM(CASE WHEN reranked = 1 THEN 1 ELSE 0 END) AS reranked
            FROM query_log
            WHERE {_DAY.format(col="ts")}
            GROUP BY day ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
    ]
    latencies = [
        row["latency_ms"]
        for row in conn.execute(
            f"SELECT latency_ms FROM query_log"
            f" WHERE {_DAY.format(col='ts')} AND latency_ms IS NOT NULL"
            f" ORDER BY latency_ms",
            (cutoff,),
        ).fetchall()
    ]

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    channel_mix = [
        dict(row)
        for row in conn.execute(
            f"SELECT COALESCE(channel, 'structural') AS channel, COUNT(*) AS queries"
            f" FROM query_log WHERE {_DAY.format(col='ts')}"
            f" GROUP BY channel ORDER BY queries DESC",
            (cutoff,),
        ).fetchall()
    ]

    watcher_per_day = [
        dict(row)
        for row in conn.execute(
            """
            SELECT day,
                   SUM(events_seen) AS events_seen,
                   SUM(events_coalesced) AS events_coalesced,
                   SUM(auto_runs) AS auto_runs
            FROM watcher_stats
            WHERE day >= date('now', ?)
            GROUP BY day ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
    ]

    # Direct grouped queries instead of a full _project_summary per project
    # (which also computes live progress) — the health table needs 6 fields
    # (PR #10 review).
    file_counts = {
        r["project_id"]: r
        for r in conn.execute(
            "SELECT project_id, COUNT(*) AS files,"
            " COALESCE(SUM(chunk_count), 0) AS chunks"
            " FROM files GROUP BY project_id"
        ).fetchall()
    }
    pending_counts = {
        r["project_id"]: r["n"]
        for r in conn.execute(
            "SELECT project_id, COUNT(*) AS n FROM pending_changes GROUP BY project_id"
        ).fetchall()
    }
    last_done = {
        r["project_id"]: r["finished_at"]
        for r in conn.execute(
            "SELECT project_id, MAX(finished_at) AS finished_at"
            " FROM index_runs WHERE status = 'done' GROUP BY project_id"
        ).fetchall()
    }
    health = []
    for row in state.list_projects(conn):
        pid = row["id"]
        counts = file_counts.get(pid)
        latest = state.get_latest_run(conn, pid)
        health.append(
            {
                "id": pid,
                "name": PurePath(row["root_path"]).name or row["root_path"],
                "file_count": counts["files"] if counts else 0,
                "chunk_count": counts["chunks"] if counts else 0,
                "pending_count": pending_counts.get(pid, 0),
                "index_age_s": _age_seconds(last_done.get(pid)),
                "files_failed": (latest["files_failed"] if latest else 0) or 0,
            }
        )

    return {
        "days": days,
        "index_activity": {
            "per_day": runs_per_day,
            "total_runs": run_stats["total_runs"] or 0,
            "fast_path_runs": run_stats["fast_path_runs"] or 0,
            "avg_duration_s": (
                None
                if run_stats["avg_duration_s"] is None
                else round(run_stats["avg_duration_s"], 2)
            ),
        },
        "search_usage": {
            "per_day": queries_per_day,
            "total_queries": sum(q["queries"] for q in queries_per_day),
            "latency_p50_ms": pct(0.50),
            "latency_p95_ms": pct(0.95),
            "channel_mix": channel_mix,
        },
        "watcher_activity": {"per_day": watcher_per_day},
        "index_health": health,
    }


def set_project_flags(
    ctx: Any,
    project_id: str,
    *,
    watch_enabled: bool | None = None,
    auto_reindex: bool | None = None,
) -> dict[str, Any] | None:
    """Persist watcher flags and apply them live. Enabling auto_reindex on
    a project with pending changes triggers an immediate catch-up scoped
    run — the toggle means "keep me fresh", not "keep me fresh starting
    with the next keystroke"."""
    project = state.get_project(ctx.conn, project_id)
    if project is None:
        return None
    state.set_project_flags(
        ctx.conn, project_id, watch_enabled=watch_enabled, auto_reindex=auto_reindex
    )
    watcher = getattr(ctx, "watcher", None)
    if watcher is not None and watch_enabled is not None:
        watcher.set_watch(project_id, watch_enabled)
    if auto_reindex:
        pending = state.list_pending_changes(ctx.conn, project_id)
        if pending:
            try:
                jobs.launch_index_run(
                    ctx,
                    project["root_path"],
                    paths=[p["path"] for p in pending],
                    triggered_by="watcher",
                )
            except ValueError as exc:
                logger.warning("catch-up reindex skipped: %s", exc)
    refreshed = state.get_project(ctx.conn, project_id)
    assert refreshed is not None
    return _project_summary(ctx, refreshed)


def reindex_pending(ctx: Any, project_id: str) -> dict[str, Any] | None:
    """Dashboard action: scoped run over the current pending set (or a
    plain incremental run when nothing is pending — the button always
    means "make it fresh now")."""
    project = state.get_project(ctx.conn, project_id)
    if project is None:
        return None
    pending = state.list_pending_changes(ctx.conn, project_id)
    paths = [p["path"] for p in pending] or None
    return jobs.launch_index_run(
        ctx, project["root_path"], paths=paths, triggered_by="manual"
    )


def _config_pin(ctx: Any) -> str | None:
    """The effective config.toml device pin shown/enforced by the dashboard.
    Either model being pinned locks the UI — set_compute_device retargets
    both models, so an embedder-only check would let a dashboard change
    silently override a reranker pin (PR #10 review)."""
    return getattr(ctx, "config_device_pin", None) or getattr(
        ctx, "config_reranker_device_pin", None
    )


def device_info(ctx: Any) -> dict[str, Any]:
    """Current compute-device state for the dashboard settings surface."""
    setting = state.get_setting(ctx.conn, "compute_device") or "auto"
    pin = _config_pin(ctx)
    embedder = ctx.embedder
    return {
        "setting": setting,
        "config_pin": pin,
        "effective_source": "config.toml" if pin else "setting",
        "resolved": getattr(embedder, "resolved_device", None),
        "available": available_devices(),
    }


def set_compute_device(ctx: Any, device: str) -> dict[str, Any]:
    """Persist the device choice and hot-retarget the loaded models
    (generation-bump reload, ADR-40). Raises ValueError on an invalid
    value or when config.toml pins either model's device (operator config
    wins)."""
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {', '.join(VALID_DEVICES)}")
    if _config_pin(ctx):
        raise ValueError(
            "device is pinned in config.toml; remove the [embedder]/[reranker]"
            " device pin to control it from the dashboard"
        )
    state.set_setting(ctx.conn, "compute_device", device)
    target = None if device == "auto" else device
    for model in (ctx.embedder, getattr(ctx, "reranker", None)):
        setter = getattr(model, "set_device", None)
        if setter is not None:
            setter(target)
    return device_info(ctx)


def delete_project(ctx: Any, project_id: str) -> bool:
    """Remove a project's index state entirely (ADR-43): cancel its running
    index task, unschedule its watch, drop its Qdrant points, delete its
    SQLite rows. Touches ONLY derived index state — never the project's
    source tree. Returns False for an unknown project.

    Order matters: task first (a run mid-flight would re-insert file rows
    and points after the wipe), then the watch (events would recreate
    pending rows), then points, then rows."""
    project = state.get_project(ctx.conn, project_id)
    if project is None:
        return False
    latest = state.get_latest_run(ctx.conn, project_id)
    if latest is not None and latest["status"] == "running":
        task = ctx.jobs.get(latest["id"])
        if task is not None:
            task.cancel()
    watcher = getattr(ctx, "watcher", None)
    if watcher is not None:
        watcher.set_watch(project_id, False)
    ctx.store.delete_project_points(project_id)
    state.delete_project(ctx.conn, project_id)
    logger.info("project %s deleted (%s)", project_id, project["root_path"])
    return True


# -- project registration (ADR-42) -------------------------------------------


def supported_languages() -> list[dict[str, Any]]:
    """Canonical languages Noesis indexes, with their extensions and whether
    structural (AST) search supports them. Static — derived from the language
    maps."""
    by_name: dict[str, list[str]] = {}
    for ext, name in EXT_TO_LANGUAGE.items():
        by_name.setdefault(name, []).append(ext)
    return [
        {
            "language": name,
            "extensions": sorted(exts),
            "structural": name in LANGUAGE_MAP,
        }
        for name, exts in sorted(by_name.items())
    ]


def _discovery_config(
    *,
    index_languages: list[str] | None,
    max_file_bytes: int | None,
    follow_symlinks: bool,
    extra_ignores: list[str] | None,
) -> DiscoveryConfig:
    kwargs: dict[str, Any] = {"follow_symlinks": follow_symlinks}
    if max_file_bytes is not None:
        kwargs["max_file_bytes"] = max_file_bytes
    if index_languages:
        kwargs["include_languages"] = frozenset(index_languages)
    if extra_ignores:
        kwargs["extra_ignore_patterns"] = tuple(extra_ignores)
    return DiscoveryConfig(**kwargs)


def browse_dir(path: str | None = None) -> dict[str, Any]:
    """List sub-directories for the register folder-picker (ADR-42).

    Directories only, never file contents — the picker chooses a project
    root. Localhost-only surface (rule 2); errors on an unreadable or
    non-directory path. Defaults to the user's home directory."""
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve()
    except OSError as exc:
        raise ValueError(f"cannot resolve path: {exc}") from exc
    if not base.is_dir():
        raise ValueError(f"not a directory: {base}")
    entries: list[dict[str, str]] = []
    try:
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            try:
                if child.is_dir():
                    entries.append({"name": child.name, "path": str(child)})
            except OSError:
                continue  # unreadable child — skip, don't fail the listing
    except OSError as exc:
        raise ValueError(f"cannot list directory: {exc}") from exc
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries}


async def preview_scan(
    ctx: Any,
    root_path: str,
    *,
    index_languages: list[str] | None = None,
    max_file_bytes: int | None = None,
    follow_symlinks: bool = False,
    extra_ignores: list[str] | None = None,
) -> dict[str, Any]:
    """Pre-flight: run discovery only (no hashing, no embed, no state write)
    and report what WOULD be indexed — file count + per-language breakdown
    (ADR-42). The same walk a real index does, in a threadpool so it never
    blocks the event loop."""
    if not os.path.isdir(root_path):
        raise ValueError(f"root_path is not an existing directory: {root_path!r}")
    cfg = _discovery_config(
        index_languages=index_languages,
        max_file_bytes=max_file_bytes,
        follow_symlinks=follow_symlinks,
        extra_ignores=extra_ignores,
    )
    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(None, discover_files, root_path, cfg)
    by_lang: dict[str, int] = {}
    for rel in files:
        lang = detect_language(rel) or "other"
        by_lang[lang] = by_lang.get(lang, 0) + 1
    breakdown = sorted(
        ({"language": k, "files": v} for k, v in by_lang.items()),
        key=lambda d: (-d["files"], d["language"]),
    )
    return {
        "root_path": str(Path(root_path).resolve()),
        "total_files": len(files),
        "by_language": breakdown,
    }


def register_project(
    ctx: Any,
    root_path: str,
    *,
    watch: bool = False,
    auto_reindex: bool = False,
    index_languages: list[str] | None = None,
    max_file_bytes: int | None = None,
    follow_symlinks: bool = False,
    extra_ignores: list[str] | None = None,
    index_now: bool = False,
) -> dict[str, Any]:
    """Register a project from the dashboard (ADR-42), persist its index
    scope + watcher flags, optionally start the first index. Raises
    ValueError on a missing directory or the mixed-model guard.

    ``index_now=False`` registers without indexing (the 'Add only' action);
    ``True`` also launches the first run ('Add + index now')."""
    if not os.path.isdir(root_path):
        raise ValueError(f"root_path is not an existing directory: {root_path!r}")
    project_id = state.register_project(ctx.conn, root_path, ctx.embedder.model_id)
    state.set_index_config(
        ctx.conn,
        project_id,
        index_languages=index_languages,
        max_file_bytes=max_file_bytes,
        follow_symlinks=follow_symlinks,
        extra_ignores=extra_ignores,
    )
    state.set_project_flags(
        ctx.conn, project_id, watch_enabled=watch, auto_reindex=auto_reindex
    )
    # Unconditional: re-registering an already-watched project with
    # watch=False must also unschedule the live OS watch, or events keep
    # flowing while the flag reads off (PR #10 review).
    watcher = getattr(ctx, "watcher", None)
    if watcher is not None:
        watcher.set_watch(project_id, watch)
    run: dict[str, Any] | None = None
    if index_now:
        run = jobs.launch_index_run(ctx, root_path, triggered_by="manual")
    project = state.get_project(ctx.conn, project_id)
    assert project is not None
    return {"project": _project_summary(ctx, project), "run": run}
