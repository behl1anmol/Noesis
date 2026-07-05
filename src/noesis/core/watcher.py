"""File watcher — near-real-time index freshness (Overview §4.9, ADR-40).

One ``watchdog`` Observer serves every watched project. The observer's
event thread does string checks only — no hashing, no file reads, no DB
access — so watching never contends with the editor writing the file
(stakeholder requirement: lightweight). Events are marshalled onto the
asyncio loop, debounced and coalesced per (project, path), then written to
``pending_changes`` — the dashboard's "files awaiting reindex" view. For a
project with ``auto_reindex`` enabled, a quiet period after the last event
triggers a *scoped* run over exactly the pending paths (never a full
re-embed; the run's own discovery + hash comparison stay authoritative).

Filtering here is deliberately a cheap approximation of discovery's rules
(EXCLUDED_DIRS, secret + lockfile skip-lists, editor temp noise, the root
.gitignore). Anything that slips through is at worst a cosmetic pending row:
the scoped run re-applies discovery in full, and its clear step removes the
row afterward. Nested .gitignore semantics are intentionally not replicated
in the event path — correctness lives in the run, not the filter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pathspec import GitIgnoreSpec

from . import jobs, state
from .discovery import EXCLUDED_DIRS, _GENERATED_SPEC, is_secret_path

if TYPE_CHECKING:
    from watchdog.events import FileSystemEvent

logger = logging.getLogger(__name__)

# Editor/tooling churn that is never project content: vim swap/backup,
# emacs autosave/lockfiles, generic tempfiles, partial downloads.
_NOISE_SUFFIXES = (".swp", ".swo", ".swx", "~", ".tmp", ".part", ".crswap")
_NOISE_PREFIXES = (".#", "#")
_VIM_PROBE = "4913"  # vim's write-probe file


def _is_noise(name: str) -> bool:
    return (
        name.endswith(_NOISE_SUFFIXES)
        or name.startswith(_NOISE_PREFIXES)
        or name == _VIM_PROBE
    )


class _ProjectWatch:
    """Per-project state: root, its watchdog handle, root .gitignore spec."""

    def __init__(self, project_id: str, root: Path) -> None:
        self.project_id = project_id
        self.root = root
        self.handle: Any = None
        self.ignore_spec: GitIgnoreSpec | None = None
        self.reload_ignore()

    def reload_ignore(self) -> None:
        gitignore = self.root / ".gitignore"
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            self.ignore_spec = GitIgnoreSpec.from_lines(lines)
        except OSError:
            self.ignore_spec = None


class WatcherManager:
    """Owns the Observer, the debounce consumer, and per-project watches.

    Constructed once in the app lifespan (``ctx.watcher``). The Observer
    thread starts lazily on the first scheduled watch, so test apps with no
    watched projects never spawn it. ``debounce_s`` is the flush cadence for
    pending rows; ``quiet_s`` is the no-events window required before an
    auto-reindex fires (a save burst becomes one scoped run, not N).
    """

    def __init__(
        self, ctx: Any, *, debounce_s: float = 0.5, quiet_s: float = 2.0
    ) -> None:
        self._ctx = ctx
        self.debounce_s = debounce_s
        self.quiet_s = quiet_s
        self._observer: Any = None
        self._watches: dict[str, _ProjectWatch] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
        self._consumer: asyncio.Task | None = None
        # Accumulated between flushes: project_id -> {rel_path: event_type}.
        self._accum: dict[str, dict[str, str]] = {}
        self._seen_since_flush: dict[str, int] = {}
        self._last_event: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Register watches for every watch_enabled project and start the
        consumer. Called from the app lifespan (loop is running)."""
        self._loop = asyncio.get_running_loop()
        self._consumer = self._loop.create_task(self._consume())
        for row in state.watched_projects(self._ctx.conn):
            self._schedule(row["id"], row["root_path"])
            # Catch-up: pending rows left over from a previous process (or
            # events missed while down) — if auto_reindex is on, run now.
            if row["auto_reindex"]:
                pending = state.list_pending_changes(self._ctx.conn, row["id"])
                if pending:
                    self._launch_scoped(row["id"], [p["path"] for p in pending])

    def stop(self) -> None:
        # Stop the observer first (no new events), then drain what already
        # reached the queue and flush the accumulator — otherwise up to one
        # debounce window of events is dropped across a clean restart, and
        # start()'s catch-up only ever sees flushed rows (PR #10 review).
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._consumer is not None:
            self._consumer.cancel()
            self._consumer = None
        while True:
            try:
                self._accumulate(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        try:
            self._flush()
        except Exception:  # noqa: BLE001 — shutdown must complete
            logger.exception("final watcher flush failed")
        self._watches.clear()

    def set_watch(self, project_id: str, enabled: bool) -> None:
        """Live toggle from the dashboard. The flag is already persisted by
        the caller (core.dashboard); this only (un)schedules the OS watch."""
        if enabled:
            project = state.get_project(self._ctx.conn, project_id)
            if project is not None and project_id not in self._watches:
                self._schedule(project_id, project["root_path"])
        else:
            watch = self._watches.pop(project_id, None)
            if watch is not None and self._observer is not None and watch.handle:
                self._observer.unschedule(watch.handle)

    def watching(self, project_id: str) -> bool:
        return project_id in self._watches

    # -- observer side (event thread!) --------------------------------------

    def _schedule(self, project_id: str, root_path: str) -> None:
        root = Path(root_path)
        if not root.is_dir():
            logger.warning("watch skipped, root missing: %s", root_path)
            return
        if self._observer is None:
            from watchdog.observers import Observer

            self._observer = Observer()
            self._observer.daemon = True
            self._observer.start()
        watch = _ProjectWatch(project_id, root)
        handler = _Handler(self, watch)
        watch.handle = self._observer.schedule(handler, str(root), recursive=True)
        self._watches[project_id] = watch
        logger.info("watching project %s at %s", project_id, root)

    def _enqueue_threadsafe(self, project_id: str, rel: str, event_type: str) -> None:
        """Called from the watchdog event thread — hop onto the loop."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (project_id, rel, event_type)
            )

    def _reload_ignore_threadsafe(self, watch: _ProjectWatch) -> None:
        """Reload a project's root .gitignore spec on the event loop — the
        observer event thread does string checks only, no file reads
        (PR #10 review keeps that invariant honest)."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(watch.reload_ignore)

    # -- consumer side (event loop) ------------------------------------------

    def _accumulate(self, item: tuple[str, str, str]) -> None:
        project_id, rel, event_type = item
        bucket = self._accum.setdefault(project_id, {})
        prior = bucket.get(rel)
        # 'created' then 'modified' is still a creation; anything then
        # 'deleted' is a deletion. Last event wins except created+modified.
        if prior == "created" and event_type == "modified":
            event_type = "created"
        bucket[rel] = event_type
        self._seen_since_flush[project_id] = (
            self._seen_since_flush.get(project_id, 0) + 1
        )
        self._last_event[project_id] = time.monotonic()

    async def _consume(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self.debounce_s
                )
                self._accumulate(item)
                while True:
                    try:
                        self._accumulate(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            except (asyncio.TimeoutError, TimeoutError):
                pass
            try:
                self._flush()
                self._maybe_auto_reindex()
            except Exception:  # noqa: BLE001 — the watcher must outlive a bad flush
                logger.exception("watcher flush failed")

    def _flush(self) -> None:
        for project_id, bucket in self._accum.items():
            if not bucket:
                continue
            state.upsert_pending_changes(
                self._ctx.conn, project_id, list(bucket.items())
            )
            seen = self._seen_since_flush.get(project_id, 0)
            state.bump_watcher_stats(
                self._ctx.conn,
                project_id,
                events_seen=seen,
                events_coalesced=max(0, seen - len(bucket)),
            )
        self._accum.clear()
        self._seen_since_flush.clear()

    def _maybe_auto_reindex(self) -> None:
        now = time.monotonic()
        for project_id, last in list(self._last_event.items()):
            if now - last < self.quiet_s:
                continue
            del self._last_event[project_id]
            project = state.get_project(self._ctx.conn, project_id)
            if project is None or not project["auto_reindex"]:
                continue
            pending = state.list_pending_changes(self._ctx.conn, project_id)
            if pending:
                self._launch_scoped(project_id, [p["path"] for p in pending])

    def _launch_scoped(self, project_id: str, paths: list[str]) -> None:
        project = state.get_project(self._ctx.conn, project_id)
        if project is None:
            return
        try:
            jobs.launch_index_run(
                self._ctx,
                project["root_path"],
                paths=paths,
                triggered_by="watcher",
            )
            state.bump_watcher_stats(self._ctx.conn, project_id, auto_runs=1)
            logger.info(
                "auto-reindex launched for %s (%d pending)", project_id, len(paths)
            )
        except ValueError as exc:
            logger.warning("auto-reindex skipped for %s: %s", project_id, exc)


class _Handler:
    """watchdog handler — runs on the observer thread. Cheap checks only."""

    def __init__(self, manager: WatcherManager, watch: _ProjectWatch) -> None:
        self._manager = manager
        self._watch = watch

    def dispatch(self, event: "FileSystemEvent") -> None:
        etype = event.event_type
        if etype == "moved":
            self._emit(event.src_path, "deleted", is_dir=event.is_directory)
            self._emit(getattr(event, "dest_path", ""), "created", is_dir=event.is_directory)
            return
        if etype not in ("created", "modified", "deleted"):
            return  # opened/closed/closed_no_write etc.
        if event.is_directory and etype != "deleted":
            return  # per-file events carry the signal; dir deletion is one row
        self._emit(event.src_path, etype, is_dir=event.is_directory)

    def _emit(self, raw_path: str | bytes, event_type: str, *, is_dir: bool) -> None:
        if not raw_path:
            return
        path = Path(raw_path if isinstance(raw_path, str) else raw_path.decode())
        try:
            rel = PurePosixPath(path.relative_to(self._watch.root)).as_posix()
        except ValueError:
            return  # outside the watched root
        if rel in (".", ""):
            return
        parts = rel.split("/")
        if any(part in EXCLUDED_DIRS for part in parts):
            return
        name = parts[-1]
        if _is_noise(name):
            return
        if name == ".gitignore":
            self._manager._reload_ignore_threadsafe(self._watch)
            return
        if not is_dir:
            if is_secret_path(rel):
                return
            if _GENERATED_SPEC.match_file(rel):
                return
            spec = self._watch.ignore_spec
            if spec is not None and spec.match_file(rel):
                return
        self._manager._enqueue_threadsafe(self._watch.project_id, rel, event_type)
