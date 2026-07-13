"""File watcher — near-real-time index freshness (Overview §4.9, ADR-40).

At most two ``watchdog`` observers serve every watched project: the native
Observer (inotify on Linux) for roots on ordinary filesystems, and a
PollingObserver for roots on filesystems where inotify is silently blind —
network/host-passthrough mounts like WSL2's 9p drvfs (``/mnt/c``), CIFS,
NFS, sshfs. On those, the kernel never delivers inotify events, so a watch
schedules cleanly yet the handler never fires; polling is the only signal.
Each observer's event thread does string checks only — no hashing, no file
reads, no DB access — so watching never contends with the editor writing
the file (stakeholder requirement: lightweight). Events are marshalled onto the
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
import os
import sqlite3
import sys
import time
from collections.abc import Iterator
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


# Filesystems where Linux inotify silently delivers no events (network or
# host-passthrough mounts — the kernel-side write happens on the other end).
# Roots on these get the PollingObserver instead of the native one. Broad on
# purpose: a false positive merely polls a root inotify could have watched,
# a false negative is the silent no-events bug this exists to fix.
_POLLING_FSTYPES = frozenset(
    {"9p", "cifs", "smb3", "nfs", "nfs4", "vboxsf", "prl_fs", "grpcfuse"}
)
_POLLING_FSTYPE_PREFIXES = ("fuse",)  # fuse.sshfs, fuse.gvfsd-fuse, fuseblk, ...


def _unescape_mount(field: str) -> str:
    """Undo /proc/mounts octal escapes (\\040 space, \\011 tab, \\012 newline,
    \\134 backslash) in a mountpoint field."""
    if "\\" not in field:
        return field
    out: list[str] = []
    i = 0
    while i < len(field):
        ch = field[i]
        if (
            ch == "\\"
            and i + 3 < len(field)
            and all(c in "01234567" for c in field[i + 1 : i + 4])
        ):
            try:
                out.append(chr(int(field[i + 1 : i + 4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(ch)
        i += 1
    return "".join(out)


def _fstype_for(path: str, mounts_text: str) -> str | None:
    """Filesystem type of the mount holding *path* (absolute, resolved):
    longest-mountpoint-prefix match over /proc/mounts content. Pure function
    so tests feed it synthetic mount tables. None when nothing matches."""
    best_len = -1
    best_fstype: str | None = None
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = _unescape_mount(fields[1])
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > best_len:
                best_len = len(mountpoint)
                best_fstype = fields[2]
    return best_fstype


def _inotify_blind_fstype(root: Path) -> str | None:
    """The fstype of *root*'s mount when it cannot deliver inotify events,
    else None (watch natively). Only Linux has both inotify and /proc/mounts;
    anything unreadable degrades to None — native was the status quo."""
    if sys.platform != "linux":
        return None
    try:
        resolved = str(root.resolve())
        mounts_text = Path("/proc/mounts").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    fstype = _fstype_for(resolved, mounts_text)
    if fstype is None:
        return None
    if fstype in _POLLING_FSTYPES or fstype.startswith(_POLLING_FSTYPE_PREFIXES):
        return fstype
    return None


def _pruned_scandir(path: str | None) -> Iterator[os.DirEntry[str]]:
    """``listdir`` for the PollingObserver's DirectorySnapshot that never
    descends into EXCLUDED_DIRS. DirectorySnapshot stat-walks the whole tree
    every poll interval; on 9p each stat is a slow network round-trip, so a
    full walk of a project's ``.venv`` (tens of thousands of files) takes
    minutes and pegs the polling thread, starving the event loop until the
    dashboard hangs. Skipping excluded dirs at the source keeps the walk to
    real project files (measured: a 73k-path / ~350s Noesis walk drops to
    151 paths / ~0.6s). The post-hoc ``_Handler`` filter still drops whatever
    slips through (secrets, generated files, .gitignore matches)."""
    with os.scandir(path) as it:  # DirectorySnapshot always passes a real dir
        for entry in it:
            # DirectorySnapshot.walk recurses on ``stat(path)`` (os.stat, which
            # follows symlinks), so a *symlinked* .venv/node_modules is walked
            # into. Prune on symlink-or-dir, not is_dir(follow_symlinks=False):
            # the latter is False for a symlink and would let the multi-minute
            # 9p walk this pruning exists to prevent slip straight back in.
            if entry.name in EXCLUDED_DIRS and (
                entry.is_symlink() or entry.is_dir(follow_symlinks=False)
            ):
                continue
            yield entry


class _ProjectWatch:
    """Per-project state: root, its watchdog handle, root .gitignore spec."""

    def __init__(self, project_id: str, root: Path) -> None:
        self.project_id = project_id
        self.root = root
        self.handle: Any = None
        # The observer that owns ``handle`` — unschedule must target it, and
        # once both observers exist the wrong one raises KeyError.
        self.observer: Any = None
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
    """Owns the observers, the debounce consumer, and per-project watches.

    Constructed once in the app lifespan (``ctx.watcher``). Each observer
    thread (native inotify, and the polling fallback for inotify-blind
    mounts) starts lazily on the first watch that needs it, so test apps
    with no watched projects never spawn either. ``debounce_s`` is the flush
    cadence for pending rows; ``quiet_s`` is the no-events window required
    before an auto-reindex fires (a save burst becomes one scoped run, not
    N); ``poll_interval_s`` is the PollingObserver snapshot cadence (config
    ``watcher.poll_interval_s``) — only polled roots pay it.
    """

    def __init__(
        self,
        ctx: Any,
        *,
        debounce_s: float = 0.5,
        quiet_s: float = 2.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._ctx = ctx
        self.debounce_s = debounce_s
        self.quiet_s = quiet_s
        self.poll_interval_s = poll_interval_s
        self._observer: Any = None
        self._polling_observer: Any = None
        # root path -> inotify-blind fstype (or None). Caches the /proc/mounts
        # read + one-time warning so toggling watch on/off on the same root
        # neither re-reads /proc/mounts nor re-logs.
        self._fstype_cache: dict[str, str | None] = {}
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

    async def stop(self) -> None:
        # Stop the observers first (no new events), then drain what already
        # reached the queue and flush the accumulator — otherwise up to one
        # debounce window of events is dropped across a clean restart, and
        # start()'s catch-up only ever sees flushed rows (PR #10 review).
        # Async: the observer join is a blocking thread join (up to 5 s on a
        # slow dispatch) that must not stall the event loop mid-shutdown,
        # and the cancelled consumer must be awaited or the loop can tear
        # down while it is still pending ("Task was destroyed…"). Signal
        # both before joining either so the 5 s budgets overlap.
        observers = [
            o for o in (self._observer, self._polling_observer) if o is not None
        ]
        self._observer = None
        self._polling_observer = None
        for observer in observers:
            observer.stop()
        for observer in observers:
            await asyncio.to_thread(observer.join, 5)
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
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
            if watch is not None and watch.observer is not None and watch.handle:
                watch.observer.unschedule(watch.handle)
            # Drop any buffered state for this project (H2). delete_project
            # calls this right before removing the projects row; a leftover
            # bucket would then fail pending_changes' FK on the next flush and,
            # because the accumulator is only cleared at the end of _flush,
            # re-raise every tick — wedging flush and auto-reindex for ALL
            # projects. Safe for a plain unwatch too: unbuffered events just
            # stop being recorded, which is what disabling the watch means.
            self._accum.pop(project_id, None)
            self._seen_since_flush.pop(project_id, None)
            self._last_event.pop(project_id, None)

    def watching(self, project_id: str) -> bool:
        return project_id in self._watches

    def mode(self, project_id: str) -> str | None:
        """``"polling"``/``"native"`` for a watched project, None otherwise.
        The dashboard shows a "polling" tag so degraded-signal roots (9p,
        network mounts) are visible, not just logged."""
        watch = self._watches.get(project_id)
        if watch is None:
            return None
        if watch.observer is self._polling_observer and watch.observer is not None:
            return "polling"
        return "native"

    # -- observer side (event thread!) --------------------------------------

    def _schedule(self, project_id: str, root_path: str) -> None:
        root = Path(root_path)
        if not root.is_dir():
            logger.warning("watch skipped, root missing: %s", root_path)
            return
        observer = self._observer_for(root)
        watch = _ProjectWatch(project_id, root)
        handler = _Handler(self, watch)
        watch.handle = observer.schedule(handler, str(root), recursive=True)
        watch.observer = observer
        self._watches[project_id] = watch
        logger.info("watching project %s at %s", project_id, root)

    def _observer_for(self, root: Path) -> Any:
        """Pick (and lazily start) the observer for this root. Roots on
        inotify-blind mounts get a PollingObserver whose snapshot walk prunes
        EXCLUDED_DIRS at the source (``_pruned_scandir``) — without that, the
        per-interval stat-walk of a project's ``.venv`` over 9p takes minutes
        and hangs the dashboard. Per-project .gitignore/secret filtering stays
        post-hoc in ``_Handler`` (it is not a walk-cost problem)."""
        key = str(root)
        if key in self._fstype_cache:
            fstype = self._fstype_cache[key]
        else:
            fstype = _inotify_blind_fstype(root)
            self._fstype_cache[key] = fstype
            if fstype is not None:
                logger.warning(
                    "root %s is on a %s filesystem where inotify delivers no "
                    "events; falling back to polling every %.1fs",
                    root,
                    fstype,
                    self.poll_interval_s,
                )
        if fstype is not None:
            if self._polling_observer is None:
                from watchdog.observers.polling import PollingObserverVFS

                self._polling_observer = PollingObserverVFS(
                    stat=os.stat,
                    listdir=_pruned_scandir,
                    polling_interval=self.poll_interval_s,
                )
                self._polling_observer.daemon = True
                self._polling_observer.start()
            return self._polling_observer
        if self._observer is None:
            from watchdog.observers import Observer

            self._observer = Observer()
            self._observer.daemon = True
            self._observer.start()
        return self._observer

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
            # Per-project fault isolation (H2): a project deleted between
            # buffering and flush trips pending_changes' FK to projects. Catch
            # it per bucket and drop that project's events so one dead project
            # can never wedge the whole flush loop (and _maybe_auto_reindex
            # after it) on every tick.
            try:
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
            except sqlite3.IntegrityError:
                logger.warning(
                    "dropping buffered events for removed project %s", project_id
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
            result = jobs.launch_index_run(
                self._ctx,
                project["root_path"],
                paths=paths,
                triggered_by="watcher",
            )
        except ValueError as exc:
            logger.warning("auto-reindex skipped for %s: %s", project_id, exc)
            return
        if result.get("status") == "already_running":
            # The launch was a no-op — a run is already in flight (H3). If we
            # let this pass as "fired", the pending files would sit until the
            # user touches another file. Re-arm the quiet-period trigger so
            # _maybe_auto_reindex retries after the current run finishes.
            # Do NOT bump auto_runs: no new run started.
            self._last_event[project_id] = time.monotonic()
            logger.info(
                "auto-reindex deferred for %s: a run is already in flight", project_id
            )
            return
        state.bump_watcher_stats(self._ctx.conn, project_id, auto_runs=1)
        logger.info("auto-reindex launched for %s (%d pending)", project_id, len(paths))


class _Handler:
    """watchdog handler — runs on the observer thread. Cheap checks only."""

    def __init__(self, manager: WatcherManager, watch: _ProjectWatch) -> None:
        self._manager = manager
        self._watch = watch

    def dispatch(self, event: "FileSystemEvent") -> None:
        etype = event.event_type
        if etype == "moved":
            self._emit(event.src_path, "deleted", is_dir=event.is_directory)
            self._emit(
                getattr(event, "dest_path", ""), "created", is_dir=event.is_directory
            )
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
