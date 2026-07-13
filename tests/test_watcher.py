"""Watcher tests (M8): event filtering, coalescing, pending rows, and the
auto-reindex path — real watchdog observer on a tmp dir, fake models.

Timing: the manager runs with shrunk debounce/quiet windows and every
assertion polls with a generous deadline, so a slow CI box makes the test
slower, not flaky.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.app import AppContext
from noesis.core import indexer, state
from noesis.core import watcher as watcher_mod
from noesis.core.embedder import FakeEmbedder
from noesis.core.vectorstore import VectorStore
from noesis.core.watcher import (
    WatcherManager,
    _fstype_for,
    _Handler,
    _inotify_blind_fstype,
    _ProjectWatch,
    _pruned_scandir,
)


def make_ctx(tmp_path) -> AppContext:
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


async def _poll(predicate, timeout: float = 10.0, interval: float = 0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(interval)


# -- handler filtering (no observer needed) ------------------------------------


class _CaptureManager:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def _enqueue_threadsafe(self, project_id, rel, event_type):
        self.items.append((project_id, rel, event_type))

    def _reload_ignore_threadsafe(self, watch):
        # unit tests run without a loop — reload synchronously
        watch.reload_ignore()


class _Event:
    def __init__(self, event_type, src_path, is_directory=False, dest_path=None):
        self.event_type = event_type
        self.src_path = src_path
        self.is_directory = is_directory
        if dest_path is not None:
            self.dest_path = dest_path


@pytest.fixture()
def handler_env(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".gitignore").write_text("ignored_dir/\n*.log\n")
    manager = _CaptureManager()
    handler = _Handler(manager, _ProjectWatch("p1", root))
    return root, manager, handler


def test_handler_passes_source_files(handler_env):
    root, manager, handler = handler_env
    handler.dispatch(_Event("modified", str(root / "src" / "main.py")))
    assert manager.items == [("p1", "src/main.py", "modified")]


@pytest.mark.parametrize(
    "rel",
    [
        "node_modules/x/index.js",  # excluded dir
        ".git/HEAD",  # excluded dir
        ".env",  # secret skip-list
        "uv.lock",  # generated skip-list
        "a.py.swp",  # editor noise
        "#buffer#",  # editor noise
        "4913",  # vim probe
        "app.log",  # root .gitignore
        "ignored_dir/f.py",  # root .gitignore
    ],
)
def test_handler_filters(handler_env, rel):
    root, manager, handler = handler_env
    handler.dispatch(_Event("modified", str(root / Path(rel))))
    assert manager.items == []


def test_handler_move_emits_delete_and_create(handler_env):
    root, manager, handler = handler_env
    handler.dispatch(
        _Event("moved", str(root / "old.py"), dest_path=str(root / "new.py"))
    )
    assert ("p1", "old.py", "deleted") in manager.items
    assert ("p1", "new.py", "created") in manager.items


def test_handler_outside_root_dropped(handler_env, tmp_path):
    root, manager, handler = handler_env
    handler.dispatch(_Event("modified", str(tmp_path / "elsewhere.py")))
    assert manager.items == []


def test_handler_gitignore_reload(handler_env):
    root, manager, handler = handler_env
    handler.dispatch(_Event("modified", str(root / "notes.txt")))
    assert len(manager.items) == 1
    (root / ".gitignore").write_text("notes.txt\n")
    handler.dispatch(_Event("modified", str(root / ".gitignore")))  # triggers reload
    handler.dispatch(_Event("modified", str(root / "notes.txt")))
    assert len(manager.items) == 1  # now ignored


# -- end-to-end: events → pending rows → auto scoped run -----------------------


def test_watch_records_pending_and_auto_reindexes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 1\n")

    async def scenario():
        ctx = make_ctx(tmp_path)
        await indexer.index_project(ctx.conn, ctx.store, ctx.embedder, str(root))
        pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
        state.set_project_flags(ctx.conn, pid, watch_enabled=True, auto_reindex=False)

        manager = WatcherManager(ctx, debounce_s=0.05, quiet_s=0.2)
        ctx.watcher = manager
        manager.start()
        try:
            assert manager.watching(pid)
            # Phase 1: auto off → change lands as a pending row, no run.
            runs_before = ctx.conn.execute(
                "SELECT COUNT(*) FROM index_runs"
            ).fetchone()[0]
            (root / "a.py").write_text("x = 2\n")
            pending = await _poll(lambda: state.list_pending_changes(ctx.conn, pid))
            assert [p["path"] for p in pending] == ["a.py"]
            await asyncio.sleep(0.5)  # well past quiet_s
            assert (
                ctx.conn.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0]
                == runs_before
            )

            # Phase 2: enabling auto_reindex triggers the catch-up scoped run.
            from noesis.core import dashboard as core_dashboard

            core_dashboard.set_project_flags(ctx, pid, auto_reindex=True)
            await _poll(lambda: not state.list_pending_changes(ctx.conn, pid))
            run = state.get_latest_run(ctx.conn, pid)
            await _poll(
                lambda: state.get_latest_run(ctx.conn, pid)["status"] != "running"
            )
            run = state.get_latest_run(ctx.conn, pid)
            assert run["triggered_by"] == "watcher"
            assert run["files_changed"] == 1  # only a.py was hashed+indexed

            # Phase 3: live event with auto on → scoped run within seconds.
            (root / "b.py").write_text("y = 2\n")
            await _poll(
                lambda: (
                    (r := state.get_latest_run(ctx.conn, pid)) is not None
                    and r["triggered_by"] == "watcher"
                    and r["status"] == "done"
                    and r["id"] != run["id"]
                )
            )
            latest = state.get_latest_run(ctx.conn, pid)
            assert latest["files_changed"] == 1
            assert not state.list_pending_changes(ctx.conn, pid)
            # Watcher runs never advance the git anchor (none here) and never
            # claim the git fast path.
            assert latest["fast_path_used"] == 0
        finally:
            await manager.stop()

    asyncio.run(scenario())


# -- filesystem-aware observer selection (9p/WSL2 fix) --------------------------

_MOUNTS = (
    "/dev/sda1 / ext4 rw,relatime 0 0\n"
    "tmpfs /tmp tmpfs rw 0 0\n"
    "C:\\134 /mnt/c 9p rw,noatime,aname=drvfs;path=C:\\;uid=1000 0 0\n"
    "D:\\134 /mnt/d 9p rw,noatime,aname=drvfs;path=D:\\;uid=1000 0 0\n"
    "//srv/share /media/share cifs rw 0 0\n"
    "host:/export /media/nfs nfs4 rw 0 0\n"
    "user@box:/ /media/ssh fuse.sshfs rw 0 0\n"
    "/dev/sdb1 /mnt/my\\040share btrfs rw 0 0\n"
    "malformed-line\n"
)


@pytest.mark.parametrize(
    ("path", "fstype"),
    [
        ("/mnt/d/Projects/X", "9p"),
        ("/mnt/d", "9p"),
        ("/mnt/dd/x", "ext4"),  # prefix boundary: /mnt/dd is NOT under /mnt/d
        ("/home/user/x", "ext4"),
        ("/media/share/code", "cifs"),
        ("/media/nfs/code", "nfs4"),
        ("/media/ssh/code", "fuse.sshfs"),
        ("/mnt/my share/f", "btrfs"),  # octal-escaped mountpoint
    ],
)
def test_fstype_for_longest_prefix(path, fstype):
    assert _fstype_for(path, _MOUNTS) == fstype


def test_fstype_for_no_match():
    assert _fstype_for("/x", "malformed\nshort line\n") is None


def test_inotify_blind_fstype_platform_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher_mod.sys, "platform", "darwin")
    assert _inotify_blind_fstype(tmp_path) is None


def test_inotify_blind_fstype_degrades_on_read_error(tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(watcher_mod.Path, "read_text", boom)
    assert _inotify_blind_fstype(tmp_path) is None


def test_pruned_scandir_skips_excluded_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".venv" / "lib.py").write_text("y = 1\n")
    names = {e.name for e in _pruned_scandir(str(tmp_path))}
    assert "src" in names and "a.py" in names
    assert ".venv" not in names  # excluded dir never yielded → never descended
    assert "node_modules" not in names


def test_pruned_scandir_skips_symlinked_excluded_dirs(tmp_path):
    # DirectorySnapshot recurses via os.stat (follows symlinks), so a *symlinked*
    # excluded dir would be walked into — the multi-minute 9p hang this prune
    # exists to prevent. It must be pruned even though it's a symlink, not a dir.
    real = tmp_path / "real_venv"
    real.mkdir()
    (real / "lib.py").write_text("y = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / ".venv").symlink_to(real, target_is_directory=True)
    names = {e.name for e in _pruned_scandir(str(tmp_path))}
    assert "src" in names
    assert ".venv" not in names  # symlinked excluded dir pruned → never descended


def test_schedule_picks_polling_observer(tmp_path, monkeypatch):
    from watchdog.observers.polling import PollingObserverVFS

    monkeypatch.setattr(watcher_mod, "_inotify_blind_fstype", lambda root: "9p")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")

    async def scenario():
        ctx = make_ctx(tmp_path)
        pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
        state.set_project_flags(ctx.conn, pid, watch_enabled=True)
        manager = WatcherManager(
            ctx, debounce_s=0.05, quiet_s=0.2, poll_interval_s=0.1
        )
        manager.start()
        try:
            assert isinstance(manager._polling_observer, PollingObserverVFS)
            assert manager._observer is None  # native thread never spawned
            assert manager._watches[pid].observer is manager._polling_observer
            assert manager.mode(pid) == "polling"
            # Events flow end-to-end through the polling path.
            (root / "a.py").write_text("x = 2\n")
            pending = await _poll(lambda: state.list_pending_changes(ctx.conn, pid))
            assert [p["path"] for p in pending] == ["a.py"]
        finally:
            await manager.stop()
        assert manager._polling_observer is None and manager._observer is None

    asyncio.run(scenario())


def test_mixed_observers_unschedule_their_own(tmp_path, monkeypatch):
    polled_root = tmp_path / "polled"
    native_root = tmp_path / "native"
    for r in (polled_root, native_root):
        r.mkdir()
        (r / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(
        watcher_mod,
        "_inotify_blind_fstype",
        lambda root: "9p" if root == polled_root else None,
    )

    async def scenario():
        ctx = make_ctx(tmp_path)
        polled = state.register_project(ctx.conn, polled_root, ctx.embedder.model_id)
        native = state.register_project(ctx.conn, native_root, ctx.embedder.model_id)
        for pid in (polled, native):
            state.set_project_flags(ctx.conn, pid, watch_enabled=True)
        manager = WatcherManager(
            ctx, debounce_s=0.05, quiet_s=0.2, poll_interval_s=0.1
        )
        manager.start()
        try:
            assert manager._observer is not None
            assert manager._polling_observer is not None
            assert manager.mode(polled) == "polling"
            assert manager.mode(native) == "native"
            # Unschedule each from its owning observer — the wrong observer
            # would raise KeyError here.
            manager.set_watch(polled, False)
            manager.set_watch(native, False)
            assert manager.mode(polled) is None
            for r in (polled_root, native_root):
                (r / "a.py").write_text("x = 2\n")
            await asyncio.sleep(0.5)
            assert state.list_pending_changes(ctx.conn, polled) == []
            assert state.list_pending_changes(ctx.conn, native) == []
        finally:
            await manager.stop()  # both observers alive → both stopped

    asyncio.run(scenario())


def test_set_watch_unschedules(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")

    async def scenario():
        ctx = make_ctx(tmp_path)
        pid = state.register_project(ctx.conn, root, ctx.embedder.model_id)
        state.set_project_flags(ctx.conn, pid, watch_enabled=True)
        manager = WatcherManager(ctx, debounce_s=0.05, quiet_s=0.2)
        manager.start()
        try:
            assert manager.watching(pid)
            manager.set_watch(pid, False)
            assert not manager.watching(pid)
            (root / "a.py").write_text("x = 2\n")
            await asyncio.sleep(0.4)
            assert state.list_pending_changes(ctx.conn, pid) == []
        finally:
            await manager.stop()

    asyncio.run(scenario())
