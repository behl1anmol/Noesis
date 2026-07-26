"""Regression tests for the PR #24 round-4 review findings.

All six concern what happens *after* the "absence is not evidence" guards
(ADR-51) fire — a guard that can never be satisfied strands the project it
protects:

1. A directory-level discovery error suppressed deletion and orphan pruning
   project-wide, so one permanently unreadable directory froze both forever
   for every other path. Suppression is now scoped to the unwalked subtrees,
   and stays project-wide only when the failure cannot be attributed to one.
2. The empty-scan guard had no escape. Discovery now reports whether the walk
   enumerated anything at all, so a fully filtered tree (an ADR-42 scope
   narrowing) converges, and an operator can force the genuinely-empty case
   (ADR-54).
3. The H1 dirty set only shrank in `set_last_indexed_commit`, which needs a
   commit — so on a non-git project it grew without bound and dragged the
   scoped candidate set with it.
4. Discovery-stage failures were labelled "hash failed" in run_file_errors.
5. A dir-only failure reported "all 1 files failed" with zero files failed.
"""

from __future__ import annotations

import os
from pathlib import Path

from noesis.core import discovery, state
from noesis.core.discovery import DiscoveryConfig, DiscoveryErrors, discover_files
from noesis.core.indexer import _under, _unwalked_prefixes, execute_run, prepare_run

from tests.test_gitfast import make_env


def _fail_scandir_on(monkeypatch, dirname: str) -> None:
    """Make `os.scandir` fail for one directory name, as a dead mount or a
    root-owned directory does — the failure os.walk routes to `onerror`."""
    real_scandir = os.scandir

    def flaky(path=".", *args, **kwargs):
        if Path(path).name == dirname:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky)


# --- unit: prefix attribution -------------------------------------------------


def test_unwalked_prefixes_attributes_a_subtree() -> None:
    errors = DiscoveryErrors(dirs=[("pkg", "EACCES"), ("a/b", "EIO")])
    assert _unwalked_prefixes(errors, follow_symlinks=False) == ("pkg", "a/b")


def test_unwalked_prefixes_voids_the_run_when_unattributable() -> None:
    """Three shapes where no rel-path prefix describes what went unseen, so
    the old project-wide suppression is the only sound answer."""
    root = DiscoveryErrors(dirs=[("<root>", "EACCES")])
    assert _unwalked_prefixes(root, follow_symlinks=False) is None

    # `relative_to` failed, so discovery recorded the absolute name.
    outside = DiscoveryErrors(dirs=[(os.path.join(os.sep, "mnt", "x"), "ESTALE")])
    assert _unwalked_prefixes(outside, follow_symlinks=False) is None

    # Following symlinks, the same directory is reachable under rel paths that
    # are not under its own, so containment stops implying "unseen".
    linked = DiscoveryErrors(dirs=[("pkg", "EACCES")])
    assert _unwalked_prefixes(linked, follow_symlinks=True) is None


def test_unwalked_prefixes_blocks_nothing_without_dir_errors() -> None:
    assert _unwalked_prefixes(DiscoveryErrors(), follow_symlinks=False) == ()


def test_under_respects_the_path_separator() -> None:
    """The containment test is the whole safety of finding 1: a plain
    `startswith` would sweep in the sibling `pkg2/`, whose walk succeeded."""
    assert _under("pkg/mod.py", ("pkg",))
    assert _under("pkg", ("pkg",))
    assert not _under("pkg2/mod.py", ("pkg",))
    assert not _under("pkgish.py", ("pkg",))
    assert not _under("top.py", ("pkg",))
    assert not _under("anything", ())


# --- unit: discovery reports what the walk saw --------------------------------


def test_entries_seen_counts_before_filtering(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "uv.lock").write_text("generated\n")  # skip-listed
    errors = DiscoveryErrors()

    assert discover_files(tmp_path, errors=errors) == ["a.py"]
    assert errors.entries_seen == 2


def test_entries_seen_separates_filtered_from_unreadable(tmp_path) -> None:
    """The discriminator finding 2 turns on: both walks return `[]`, and only
    `entries_seen` says whether the tree was readable."""
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "a.py").write_text("x = 1\n")
    empty = tmp_path / "empty"
    empty.mkdir()

    filtered = DiscoveryErrors()
    rust_only = DiscoveryConfig(include_languages=frozenset({"rust"}))
    assert discover_files(populated, rust_only, errors=filtered) == []
    assert filtered.entries_seen == 1

    unreadable = DiscoveryErrors()
    assert discover_files(empty, errors=unreadable) == []
    assert unreadable.entries_seen == 0


# --- finding 1: suppression scoped to the unwalked subtree --------------------


async def test_unreadable_directory_does_not_freeze_deletions_elsewhere(
    tmp_path, monkeypatch
) -> None:
    """A permanently unreadable directory (root-owned, dead mount) used to
    suppress deletion for the whole project on every run, forever. Deletions
    outside it were provable all along — os.walk loses only the failing
    directory's own subtree."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "gone.py").write_text("y = 2\n")
    (root / "pkg" / "mod.py").write_text("z = 3\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert set(state.get_file_states(conn, pid)) == {"top.py", "gone.py", "pkg/mod.py"}

    (root / "gone.py").unlink()
    _fail_scandir_on(monkeypatch, "pkg")
    second = await run()
    monkeypatch.undo()

    # Provable outside the failed subtree: deleted.
    assert second.files_deleted == 1
    assert "gone.py" not in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("gone.py", 0) == 0
    # Unprovable inside it: untouched and re-queued as itself.
    assert "pkg/mod.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0
    assert "pkg/mod.py" in second.failed_paths


async def test_unattributable_dir_error_still_suppresses_the_whole_run(
    tmp_path, monkeypatch
) -> None:
    """Over-correction guard: when the failure is the walk ROOT, nothing was
    seen and no prefix describes it — the project-wide suppression must
    remain."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id

    resolved = root.resolve()
    real_scandir = os.scandir

    def broken(path=".", *args, **kwargs):
        if Path(path) == resolved:
            raise PermissionError(13, "Permission denied", str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", broken)
    second = await run()
    monkeypatch.undo()

    assert second.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py", "b.py"}
    assert state.get_latest_run(conn, pid)["status"] == "failed"


async def test_orphan_pruning_scoped_to_the_unwalked_subtree(
    tmp_path, monkeypatch
) -> None:
    """Same split on the drift path: a true orphan outside the failed subtree
    is pruned, while the live-file-with-no-state-row inside it keeps its
    points (the crash window between `upsert_chunks` and `upsert_file`)."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "ghost.py").write_text("y = 2\n")
    (root / "pkg" / "mod.py").write_text("z = 3\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id

    # ghost.py: absent from state AND from disk — a true orphan.
    state.delete_files(conn, pid, ["ghost.py"])
    (root / "ghost.py").unlink()
    # pkg/mod.py: absent from state but live on disk, and about to be hidden
    # by the failed walk — an orphan *candidate* that must be spared.
    state.delete_files(conn, pid, ["pkg/mod.py"])

    _fail_scandir_on(monkeypatch, "pkg")
    await run()
    monkeypatch.undo()

    assert store.per_file_point_counts(pid).get("ghost.py", 0) == 0
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0


# --- finding 2: the empty-scan guard can now converge -------------------------


async def test_narrowed_scope_converges_without_a_force(tmp_path) -> None:
    """ADR-42 narrowing on a perfectly readable root: the walk enumerates
    files and the filter removes all of them, so the emptiness is a scope
    decision and the now-out-of-scope files must leave the index."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(config=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            discovery_config=config,
        )

    first = await run()
    pid = first.project_id
    assert set(state.get_file_states(conn, pid)) == {"a.py"}

    second = await run(DiscoveryConfig(include_languages=frozenset({"rust"})))

    assert second.files_deleted == 1
    assert state.get_file_states(conn, pid) == {}
    assert store.per_file_point_counts(pid).get("a.py", 0) == 0
    assert state.get_latest_run(conn, pid)["status"] == "done"


async def test_force_accepts_a_genuinely_empty_root(tmp_path) -> None:
    """The operator's assertion. Nothing observable separates an emptied root
    from an unmounted one, so without this the project could never converge
    and the only escape was delete-and-re-register."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(force=False):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            force=force,
        )

    first = await run()
    pid = first.project_id
    (root / "a.py").unlink()

    refused = await run()
    assert refused.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == {"a.py"}
    assert state.get_latest_run(conn, pid)["status"] == "failed"

    forced = await run(force=True)
    assert forced.files_deleted == 1
    assert state.get_file_states(conn, pid) == {}
    assert store.per_file_point_counts(pid).get("a.py", 0) == 0
    assert state.get_latest_run(conn, pid)["status"] == "done"


async def test_force_does_not_relax_directory_errors(tmp_path, monkeypatch) -> None:
    """Over-correction guard: `force` asserts an empty root is real. It says
    nothing about a subtree that failed to scan, which is a separate and still
    open proof problem."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("x = 1\n")
    (root / "pkg" / "mod.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(force=False):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            force=force,
        )

    first = await run()
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    second = await run(force=True)
    monkeypatch.undo()

    assert second.files_deleted == 0
    assert "pkg/mod.py" in state.get_file_states(conn, pid)
    assert store.per_file_point_counts(pid).get("pkg/mod.py", 0) > 0


# --- finding 3: the H1 dirty set must be able to shrink -----------------------


async def test_clean_full_run_clears_the_dirty_set_without_a_commit(
    tmp_path,
) -> None:
    """On a non-git project `set_last_indexed_commit` never fires, so nothing
    ever removed from the dirty set while every scoped run unioned into it —
    the scoped candidate set crept toward "every file ever indexed"."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(paths=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            paths=paths,
        )

    first = await run()
    pid = first.project_id

    (root / "a.py").write_text("x = 2\n")
    await run(paths=["a.py"])
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py"})

    # A full walk re-hashes every discovered file, so nothing is owed H1
    # re-admission afterwards.
    await run()
    assert state.get_dirty_paths(conn, pid) == frozenset()


async def test_dirty_set_survives_a_full_run_that_was_not_clean(
    tmp_path, monkeypatch
) -> None:
    """Over-correction guard: the clear is only sound because a clean full
    walk re-hashed everything. A run with any discovery error did not, so the
    re-admission set must stay."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    (root / "pkg" / "mod.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run(paths=None):
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn,
            store,
            embedder,
            str(root),
            project_id,
            run_id,
            git_fast_path=False,
            paths=paths,
        )

    first = await run()
    pid = first.project_id
    (root / "a.py").write_text("x = 2\n")
    await run(paths=["a.py"])
    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py"})

    _fail_scandir_on(monkeypatch, "pkg")
    await run()
    monkeypatch.undo()

    assert state.get_dirty_paths(conn, pid) == frozenset({"a.py"})


# --- findings 4 and 5: what the operator reads --------------------------------


async def test_discovery_errors_are_labelled_by_their_own_stage(
    tmp_path, monkeypatch
) -> None:
    """`diff.errored` carries both stages; labelling all of it "hash failed"
    pointed the operator at hashing for a failure that happened in the stat."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("y = 2\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    assert pid

    real = discovery._is_binary

    def flaky(path):
        if path.name == "a.py":
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(discovery, "_is_binary", flaky)
    second = await run()
    monkeypatch.undo()

    errors = {row["path"]: row["error"] for row in
              state.list_file_errors(conn, second.run_id)}
    assert errors["a.py"].startswith("discovery failed: ")
    assert "hash failed" not in errors["a.py"]


async def test_dir_only_failure_does_not_claim_files_failed(tmp_path) -> None:
    """The empty-root guard produces a dir error and zero file errors, yet the
    run reported "all 1 files failed"."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    conn, store, embedder = make_env(tmp_path)

    async def run():
        project_id, run_id = prepare_run(conn, embedder, str(root))
        return await execute_run(
            conn, store, embedder, str(root), project_id, run_id, git_fast_path=False
        )

    first = await run()
    pid = first.project_id
    (root / "a.py").unlink()
    await run()

    row = state.get_latest_run(conn, pid)
    assert row["status"] == "failed"
    assert row["error"] == "discovery failed for 1 location(s); no files could be verified"
