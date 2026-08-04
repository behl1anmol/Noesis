"""Issue #28 — the git anchor no longer latches on a failure it can name (ADR-60).

PR #24 (ADR-51/54/55) stopped discovery from reading a filesystem fault as
deletion evidence. To do it, it hardened one gate: a run with *any* failure does
not advance ``projects.last_indexed_commit``. That was right about deletion and
too blunt about the anchor — one permanently unreadable directory, or one
permanently unreadable *file*, froze the anchor for the whole project forever,
and ``gitfast.compute_candidates`` then diffed against an ever-older commit
while the candidate set grew with every commit since.

ADR-60 keeps the property and pays for it differently. The gate exists so that
**the next fast path never skips a file whose state is unknown**. That is
delivered by NAMING those files and carrying them in ``dirty_paths`` — which
already means "owed re-admission next run", and which the H1 union feeds
straight back into the candidate set — rather than by refusing to move.

So the rule is: *advance when the doubt can be named; block when it cannot.*

Two things still block completely, and they have the first two tests in this
file because they are what stops this being an over-correction:

* an **unattributable** failure (the ``<root>`` sentinel, a path outside the
  root, ``follow_symlinks``) — the walk proved nothing about which paths are
  affected;
* a run that finished ``failed`` — §3.2 rule 4, untouched.

Coverage of ``clean_run``'s four terms is split across three files by which
question each one belongs to, so nothing is asserted twice:

* ``file_errors`` — ``test_gitfast.py::test_anchor_advances_carrying_a_contained_failure``
  (end to end: the next run re-indexes the file);
* ``hash_errors`` and ``dir_errors`` — here;
* ``screening_held`` — ``test_issue26_screening_faults.py::test_a_held_back_deletion_is_carried_past_the_anchor``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noesis.core import hashdiff, state
from noesis.core.indexer import execute_run, index_project, prepare_run

from tests.test_gitfast import (
    anchor_of,
    build_files,
    git,
    git_head,
    make_env,
    requires_git,
)
from tests.test_unwalkable_dirs import _fail_scandir_at_root, _fail_scandir_on

# A tree with a subdirectory, so "the subtree that could not be walked" is a
# real prefix rather than the whole project.
TREE = {
    "app.py": "app = 1\n",
    "util.py": "util = 1\n",
    "pkg/mod.py": "mod = 1\n",
    "pkg/other.py": "other = 1\n",
}
UNDER_PKG = frozenset({"pkg/mod.py", "pkg/other.py"})

# Distinct files for the growth test to touch, one per commit. The candidate
# set behind a frozen anchor is the union of DISTINCT paths changed since it,
# so editing ONE file repeatedly leaves it one wide however many commits land —
# a trap the first version of that test fell into, where the growth assertion
# passed against the pre-fix tree and proved nothing.
ROTATE = [f"r{i}.py" for i in range(5)]
TREE.update({rel: f"{rel[:-3]} = 1\n" for rel in ROTATE})


def build_repo(root: Path) -> None:
    build_files(root, TREE)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")


def commit_edit(root: Path, rel: str, text: str) -> str:
    (root / rel).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", f"edit {rel}")
    return git_head(root)


# --- the two over-correction guards, first ------------------------------------


@requires_git
async def test_unattributable_failure_still_blocks_the_anchor_completely(
    tmp_path, monkeypatch
) -> None:
    """OVER-CORRECTION GUARD, and the one issue #28 asked for by name.

    When the walk root itself cannot be read, no prefix describes what went
    unseen. ``unverified`` does still technically enumerate the paths — it is
    every stored path — so this is a judgement rather than a necessity, and the
    judgement is: advancing here would write the ENTIRE index into a JSON
    column on every run of a broken-root project, to buy a next run that hashes
    everything anyway. The walk proved nothing. Nothing moves.

    Passes in both directions by construction, which is the point of a guard —
    it is not evidence that anything was fixed, it is protection against the
    obvious wrong version of the fix.
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    anchor = anchor_of(conn, pid)
    assert anchor == git_head(repo)
    assert state.get_dirty_paths(conn, pid) == frozenset()

    head = commit_edit(repo, "app.py", "app = 2\n")
    assert head != anchor

    _fail_scandir_at_root(monkeypatch, repo)
    result = await index_project(conn, store, embedder, str(repo))
    monkeypatch.undo()

    assert anchor_of(conn, pid) == anchor, "the anchor must not move"
    assert state.get_dirty_paths(conn, pid) == frozenset(), "nothing carried"
    # And the reason the run is not evidence: it saw nothing at all.
    assert result.files_deleted == 0
    assert set(state.get_file_states(conn, pid)) == set(TREE)
    conn.close()


@requires_git
async def test_a_clean_run_carries_nothing(tmp_path) -> None:
    """OVER-CORRECTION GUARD in the other direction: do not carry what is not
    owed.

    The new rule has to SUBSUME the old one, not sit beside it. On a healthy
    project every term of ``clean_run`` is empty, so the carried set is empty
    and ``dirty_paths`` holds exactly what it held before ADR-60 — the
    working-tree-dirty set and nothing else. A regression that over-carries
    (say, folding in every discovered path) would make every subsequent scoped
    run re-hash the whole tree, silently.

    Passes in both directions by construction.
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    assert state.get_dirty_paths(conn, pid) == frozenset()

    # One uncommitted edit: git status reports it, so it — and only it — is the
    # legitimate dirty set at the next anchor.
    (repo / "util.py").write_text("util = 99\n")
    await index_project(conn, store, embedder, str(repo))

    assert anchor_of(conn, pid) == git_head(repo)
    assert state.get_dirty_paths(conn, pid) == frozenset({"util.py"})
    conn.close()


# --- the fix: each nameable failure advances and carries -----------------------


@requires_git
async def test_attributable_directory_error_advances_and_carries(
    tmp_path, monkeypatch
) -> None:
    """`dir_errors` — the term issue #28 is titled after.

    The stored paths under the unwalkable prefix are exactly what this run
    could not verify, so they are what rides along with the anchor.
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    anchor = anchor_of(conn, pid)

    head = commit_edit(repo, "app.py", "app = 2\n")
    assert head != anchor

    _fail_scandir_on(monkeypatch, "pkg")
    result = await index_project(conn, store, embedder, str(repo))
    monkeypatch.undo()

    assert anchor_of(conn, pid) == head, "the anchor tracks HEAD again"
    assert state.get_dirty_paths(conn, pid) == UNDER_PKG
    # Carrying is a RETRY decision and never a deletion one — the separation
    # PR #24 and PR #30 both turn on, and ADR-60 does not touch.
    assert result.files_deleted == 0
    assert UNDER_PKG <= set(state.get_file_states(conn, pid))
    conn.close()


@requires_git
async def test_unreadable_file_advances_and_carries(tmp_path, monkeypatch) -> None:
    """`hash_errors` — the two-thirds of the population a directory-only fix
    would have missed.

    A permanently unreadable FILE pins the anchor exactly as a directory does,
    and leaves no ``unwalkable_dirs`` row for anything to key on — the sibling
    instance the PR #30 round-2 review found for the promotion trigger, here in
    its original home. Keying ADR-60 on the OUTCOME (can the doubt be named?)
    rather than on the cause covers it with no special case.
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    anchor = anchor_of(conn, pid)

    # Commit it so the fast path makes it a candidate — a non-candidate is
    # never hashed, so the fault would not fire at all (the property that made
    # `last_full_run_was_clean` need "full walk" rather than "last run").
    head = commit_edit(repo, "util.py", "util = 2\n")
    assert head != anchor

    real_hash = hashdiff.hash_file

    def flaky(path, *args, **kwargs):
        if Path(path).name == "util.py":
            raise PermissionError(13, "Permission denied", str(path))
        return real_hash(path, *args, **kwargs)

    monkeypatch.setattr(hashdiff, "hash_file", flaky)
    result = await index_project(conn, store, embedder, str(repo))
    monkeypatch.undo()

    assert result.files_failed == 1
    assert anchor_of(conn, pid) == head
    assert "util.py" in state.get_dirty_paths(conn, pid)
    # H7: an unreadable file is not a deleted one.
    assert result.files_deleted == 0
    assert "util.py" in state.get_file_states(conn, pid)
    conn.close()


@requires_git
async def test_directory_hiding_nothing_indexed_stops_latching(
    tmp_path, monkeypatch
) -> None:
    """Gate on what was put in doubt, not on the fault — ADR-58's own rule,
    extended to the other terms.

    An unreadable directory holding nothing this project has indexed puts no
    stored path in doubt, so it carries nothing and must not hold the anchor
    back. Under the old gate the mere presence of a `dir_errors` entry froze
    the project, which is the "guard that can never be satisfied" shape ADR-55
    was written to avoid.
    """
    repo = tmp_path / "repo"
    build_files(repo, {"app.py": "app = 1\n"})
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    assert set(state.get_file_states(conn, pid)) == {"app.py"}

    # Appears after the project was indexed, so nothing under it is stored.
    (repo / "vendor").mkdir()
    (repo / "vendor" / "blob.py").write_text("blob = 1\n")
    head = commit_edit(repo, "app.py", "app = 2\n")

    _fail_scandir_on(monkeypatch, "vendor")
    result = await index_project(conn, store, embedder, str(repo))
    monkeypatch.undo()

    assert result.files_failed == 1, "the directory error is still reported"
    assert anchor_of(conn, pid) == head
    assert state.get_dirty_paths(conn, pid) == frozenset()
    conn.close()


# --- the consequences the issue is actually about ------------------------------


@requires_git
async def test_the_candidate_set_stops_growing_on_a_latched_project(
    tmp_path, monkeypatch
) -> None:
    """The headline cost issue #28 reports, measured as an assertion.

    With the anchor frozen, ``compute_candidates`` diffs against an ever-older
    commit, so the candidate set grows by one file per commit and never stops.
    With the anchor tracking HEAD it is flat: this run's real change, plus the
    fixed set of paths the broken subtree hides.

    The assertion is on the SHAPE rather than on absolute numbers, so it stays
    honest if the fixture changes size: pre-ADR-60 the sequence increases with
    every run and never stops; post-ADR-60 it settles and stays settled.

    Measured here: pre-fix ``[1, 2, 3, 4, 5]``, post-fix ``[1, 3, 3, 3, 3]``.
    The first entry is lower rather than higher, and the reason is worth
    knowing — that run is the one that *discovers* the fault, so it opens with
    an empty carried set and closes by recording one. Every run after it opens
    and closes with the same set. That is the settled state, and it is what the
    pre-fix sequence never reaches.
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id

    _fail_scandir_on(monkeypatch, "pkg")
    counts = []
    for i, rel in enumerate(ROTATE):
        # A DIFFERENT file each commit — see ROTATE's note. Editing one file
        # repeatedly keeps the frozen anchor's diff one path wide, so the
        # pre-fix tree would look flat too and this test would prove nothing.
        commit_edit(repo, rel, f"{rel[:-3]} = {i + 2}\n")
        result = await index_project(conn, store, embedder, str(repo))
        assert result.fast_path_used is True
        counts.append(result.candidate_count)
    monkeypatch.undo()

    settled = counts[1:]
    assert len(set(settled)) == 1, f"should settle after run 1, got {counts}"
    assert settled[0] == 1 + len(UNDER_PKG), (
        "settles at this run's one real change plus the paths pkg/ hides"
    )
    # The property in its own right, independent of the numbers above: the
    # sequence must never grow. Pre-fix it grows by exactly one per commit.
    assert all(b <= a for a, b in zip(settled, settled[1:]))
    assert anchor_of(conn, pid) == git_head(repo)
    conn.close()


async def test_a_faulted_full_walk_drains_to_the_carried_floor(tmp_path) -> None:
    """The non-git branch, and the "bounded sawtooth" claim behind it.

    Scoped runs union what they indexed into the dirty set and never drain it
    (ADR-40), so on a non-git project it climbed without bound and dragged every
    later scoped run's candidate set up with it. The drain existed but was gated
    on a clean run, so one permanent fault pinned it (issue #28).

    Now a full walk on a faulted project drains to the FLOOR — the paths it
    could not verify — instead of not draining at all. That floor is what
    ``state.last_full_run_was_clean`` deliberately still refuses to count
    toward the promotion fraction: it is real, and no cadence gets under it.
    """
    root = tmp_path / "proj"
    build_files(root, TREE)
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
    assert state.get_dirty_paths(conn, pid) == frozenset()

    # Three scoped runs accumulate, exactly as they did before.
    for rel in ("app.py", "util.py", "pkg/mod.py"):
        (root / rel).write_text("changed = 1\n")
        await run(paths=[rel])
    assert state.get_dirty_paths(conn, pid) == frozenset(
        {"app.py", "util.py", "pkg/mod.py"}
    )

    # A full walk that hits a permanent fault. It re-hashed everything it could
    # reach, so only what it could NOT reach is still owed. (A local
    # MonkeyPatch rather than the fixture: the fault must start AFTER the
    # scoped runs above have built the set this walk is meant to drain.)
    monkeypatch = pytest.MonkeyPatch()
    _fail_scandir_on(monkeypatch, "pkg")
    result = await run()
    monkeypatch.undo()

    assert state.get_dirty_paths(conn, pid) == UNDER_PKG
    assert result.files_deleted == 0
    conn.close()


@requires_git
async def test_an_edit_hidden_by_an_unreadable_dir_survives_the_anchor_moving(
    tmp_path, monkeypatch
) -> None:
    """The load-bearing end-to-end case, and the one that would make ADR-60
    unsafe if it were wrong.

    A file edited and committed while its directory is unreadable, with the
    anchor then advancing past that commit, is the scenario the old gate was
    protecting against: once the anchor is past it, ``git diff`` will never
    mention that file again, so anything that forgets it leaves stale content
    indexed permanently.

    Stated honestly per lesson #14: the *content* assertion at the end passes
    against the pre-fix tree too — there the anchor never moved, so the diff
    still covered the file. That makes this a regression guard on the recovery
    path rather than proof of the fix. What fails pre-fix is the anchor
    assertion in the middle. Both belong in one test, because "the anchor moved"
    is only acceptable together with "and the edit still landed".
    """
    repo = tmp_path / "repo"
    build_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    first = await index_project(conn, store, embedder, str(repo))
    pid = first.project_id
    original = state.get_file_states(conn, pid)["pkg/mod.py"]

    _fail_scandir_on(monkeypatch, "pkg")
    # The edit nothing will be able to see, committed while pkg is unreadable.
    commit_edit(repo, "pkg/mod.py", "mod = 999\n")
    await index_project(conn, store, embedder, str(repo))
    # Two more commits, so the anchor moves well past the hidden one.
    commit_edit(repo, "app.py", "app = 2\n")
    await index_project(conn, store, embedder, str(repo))
    head = commit_edit(repo, "util.py", "util = 2\n")
    await index_project(conn, store, embedder, str(repo))

    assert anchor_of(conn, pid) == head, "the anchor kept up with HEAD"
    assert state.get_dirty_paths(conn, pid) == UNDER_PKG
    assert state.get_file_states(conn, pid)["pkg/mod.py"] == original, (
        "still the stale hash — the run never got to read it"
    )

    # pkg becomes readable. git has nothing to say (no commit since the anchor),
    # so the ONLY reason this file is re-examined is that it was carried.
    monkeypatch.undo()
    result = await index_project(conn, store, embedder, str(repo))

    assert state.get_file_states(conn, pid)["pkg/mod.py"] != original
    assert result.files_deleted == 0
    # And once verified it stops being owed: the carry drains, it does not pin.
    assert state.get_dirty_paths(conn, pid) == frozenset()
    conn.close()
