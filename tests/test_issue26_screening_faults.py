"""Issue #26 — the three unguarded per-entry filesystem calls in the walk loop.

PR #24 rewrote `discovery.py` so that no filesystem fault could make a path
vanish from the scan unrecorded, because downstream "absent from the walk" is
indistinguishable from "deleted from disk". Rounds 7 and 8 of that review found
three calls the rewrite missed, and deferred them here because the fix is a
taxonomy decision rather than a mechanism.

CPython's `_ignore_error` covers only `(ENOENT, ENOTDIR, EBADF, ELOOP)`
(`pathlib.py:31`, verified on the 3.12.3 interpreter this project runs), so
`Path.is_file()` and `Path.is_symlink()` re-raise EACCES and ESTALE:

* `gitignore.is_file()` ran once per walked directory, unconditionally, and
  took the whole run down with it — plus `structural_search` (an unhandled 500)
  and the registration preview, neither of which passes an error collector.
* the per-child `is_symlink()` did the same.
* `_IgnoreStack.push`'s `read_text` went the other way and swallowed its
  failure with a bare `return`.

ADR-58 sorts them into two new collector lists, and the split is the whole
safety argument:

* `unscreened` (the two .gitignore sites) SUPPRESSES DELETION under its own
  directory prefix. Not for the reason the issue gave — it filed this as purely
  a risk of *under*-ignoring — but because the fault also runs the other way.
  Git is last-match-wins, so a deeper .gitignore can negate a shallower
  exclusion; lose the deeper spec and the outer exclusion stands unopposed, the
  file drops out of the results, and every consumer downstream reads that
  absence as a deletion. `test_lost_gitignore_negation_does_not_delete_the_file`
  is that case, and it is the reason "log it and keep going" was not enough.
* `unidentified` (the symlink/cycle-guard sites) suppresses NOTHING. It can
  only make the walk over-inclusive — a subtree walked twice, or a link
  descended into — so it is never evidence that a file is gone. That is the
  round-7 finding, and `test_identity_fault_never_suppresses_a_deletion` keeps
  it true now that the fault is recorded rather than merely logged.

Test discipline (lesson #14). Every test below was run against `bfa5104` (the
merge of PR #30, this branch's parent) with `PYTHONPATH` pointed at its `src/`,
and observed to fail there *for the reason it is written* — the four crash
sites on the escaped `PermissionError`, the negation test on
`files_deleted == 1`, the anchor test on an anchor that moved.

Two are declared over-correction guards and pass in both directions by
construction, labelled in their own docstrings. A third
(`test_identity_fault_never_suppresses_a_deletion`) reads like one but is not:
pre-fix the walk raises before it can prove anything, so it flips. It is kept
because it pins the round-7 invariant against a future change that routes
identity faults into the suppressing list.

EACCES is injected rather than produced: the container runs as uid 0 and root
bypasses DAC permission checks, so `chmod` cannot create the condition. Issue
#26 states the same caveat about its own verification, and the round-7 tests
established the monkeypatch pattern reused here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from noesis.core import state
from noesis.core.discovery import DiscoveryConfig, DiscoveryErrors, discover_files
from noesis.core.indexer import execute_run, prepare_run
from noesis.core.structural import structural_search

from tests.test_gitfast import git, git_head, make_env, requires_git

DISCOVERY_LOGGER = "noesis.core.discovery"


# --- fault injection ----------------------------------------------------------
#
# Each helper breaks ONE method for ONE path and delegates everything else. A
# blanket patch is not usable: `Path.stat` is reached by the file-screening loop
# as well as by `is_file()`, and breaking it wholesale would fail the walk for
# reasons that have nothing to do with the site under test.


def _fail_method_on(monkeypatch, method: str, target: Path) -> None:
    # The target is resolved once, HERE, before the patch is installed. Doing it
    # inside `flaky` is what the first draft did and it recursed to death:
    # `Path.resolve()` calls `stat()`, which for `method="stat"` is the very
    # function being defined. `discover_files` resolves its root and builds
    # every path from it, so plain equality is enough — and it is what the
    # round-7 helper does for the same reason.
    real = getattr(Path, method)
    resolved = target.resolve()

    def flaky(self, *args, **kwargs):
        if Path(self) == resolved:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, flaky)


def _tree(tmp_path: Path) -> Path:
    """A root and one subdirectory, each holding an indexable file."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "top.py").write_text("top = 1\n")
    (root / "pkg" / "mod.py").write_text("mod = 2\n")
    return root


def _negation_tree(tmp_path: Path) -> Path:
    """The shape where losing an ignore file DELETES rather than admits.

    The root excludes `*.gen.py`; `generated/.gitignore` negates that for one
    file. With both specs loaded the file is kept — with the deeper one lost,
    the outer exclusion wins and the file leaves the results entirely.
    """
    root = tmp_path / "proj"
    (root / "generated").mkdir(parents=True)
    (root / ".gitignore").write_text("*.gen.py\n")
    (root / "generated" / ".gitignore").write_text("!important.gen.py\n")
    (root / "generated" / "important.gen.py").write_text("important = 1\n")
    (root / "keep.py").write_text("keep = 2\n")
    return root


async def _run(conn, store, embedder, root, **kwargs):
    project_id, run_id = prepare_run(conn, embedder, str(root))
    result = await execute_run(
        conn, store, embedder, str(root), project_id, run_id, **kwargs
    )
    return project_id, result


# --- 1-3: the three sites no longer crash, and no longer vanish ---------------


def test_unstattable_gitignore_no_longer_kills_the_walk(tmp_path, monkeypatch):
    """Site A. `is_file()` re-raises EACCES; it ran once per walked directory."""
    root = _tree(tmp_path)
    _fail_method_on(monkeypatch, "is_file", root / "pkg" / ".gitignore")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(), errors=errors)

    # Pre-fix this line was never reached — the PermissionError escaped
    # `discover_files` and `execute_run` marked the whole run failed.
    assert sorted(found) == ["pkg/mod.py", "top.py"]
    assert [key for key, _ in errors.unscreened] == ["pkg"]
    # The errno is the actionable half, so assert it survived rather than the
    # sentence it arrived in (lesson #14).
    assert "Permission denied" in errors.unscreened[0][1]
    # Never existence evidence: nothing went unseen and no file failed.
    assert errors.dirs == []
    assert errors.files == []


def test_unstattable_child_symlink_test_no_longer_kills_the_walk(tmp_path, monkeypatch):
    """Site B. Same re-raise, on the per-child `is_symlink()` in the dir loop."""
    root = _tree(tmp_path)
    _fail_method_on(monkeypatch, "is_symlink", root / "pkg")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(), errors=errors)

    # The child is still WALKED. Pruning an unstattable directory would hide
    # what it holds and hand the deletion path a set of "absent" files nobody
    # looked for — purge on doubt, which ADR-51 refuses.
    assert sorted(found) == ["pkg/mod.py", "top.py"]
    assert [key for key, _ in errors.unidentified] == ["pkg"]
    assert "Permission denied" in errors.unidentified[0][1]
    assert errors.dirs == []
    assert errors.files == []
    # An identity fault is not a screening fault: it must not reach the list
    # that suppresses deletions.
    assert errors.unscreened == []


def test_unreadable_gitignore_is_recorded_not_only_logged(tmp_path, monkeypatch):
    """Site C. Round 8 made it log; ADR-58 makes it a recorded, scoped fault."""
    root = _tree(tmp_path)
    (root / "pkg" / ".gitignore").write_text("*.log\n")
    _fail_method_on(monkeypatch, "read_text", root / "pkg" / ".gitignore")

    errors = DiscoveryErrors()
    found = discover_files(root, DiscoveryConfig(), errors=errors)

    # `.gitignore` is itself an ordinary indexable text file — it is unreadable
    # here, not unstattable, so it still passes screening and lands in the
    # results. Only the RULES it holds were lost.
    assert sorted(found) == ["pkg/.gitignore", "pkg/mod.py", "top.py"]
    assert [key for key, _ in errors.unscreened] == ["pkg"]
    assert "Permission denied" in errors.unscreened[0][1]
    assert errors.dirs == []
    assert errors.files == []


def test_persisted_screening_detail_carries_no_absolute_path(tmp_path, monkeypatch):
    """PR #31 review: `str(exc)` interpolates `filename`, so the obvious
    one-liner persisted the absolute `.gitignore` path into `run_file_errors`
    and onto the dashboard — while `_record_degraded`'s own docstring promised
    the opposite. The recorded pair is keyed on the directory rel already, so
    the path added nothing but the machine-specific root prefix (ADR-25).

    Asserted on the invariant rather than the wording: the errno and the
    strerror must survive (they are the actionable half), the absolute path
    must not, and the DEBUG log must still carry it for whoever has to go fix
    the directory.
    """
    root = _tree(tmp_path)
    (root / "pkg" / ".gitignore").write_text("*.log\n")
    _fail_method_on(monkeypatch, "read_text", root / "pkg" / ".gitignore")

    errors = DiscoveryErrors()
    discover_files(root, DiscoveryConfig(), errors=errors)

    _, message = errors.unscreened[0]
    assert str(root) not in message
    assert str(root / "pkg" / ".gitignore") not in message
    # The actionable half is still there.
    assert "Permission denied" in message
    assert "13" in message


def test_collectorless_callers_still_get_a_log_line(tmp_path, monkeypatch, caplog):
    """`errors=None` keeps the silent-skip contract — but never a silent loss.

    Structural search and the registration preview pass no collector. They must
    not start raising, and the fault must still leave a trace.
    """
    root = _tree(tmp_path)
    _fail_method_on(monkeypatch, "is_file", root / "pkg" / ".gitignore")

    with caplog.at_level(logging.DEBUG, logger=DISCOVERY_LOGGER):
        found = discover_files(root, DiscoveryConfig())

    assert sorted(found) == ["pkg/mod.py", "top.py"]
    assert any(
        rec.name == DISCOVERY_LOGGER and "Permission denied" in rec.getMessage()
        for rec in caplog.records
    )


# --- 4: the headline regression ----------------------------------------------


@pytest.mark.asyncio
async def test_lost_gitignore_negation_does_not_delete_the_file(tmp_path, monkeypatch):
    """A lost .gitignore can DELETE, not just admit — the claim #26 got wrong.

    Both the issue and the shipped ADR-51 row asserted that losing an ignore
    file "cannot cause over-deletion". Git's last-match-wins semantics say
    otherwise: the deeper spec's negation stops applying, the outer exclusion
    wins, and the file leaves `discovered` — which `hashdiff.partition` reports
    as `deleted`. Against `bfa5104` this test fails with the file's chunks
    purged and its state row gone.
    """
    conn, store, embedder = make_env(tmp_path)
    root = _negation_tree(tmp_path)

    project_id, first = await _run(conn, store, embedder, root)
    # The .gitignore files are ordinary indexable text and appear here too;
    # what matters is that the negated file is present to begin with.
    assert set(state.get_file_states(conn, project_id)) == {
        ".gitignore",
        "generated/.gitignore",
        "generated/important.gen.py",
        "keep.py",
    }
    assert (
        store.per_file_point_counts(project_id).get("generated/important.gen.py", 0) > 0
    )

    # Second run: the negating spec cannot be read, so the root's `*.gen.py`
    # exclusion stands unopposed and the file drops out of the walk.
    _fail_method_on(monkeypatch, "read_text", root / "generated" / ".gitignore")
    _, second = await _run(conn, store, embedder, root)

    assert second.files_deleted == 0, "a screening fault was read as a deletion"
    assert "generated/important.gen.py" in state.get_file_states(conn, project_id)
    assert (
        store.per_file_point_counts(project_id).get("generated/important.gen.py", 0) > 0
    )
    # Reported, not silent — and not dressed up as a failure either.
    assert second.discovery_degraded == 1
    assert second.files_failed == 0
    errors = {
        row["path"]: row["error"] for row in state.list_file_errors(conn, second.run_id)
    }
    assert "<screening>:generated" in errors
    assert "screening degraded" in errors["<screening>:generated"]
    # The control file, outside the degraded prefix, is untouched throughout.
    assert "keep.py" in state.get_file_states(conn, project_id)
    assert first.run_id != second.run_id


# --- 5, 7: over-correction guards --------------------------------------------


@pytest.mark.asyncio
async def test_screening_suppression_is_contained_on_the_separator(
    tmp_path, monkeypatch
):
    """OVER-CORRECTION GUARD — passes before and after the fix by design.

    ADR-54's containment rule applied to the new prefix list. `src/my_module`
    must not shield `src/my_moduleX/...`: `_under` compares on the separator,
    not with `startswith`, and directory names contain the neighbouring
    characters constantly.
    """
    conn, store, embedder = make_env(tmp_path)
    root = tmp_path / "proj"
    (root / "src" / "my_module").mkdir(parents=True)
    (root / "src" / "my_moduleX").mkdir(parents=True)
    (root / "src" / "my_module" / ".gitignore").write_text("*.tmp\n")
    (root / "src" / "my_module" / "a.py").write_text("a = 1\n")
    (root / "src" / "my_moduleX" / "gone.py").write_text("gone = 2\n")

    project_id, _ = await _run(conn, store, embedder, root)
    assert "src/my_moduleX/gone.py" in state.get_file_states(conn, project_id)

    # Delete the neighbour for real, and degrade the OTHER directory.
    (root / "src" / "my_moduleX" / "gone.py").unlink()
    _fail_method_on(monkeypatch, "read_text", root / "src" / "my_module" / ".gitignore")
    _, second = await _run(conn, store, embedder, root)

    assert second.files_deleted == 1
    assert "src/my_moduleX/gone.py" not in state.get_file_states(conn, project_id)
    assert "src/my_module/a.py" in state.get_file_states(conn, project_id)


@pytest.mark.asyncio
async def test_identity_fault_never_suppresses_a_deletion(tmp_path, monkeypatch):
    """The round-7 invariant, now that the fault is recorded instead of logged.

    An identity fault says nothing about whether any file exists — `os.walk`
    already enumerated that directory and its files are in the result. Routing
    it anywhere that suppresses deletion would freeze pruning for a fully
    walked subtree, which is ADR-54's over-correction reached from the other
    side. This is what keeps `unidentified` and `unscreened` separate lists
    rather than one convenient bucket.
    """
    conn, store, embedder = make_env(tmp_path)
    root = _tree(tmp_path)

    project_id, _ = await _run(conn, store, embedder, root)
    assert "pkg/mod.py" in state.get_file_states(conn, project_id)

    (root / "pkg" / "mod.py").unlink()
    _fail_method_on(monkeypatch, "is_symlink", root / "pkg")
    _, second = await _run(conn, store, embedder, root)

    assert second.files_deleted == 1, "an identity fault blocked a real deletion"
    assert "pkg/mod.py" not in state.get_file_states(conn, project_id)
    assert second.discovery_degraded == 1


# --- 6: the anchor rule, in both directions ----------------------------------


@requires_git
@pytest.mark.asyncio
async def test_held_back_deletion_blocks_the_anchor(tmp_path, monkeypatch):
    """A file wrongly excluded is a file whose stored hash may now be stale.

    Advancing past it would leave the next fast path with no reason to look at
    it again — the same rule `file_errors` and `dir_errors` already follow.
    """
    conn, store, embedder = make_env(tmp_path)
    root = _negation_tree(tmp_path)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")

    project_id, _ = await _run(conn, store, embedder, root)
    first_anchor = state.get_project(conn, project_id)["last_indexed_commit"]
    assert first_anchor == git_head(root)

    (root / "keep.py").write_text("keep = 3\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "second")
    assert git_head(root) != first_anchor

    _fail_method_on(monkeypatch, "read_text", root / "generated" / ".gitignore")
    _, second = await _run(conn, store, embedder, root)

    # Anchor first, deliberately: pre-fix BOTH of these are wrong, and if the
    # deletion assertion ran first the recorded failure would only ever prove
    # the deletion half. Ordered this way, the `bfa5104` run fails on the anchor
    # line, which is what this test is named for.
    assert state.get_project(conn, project_id)["last_indexed_commit"] == first_anchor
    assert second.files_deleted == 0


@requires_git
@pytest.mark.asyncio
async def test_a_degraded_gitignore_that_holds_nothing_back_still_anchors(
    tmp_path, monkeypatch
):
    """OVER-CORRECTION GUARD against the obvious wrong fix.

    Gating the anchor on the FAULT rather than on what it actually held back
    would latch any project with a permanently unreadable .gitignore onto full
    walks forever — the "guard that can never be satisfied" shape ADR-55 was
    written to avoid. The test is on `screening_held`, so the common case
    (rules that exclude nothing currently indexed) still advances.
    """
    conn, store, embedder = make_env(tmp_path)
    root = _tree(tmp_path)
    (root / "pkg" / ".gitignore").write_text("*.tmp\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")

    project_id, _ = await _run(conn, store, embedder, root)
    first_anchor = state.get_project(conn, project_id)["last_indexed_commit"]

    (root / "top.py").write_text("top = 9\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "second")
    second_head = git_head(root)
    assert second_head != first_anchor

    _fail_method_on(monkeypatch, "read_text", root / "pkg" / ".gitignore")
    _, second = await _run(conn, store, embedder, root)

    # Deliberately the ONLY assertion. An earlier draft also checked
    # `discovery_degraded`, which does not exist on the pre-fix `IndexResult` —
    # so against `bfa5104` this failed on AttributeError and never reached the
    # anchor, making the "passes both ways" label untrue and the guard worth
    # nothing (lesson #14). That the fault is recorded is already pinned by the
    # three site tests above; this one exists solely to prove the anchor is
    # gated on what was held back, not on the fault.
    assert state.get_project(conn, project_id)["last_indexed_commit"] == second_head


# --- 8, 9: the collateral callers, and the recording key ---------------------


@pytest.mark.asyncio
async def test_structural_search_survives_an_unstattable_gitignore(
    tmp_path, monkeypatch
):
    """`api/routes.py` catches only StructuralSearchError, so this was a 500."""
    conn, store, embedder = make_env(tmp_path)
    root = _tree(tmp_path)
    project_id, _ = await _run(conn, store, embedder, root)

    _fail_method_on(monkeypatch, "is_file", root / "pkg" / ".gitignore")
    result = await structural_search(conn, project_id, "$X = $Y", "python")

    # Pre-fix the PermissionError escaped `discover_files` and propagated
    # straight out of the route, which catches only StructuralSearchError.
    assert result["scanned_files"] >= 1
    # The degraded directory is still scanned, not skipped.
    assert "pkg/mod.py" in {m["file_path"] for m in result["matches"]}


@pytest.mark.asyncio
async def test_screening_row_cannot_overwrite_a_real_file_error(tmp_path, monkeypatch):
    """The r-without-x case produces BOTH rows for the same .gitignore.

    `Path.is_file()` goes through `stat()`, so a directory missing execute
    permission fails the screening test AND every per-file stat in it —
    including the stat of the .gitignore itself, which is enumerated as an
    ordinary file entry. `record_file_errors` is INSERT OR REPLACE on
    (run_id, path), and round 6 of PR #24 already lost an operator's real errno
    to exactly this collision. The `<screening>:` prefix is what keeps the two
    keys apart; a real rel path can never begin with `<`.
    """
    conn, store, embedder = make_env(tmp_path)
    root = _tree(tmp_path)
    (root / "pkg" / ".gitignore").write_text("*.tmp\n")

    _fail_method_on(monkeypatch, "stat", root / "pkg" / ".gitignore")
    _, result = await _run(conn, store, embedder, root)

    errors = {
        row["path"]: row["error"] for row in state.list_file_errors(conn, result.run_id)
    }
    assert "pkg/.gitignore" in errors, "the real per-file error was overwritten"
    assert "<screening>:pkg" in errors, "the screening fault was overwritten"
    assert "discovery failed" in errors["pkg/.gitignore"]
    assert "screening degraded" in errors["<screening>:pkg"]
    # The file-level error is a real failure and counts; the screening one is
    # not and does not.
    assert result.files_failed == 1
    assert result.discovery_degraded == 1
