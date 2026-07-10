"""M7 tests: git candidate-set fast path (§3.2).

Unit coverage of the -z parsers, a per-condition fallback matrix (every
uncertainty → full hash-walk, silently), and the positive fast path down to
the exit criterion: change 3 files in a committed repo and exactly 3 files
get hashed on the next run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.core import gitfast, hashdiff, state
from noesis.core.discovery import discover_files
from noesis.core.embedder import FakeEmbedder
from noesis.core.gitfast import (
    CandidatePathSet,
    _parse_diff_z,
    _parse_status_z,
    _strip_prefix,
    compute_candidates,
    resolve_head,
)
from noesis.core.hashdiff import partition
from noesis.core.indexer import execute_run, index_project, prepare_run
from noesis.core.vectorstore import VectorStore

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not on PATH"
)


# --- helpers -----------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@test",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(root),
            *args,
        ),
        check=True,
        capture_output=True,
    )


def git_head(root: Path) -> str:
    proc = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
    )
    return proc.stdout.decode().strip()


FILES = {
    "alpha.py": "def alpha():\n    return 1\n",
    "beta.py": "def beta():\n    return 2\n",
    "gamma.py": "def gamma():\n    return 3\n",
}


def build_files(root: Path, files: dict[str, str] = FILES) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def build_git_repo(root: Path, files: dict[str, str] = FILES) -> None:
    build_files(root, files)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")


def make_env(tmp_path: Path):
    """conn + in-memory store + FakeEmbedder — same offline rig as test_api."""
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return conn, store, embedder


def anchor_of(conn, project_id: str) -> str | None:
    return state.get_project(conn, project_id)["last_indexed_commit"]


class ExplodingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")


# --- A. unit: parsers (no git needed) ----------------------------------------


def test_parse_diff_z_status_entries() -> None:
    """§3.2 rule 2: M/A/D name-status entries each yield one path."""
    out = b"M\x00alpha.py\x00A\x00fresh.py\x00D\x00gone.py\x00"
    assert _parse_diff_z(out) == {"alpha.py", "fresh.py", "gone.py"}


def test_parse_diff_z_rename_yields_both_paths() -> None:
    """§3.2 robustness: an R100 entry contributes old AND new path."""
    out = b"R100\x00old.py\x00new.py\x00M\x00other.py\x00"
    assert _parse_diff_z(out) == {"old.py", "new.py", "other.py"}


def test_parse_diff_z_trailing_and_empty_tokens_harmless() -> None:
    """§3.2 robustness: trailing NULs / empty tokens never crash or add paths."""
    assert _parse_diff_z(b"M\x00alpha.py\x00\x00\x00") == {"alpha.py"}
    assert _parse_diff_z(b"") == set()


def test_parse_status_z_entries() -> None:
    """§3.2 rule 2: staged, unstaged, untracked, space and unicode paths parse."""
    out = (
        b" M unstaged.py\x00"
        b"?? untracked dir/new file.py\x00"
        b"A  staged.py\x00"
        b" M h\xc3\xa9llo.py\x00"
    )
    assert _parse_status_z(out) == {
        "unstaged.py",
        "untracked dir/new file.py",
        "staged.py",
        "héllo.py",
    }


def test_parse_status_z_rename_consumes_original_path_token() -> None:
    """§3.2 robustness: an R entry's extra original-path token is a candidate too."""
    out = b"R  new.py\x00old.py\x00 M other.py\x00"
    assert _parse_status_z(out) == {"new.py", "old.py", "other.py"}


def test_parse_status_z_empty_output() -> None:
    """§3.2: clean worktree (empty porcelain output) → empty candidate set."""
    assert _parse_status_z(b"") == set()


def test_strip_prefix_empty_passthrough() -> None:
    """§3.2: repo toplevel (empty --show-prefix) leaves paths untouched."""
    paths = {"a.py", "sub/b.py"}
    assert _strip_prefix(paths, "") == paths


def test_strip_prefix_strips_and_drops_outside_paths() -> None:
    """§3.2: subdir roots get project-relative paths; outside-prefix paths drop."""
    paths = {"sub/a.py", "sub/deep/b.py", "other/c.py", "top.py"}
    assert _strip_prefix(paths, "sub/") == {"a.py", "deep/b.py"}


def test_candidate_path_set_exact_match() -> None:
    """§3.2: an explicit file entry matches itself and nothing else."""
    cands = CandidatePathSet(["a.py", "sub/b.py"])
    assert "a.py" in cands
    assert "sub/b.py" in cands
    assert "c.py" not in cands


def test_candidate_path_set_directory_entry_matches_descendants() -> None:
    """§3.2: a 'dir/' entry (untracked nested repo) covers dir/inner/file.py."""
    cands = CandidatePathSet(["nested/"])
    assert "nested/file.py" in cands
    assert "nested/inner/file.py" in cands
    assert "nested" in cands  # trailing slash stripped on construction
    assert "nestedish/file.py" not in cands  # prefix must be a whole component


def test_candidate_path_set_bare_directory_entry_matches_descendants() -> None:
    """§3.2: a bare 'dir' entry (submodule gitlink) covers dir/file.py."""
    cands = CandidatePathSet(["submod"])
    assert "submod/file.py" in cands
    assert "submod" in cands
    assert "other/file.py" not in cands


def test_candidate_path_set_non_member_and_non_str() -> None:
    """§3.2: non-members and non-str objects are simply not contained."""
    cands = CandidatePathSet(["a.py"])
    assert "b.py" not in cands
    assert 42 not in cands
    assert None not in cands
    assert b"a.py" not in cands


def test_candidate_path_set_len_counts_explicit_entries() -> None:
    """§3.2: len() is the explicit entry count, not the matched universe."""
    cands = CandidatePathSet(["dir/", "a.py"])
    assert len(cands) == 2
    assert CandidatePathSet([]) == frozenset()
    # AbstractSet equality with a frozenset of the same explicit entries.
    assert CandidatePathSet(["a.py", "dir/"]) == frozenset({"a.py", "dir"})


# --- B. fallback matrix (each uncertainty → None → full walk) -----------------


@requires_git
async def test_fallback_not_a_git_repo(tmp_path: Path) -> None:
    """§3.2 rule 3: plain directory → fallback, index still fully correct."""
    repo = tmp_path / "repo"
    build_files(repo)
    conn, store, embedder = make_env(tmp_path)

    assert compute_candidates(repo, "deadbeef") is None
    assert resolve_head(repo) is None

    result = await index_project(conn, store, embedder, str(repo))
    assert result.fast_path_used is False
    assert result.candidate_count is None
    assert result.files_indexed == len(FILES)
    assert anchor_of(conn, result.project_id) is None  # no HEAD to record
    assert set(state.get_file_states(conn, result.project_id)) == set(FILES)
    conn.close()


@requires_git
async def test_fallback_first_index_has_no_anchor_but_records_head(
    tmp_path: Path,
) -> None:
    """§3.2 rule 4: first run of a git repo full-walks, then stores HEAD anchor."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)

    result = await index_project(conn, store, embedder, str(repo))
    assert result.fast_path_used is False
    assert result.candidate_count is None
    assert result.files_indexed == len(FILES)
    assert anchor_of(conn, result.project_id) == git_head(repo)
    conn.close()


@requires_git
async def test_fallback_anchor_unreachable_after_amend(tmp_path: Path) -> None:
    """§3.2 rule 3 / Risk 12: rewritten history (anchor not ancestor) → fallback."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / "alpha.py").write_text("def alpha():\n    return 100\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--amend", "-m", "rewritten")
    assert compute_candidates(repo, anchor) is None

    expected = partition(
        repo, discover_files(repo), state.get_file_states(conn, result1.project_id)
    )
    assert expected.changed == ("alpha.py",)

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == len(expected.new) + len(expected.changed) == 1
    stored = state.get_file_states(conn, result2.project_id)
    assert stored["alpha.py"] == expected.hashes["alpha.py"]
    # Successful full-walk run still advances the anchor to the new HEAD.
    assert anchor_of(conn, result2.project_id) == git_head(repo)
    conn.close()


@requires_git
async def test_fallback_corrupt_git_head(tmp_path: Path) -> None:
    """§3.2 rule 3: git exits non-zero (corrupt .git/HEAD) → fallback, run succeeds."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / ".git" / "HEAD").write_text("garbage not a ref\n")
    assert resolve_head(repo) is None
    assert compute_candidates(repo, anchor) is None

    (repo / "beta.py").write_text("def beta():\n    return 200\n")
    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == 1
    # HEAD unresolvable → the (stale but reachable-at-the-time) anchor stays.
    assert anchor_of(conn, result2.project_id) == anchor
    conn.close()


@requires_git
async def test_fallback_mid_merge(tmp_path: Path) -> None:
    """§3.2 rule 3: MERGE_HEAD present (in-progress merge) → fallback."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / ".git" / "MERGE_HEAD").write_text("a" * 40 + "\n")
    assert compute_candidates(repo, anchor) is None

    (repo / "alpha.py").write_text("def alpha():\n    return 11\n")
    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == 1
    conn.close()


@requires_git
async def test_fallback_mid_rebase(tmp_path: Path) -> None:
    """§3.2 rule 3: rebase-merge dir present (in-progress rebase) → fallback."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / ".git" / "rebase-merge").mkdir()
    assert compute_candidates(repo, anchor) is None

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == 0  # nothing changed; full walk agrees
    conn.close()


@requires_git
async def test_fallback_detached_head(tmp_path: Path) -> None:
    """§3.2 rule 3: detached HEAD → fallback, changes still detected by hash."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    git(repo, "checkout", "--detach", "-q")
    assert compute_candidates(repo, anchor) is None

    (repo / "gamma.py").write_text("def gamma():\n    return 33\n")
    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == 1
    stored = state.get_file_states(conn, result2.project_id)
    assert stored["gamma.py"] == hashdiff.hash_file(repo / "gamma.py")
    conn.close()


@requires_git
async def test_fast_path_disabled_by_config(tmp_path: Path) -> None:
    """§3.7 [git] fast_path=false: no candidates AND anchor never touched."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)
    assert anchor == git_head(repo)

    (repo / "alpha.py").write_text("def alpha():\n    return 12\n")
    git(repo, "commit", "-q", "-a", "-m", "change")
    assert git_head(repo) != anchor

    project_id, run_id = prepare_run(conn, embedder, str(repo))
    result2 = await execute_run(
        conn, store, embedder, str(repo), project_id, run_id, git_fast_path=False
    )
    assert result2.fast_path_used is False
    assert result2.candidate_count is None
    assert result2.files_indexed == 1
    # Disabled means HEAD is never resolved — the old anchor must remain.
    assert anchor_of(conn, project_id) == anchor
    conn.close()


@requires_git
async def test_fallback_git_binary_absent(tmp_path: Path, monkeypatch) -> None:
    """§3.2 rule 3: git spawn raising FileNotFoundError → None → full walk."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(gitfast.subprocess, "run", no_git)
    assert resolve_head(repo) is None
    assert compute_candidates(repo, anchor) is None

    (repo / "beta.py").write_text("def beta():\n    return 22\n")
    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is False
    assert result2.files_indexed == 1
    assert anchor_of(conn, result2.project_id) == anchor  # HEAD unresolvable
    conn.close()


# --- C. positive fast path ----------------------------------------------------


@requires_git
async def test_exit_criterion_three_changed_files_three_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    """M7 exit criterion / §3.2 rule 1: 3 changed files → exactly 3 hash calls."""
    repo = tmp_path / "repo"
    files = {f"mod_{i}.py": f"def f{i}():\n    return {i}\n" for i in range(10)}
    build_git_repo(repo, files)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    assert result1.files_indexed == 10

    (repo / "mod_0.py").write_text("def f0():\n    return 100\n")
    (repo / "mod_1.py").write_text("def f1():\n    return 101\n")
    git(repo, "commit", "-q", "-a", "-m", "modify two")
    (repo / "brand_new.py").write_text("def fresh():\n    return -1\n")

    calls: list[Path] = []
    real_hash = hashdiff.hash_file

    def counting_hash(path):
        calls.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(hashdiff, "hash_file", counting_hash)

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert result2.candidate_count == 3
    assert len(calls) == 3
    assert {p.name for p in calls} == {"mod_0.py", "mod_1.py", "brand_new.py"}
    assert result2.files_indexed == 3  # 2 changed + 1 new
    assert result2.files_deleted == 0

    run_row = state.get_latest_run(conn, result2.project_id)
    assert run_row["id"] == result2.run_id
    assert run_row["fast_path_used"] == 1
    assert run_row["candidate_count"] == 3
    conn.close()


@requires_git
async def test_uncommitted_changes_are_candidates(tmp_path: Path) -> None:
    """§3.2 rule 2: staged, unstaged, and untracked changes all become candidates."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / "alpha.py").write_text("def alpha():\n    return 111\n")  # staged mod
    git(repo, "add", "alpha.py")
    (repo / "beta.py").write_text("def beta():\n    return 222\n")  # unstaged mod
    (repo / "delta.py").write_text("def delta():\n    return 4\n")  # untracked

    cands = compute_candidates(repo, anchor)
    assert cands is not None
    assert {"alpha.py", "beta.py", "delta.py"} <= cands.candidates

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert result2.files_indexed == 3
    stored = state.get_file_states(conn, result2.project_id)
    assert stored["alpha.py"] == hashdiff.hash_file(repo / "alpha.py")
    assert stored["beta.py"] == hashdiff.hash_file(repo / "beta.py")
    assert "delta.py" in stored
    conn.close()


@requires_git
async def test_deletions_committed_and_uncommitted(tmp_path: Path) -> None:
    """§3.2: git-rm'd and plainly removed files both leave state on a fast run."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))

    git(repo, "rm", "-q", "alpha.py")
    git(repo, "commit", "-q", "-m", "drop alpha")
    os.remove(repo / "beta.py")

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert result2.files_deleted == 2
    assert result2.files_indexed == 0
    stored = state.get_file_states(conn, result2.project_id)
    assert "alpha.py" not in stored
    assert "beta.py" not in stored
    assert "gamma.py" in stored
    conn.close()


@requires_git
async def test_rename_indexes_new_and_deletes_old(tmp_path: Path) -> None:
    """§3.2 (--no-renames): git mv surfaces as new path indexed + old deleted."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    await index_project(conn, store, embedder, str(repo))

    git(repo, "mv", "alpha.py", "renamed.py")
    git(repo, "commit", "-q", "-m", "rename")

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert result2.files_indexed == 1
    assert result2.files_deleted == 1
    stored = state.get_file_states(conn, result2.project_id)
    assert "renamed.py" in stored
    assert "alpha.py" not in stored
    conn.close()


@requires_git
async def test_project_root_in_repo_subdirectory(tmp_path: Path) -> None:
    """§3.2 (--show-prefix): subdir roots get sub-relative candidates only."""
    repo = tmp_path / "repo"
    sub = repo / "sub"
    build_files(sub, {"inner.py": "def inner():\n    return 1\n"})
    (repo / "outer.py").write_text("def outer():\n    return 0\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(sub))
    assert result1.files_indexed == 1
    anchor = anchor_of(conn, result1.project_id)
    assert anchor == git_head(repo)

    (sub / "inner.py").write_text("def inner():\n    return 999\n")
    (repo / "outer.py").write_text("def outer():\n    return 888\n")
    git(repo, "commit", "-q", "-a", "-m", "touch both")

    cands = compute_candidates(sub, anchor)
    assert cands is not None
    assert cands.candidates == frozenset({"inner.py"})

    result2 = await index_project(conn, store, embedder, str(sub))
    assert result2.fast_path_used is True
    assert result2.candidate_count == 1
    assert result2.files_indexed == 1
    stored = state.get_file_states(conn, result2.project_id)
    assert set(stored) == {"inner.py"}
    conn.close()


@requires_git
async def test_nested_git_repo_change_caught_on_fast_path(tmp_path: Path) -> None:
    """§3.2: parent status collapses a nested repo to 'nested/'; ancestor
    matching still re-hashes its changed files on a fast-path run."""
    repo = tmp_path / "repo"
    build_git_repo(repo)  # outer files committed; nested does not exist yet
    nested = repo / "nested"
    build_files(nested, {"file.py": "def inner():\n    return 1\n"})
    git(nested, "init", "-q")
    git(nested, "add", "-A")
    git(nested, "commit", "-q", "-m", "nested init")

    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    assert result1.fast_path_used is False  # no anchor yet
    stored = state.get_file_states(conn, result1.project_id)
    assert "nested/file.py" in stored  # discovery descends into the nested repo
    anchor = anchor_of(conn, result1.project_id)
    assert anchor == git_head(repo)

    # Change ONLY inside the nested repo — the parent's status shows just
    # "?? nested/", never the file itself.
    (nested / "file.py").write_text("def inner():\n    return 999\n")

    cands = compute_candidates(repo, anchor)
    assert cands is not None
    assert "nested/file.py" in cands.candidates

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert result2.files_indexed == 1
    stored = state.get_file_states(conn, result2.project_id)
    assert stored["nested/file.py"] == hashdiff.hash_file(nested / "file.py")
    conn.close()


@requires_git
async def test_anchor_advances_after_successful_fast_run(tmp_path: Path) -> None:
    """§3.2 rule 4: a successful fast-path run moves the anchor to current HEAD."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    old_anchor = anchor_of(conn, result1.project_id)

    (repo / "alpha.py").write_text("def alpha():\n    return 7\n")
    git(repo, "commit", "-q", "-a", "-m", "advance")
    new_head = git_head(repo)
    assert new_head != old_anchor

    result2 = await index_project(conn, store, embedder, str(repo))
    assert result2.fast_path_used is True
    assert anchor_of(conn, result2.project_id) == new_head
    conn.close()


@requires_git
async def test_anchor_not_updated_on_failed_run(tmp_path: Path) -> None:
    """§3.2 rule 4: a failed run must not advance last_indexed_commit."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / "alpha.py").write_text("def alpha():\n    return 13\n")
    git(repo, "commit", "-q", "-a", "-m", "will fail to index")
    assert git_head(repo) != anchor

    # Since ADR-41 (M8), a per-file embed failure is contained rather than
    # propagated; with every changed file failing, the run is marked
    # 'failed' without raising. The property under test is unchanged:
    # a failed run must not move the anchor.
    result2 = await index_project(conn, store, ExplodingEmbedder(dim=8), str(repo))
    assert result2.files_failed == 1

    run_row = state.get_latest_run(conn, result1.project_id)
    assert run_row["status"] == "failed"
    assert anchor_of(conn, result1.project_id) == anchor
    conn.close()


@requires_git
async def test_partition_equivalence_full_vs_candidates(tmp_path: Path) -> None:
    """§3.2 rule 1: candidate-narrowed partition equals the full walk, field-by-field."""
    repo = tmp_path / "repo"
    build_git_repo(repo)
    conn, store, embedder = make_env(tmp_path)
    result1 = await index_project(conn, store, embedder, str(repo))
    anchor = anchor_of(conn, result1.project_id)

    (repo / "alpha.py").write_text("def alpha():\n    return 77\n")
    git(repo, "commit", "-q", "-a", "-m", "modify alpha")
    (repo / "delta.py").write_text("def delta():\n    return 4\n")  # untracked new
    os.remove(repo / "beta.py")  # uncommitted delete

    cands = compute_candidates(repo, anchor)
    assert cands is not None

    discovered = discover_files(repo)
    stored = state.get_file_states(conn, result1.project_id)
    full = partition(repo, discovered, stored)
    fast = partition(repo, discovered, stored, candidates=cands.candidates)

    assert fast.new == full.new == ("delta.py",)
    assert fast.changed == full.changed == ("alpha.py",)
    assert fast.unchanged == full.unchanged
    assert fast.deleted == full.deleted == ("beta.py",)
    assert dict(fast.hashes) == dict(full.hashes)
    assert fast == full
    conn.close()
