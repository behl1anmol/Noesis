"""Git candidate-set fast path (§3.2) — subprocess git CLI (ADR-23).

The fast path may only *shrink* the set of files that get hashed; it never
decides changed/unchanged on its own authority (§3.2 rule 1) — that stays
with the SHA-256 comparison in :mod:`noesis.core.hashdiff`. Any uncertainty
— no repo, git binary absent, in-progress merge/rebase, detached HEAD,
missing or unreachable anchor commit, non-zero git exit, timeout — returns
``None`` and the caller falls back to the full hash-walk, silently for the
API caller and logged for the operator (rule 3).

Subprocess against the system git CLI is deliberate (rule 5 / ADR-23):
zero Python deps, license-clean, and the ~10ms spawn cost is noise next to
embedding. pygit2 remains the documented upgrade if profiling ever says
otherwise.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_S = 30.0

# Presence of any of these inside the git dir means an in-progress
# operation (merge/rebase/cherry-pick/revert/bisect) — the worktree is in
# a transient state, so fall back (§3.2 rule 3, ".git/MERGE_HEAD etc.").
_IN_PROGRESS_MARKERS = (
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
    "rebase-apply",
    "rebase-merge",
)


class CandidatePathSet(AbstractSet[str]):
    """Candidate paths where an entry may denote a whole directory.

    Git collapses some changes to a single directory entry: an untracked
    nested repository is one ``dir/`` line even under ``-uall``, and a
    submodule change is one gitlink path with no slash. Discovery *does*
    descend into those directories, so a plain path-equality test would
    carry their changed files forward as unchanged. Membership here also
    matches any ancestor directory of the queried path — this only widens
    the candidate set (more files hashed, never fewer), so rule 1 holds.
    """

    __slots__ = ("_entries",)

    def __init__(self, paths: Iterable[str]) -> None:
        self._entries = frozenset(p.rstrip("/") for p in paths if p.rstrip("/"))

    def __contains__(self, rel: object) -> bool:
        if not isinstance(rel, str):
            return False
        if rel in self._entries:
            return True
        i = rel.rfind("/")
        while i > 0:
            if rel[:i] in self._entries:
                return True
            i = rel.rfind("/", 0, i)
        return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"CandidatePathSet({sorted(self._entries)!r})"


@dataclass(frozen=True)
class GitCandidates:
    """Candidate changed set for one run.

    ``candidates`` holds project-root-relative POSIX paths (same shape as
    discovery output); ``head_commit`` is the HEAD sha the diff was taken
    against — the anchor to store once the run completes (rule 4).
    """

    candidates: CandidatePathSet
    head_commit: str


def _git(root: str | Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    """Run git in *root*; None on spawn failure or timeout (both → fallback)."""
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args),
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("git %s did not run (%s)", args[0] if args else "", exc)
        return None


def resolve_head(root: str | Path) -> str | None:
    """HEAD commit sha, or None (not a repo, no commits yet, git absent).

    Called even on full-walk runs of a git worktree so the *next* run has
    an anchor to fast-path from.
    """
    proc = _git(root, "rev-parse", "HEAD")
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.decode("ascii", "replace").strip() or None


def compute_candidates(root: str | Path, anchor: str) -> GitCandidates | None:
    """Candidate changed set since *anchor*, or None → full hash-walk.

    The set is ``git diff --name-status anchor..HEAD`` ∪
    ``git status --porcelain`` (staged + unstaged + untracked), per §3.2
    rule 2. Deleted paths are included; they are not on disk, so discovery
    never yields them and the partition marks them deleted exactly as the
    full walk would.
    """
    proc = _git(root, "rev-parse", "--absolute-git-dir", "--show-prefix")
    if proc is None or proc.returncode != 0:
        return _fallback("not a git repository")
    lines = os.fsdecode(proc.stdout).splitlines()
    if not lines:
        return _fallback("unparseable rev-parse output")
    git_dir = Path(lines[0])
    # --show-prefix is empty at the repo toplevel; rev-parse still emits the
    # (empty) line, but be tolerant of it being absent.
    prefix = lines[1] if len(lines) > 1 else ""

    for marker in _IN_PROGRESS_MARKERS:
        if (git_dir / marker).exists():
            return _fallback(f"in-progress git operation ({marker} present)")

    proc = _git(root, "symbolic-ref", "-q", "HEAD")
    if proc is None or proc.returncode != 0:
        return _fallback("detached HEAD")

    head = resolve_head(root)
    if head is None:
        return _fallback("cannot resolve HEAD")

    # Anchor must be an ancestor of HEAD — merely existing in the object
    # store is not enough (a rebased/force-pushed-away commit lingers until
    # gc; diffing against it would miss the rewritten history — Risk 12).
    # Exit 1 = not an ancestor, 128 = unknown object; both → fallback.
    proc = _git(root, "merge-base", "--is-ancestor", anchor, "HEAD")
    if proc is None or proc.returncode != 0:
        return _fallback("anchor commit missing or unreachable")

    # --no-renames: renames surface as D old + A new, so both paths become
    # candidates with a trivial parser, independent of user diff.renames
    # config. -z: NUL-separated raw paths — immune to quoting/escaping.
    diff = _git(root, "diff", "--name-status", "-z", "--no-renames", anchor, "HEAD")
    if diff is None or diff.returncode != 0:
        return _fallback("git diff failed")

    # --untracked-files=all lists files inside untracked directories
    # individually (default collapses to "dir/", which is not a file path).
    status = _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"
    )
    if status is None or status.returncode != 0:
        return _fallback("git status failed")

    paths = _parse_diff_z(diff.stdout) | _parse_status_z(status.stdout)
    return GitCandidates(
        candidates=CandidatePathSet(_strip_prefix(paths, prefix)), head_commit=head
    )


def _fallback(reason: str) -> None:
    logger.info("git fast-path unavailable: %s — falling back to full hash-walk", reason)
    return None


def _parse_diff_z(out: bytes) -> set[str]:
    """Paths from ``diff --name-status -z``: STATUS NUL PATH NUL ...

    --no-renames is always passed, but stay robust: R/C statuses carry two
    path tokens (old, new) and both are candidates.
    """
    tokens = out.split(b"\0")
    paths: set[str] = set()
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue
        npaths = 2 if status[:1] in (b"R", b"C") else 1
        for j in range(1, npaths + 1):
            if i + j < len(tokens) and tokens[i + j]:
                paths.add(os.fsdecode(tokens[i + j]))
        i += 1 + npaths
    return paths


def _parse_status_z(out: bytes) -> set[str]:
    """Paths from ``status --porcelain=v1 -z``: "XY path" NUL entries;
    rename entries are followed by one extra NUL-terminated original path."""
    entries = out.split(b"\0")
    paths: set[str] = set()
    i = 0
    while i < len(entries):
        entry = entries[i]
        if len(entry) >= 4:  # minimum "XY p"
            paths.add(os.fsdecode(entry[3:]))
            if entry[:1] in (b"R", b"C"):
                i += 1
                if i < len(entries) and entries[i]:
                    paths.add(os.fsdecode(entries[i]))
        i += 1
    return paths


def _strip_prefix(paths: set[str], prefix: str) -> set[str]:
    """Translate repo-root-relative paths to project-root-relative ones.

    When the project root is a subdirectory of the repo (``--show-prefix``
    non-empty), paths outside the project root are dropped — they cannot be
    discovery candidates.
    """
    if not prefix:
        return paths
    return {p[len(prefix) :] for p in paths if p.startswith(prefix)}
