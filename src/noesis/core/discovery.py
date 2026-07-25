"""File discovery: walk a project tree and yield indexable files.

Filters, in order: excluded directories, .gitignore (git semantics, nested
files, negation), the secret skip-list, the generated-lockfile skip-list,
symlinks, size cap, binary sniff.
The secret skip-list is defense-in-depth on top of .gitignore — a secret
file is skipped even when no .gitignore mentions it, so it can never enter
the index (a retrievable surface) or M5 structural-search results.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from .languages import detect_language

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "target",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        ".idea",
        ".vscode",
    }
)

SECRET_SKIP_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "*.ppk",
    "credentials*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*.tfvars",
    "secrets.*",
    "*.secret",
    "**/.aws/**",
    "**/.ssh/**",
)

_SECRET_SPEC = GitIgnoreSpec.from_lines(SECRET_SKIP_PATTERNS)

# Generated lockfiles: committed (so not gitignored), text, often huge, and
# pure noise for retrieval — indexing one can dominate a small repo's embed
# cost (decision row 31). Same skip-list pattern as secrets.
GENERATED_SKIP_PATTERNS: tuple[str, ...] = (
    "uv.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
    "packages.lock.json",
    "flake.lock",
)

_GENERATED_SPEC = GitIgnoreSpec.from_lines(GENERATED_SKIP_PATTERNS)

_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class DiscoveryConfig:
    max_file_bytes: int = 1_048_576
    follow_symlinks: bool = False
    # ADR-42 per-project scope. ``include_languages`` None = index every
    # file (today's behavior); a set keeps only files whose detected
    # language is in it — files with no detected language are dropped when
    # a filter is active, since the user asked for specific languages.
    # ``extra_ignore_patterns`` are additional gitignore-style globs applied
    # like the secret skip-list, anchored at the project root.
    include_languages: frozenset[str] | None = None
    extra_ignore_patterns: tuple[str, ...] = ()


@dataclass
class DiscoveryErrors:
    """Non-fatal discovery failures, (root-relative path, message) pairs.

    ``files``: a file that exists but could not be screened (stat/read
    error) — it is excluded from the results, but "absent from discovery"
    must not be read as "deleted from disk" (same discrimination as H7).
    ``dirs``: a directory whose scandir failed mid-walk — its whole subtree
    went unseen, so deletion evidence from this walk is untrustworthy.
    """

    files: list[tuple[str, str]] = field(default_factory=list)
    dirs: list[tuple[str, str]] = field(default_factory=list)


def is_secret_path(rel_posix: str) -> bool:
    """True if a project-relative POSIX path matches the secret skip-list."""
    return bool(_SECRET_SPEC.match_file(rel_posix))


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\x00" in f.read(_BINARY_SNIFF_BYTES)


class _IgnoreStack:
    """Nested .gitignore evaluation with git's last-match-wins semantics.

    Each spec is anchored at the directory holding its .gitignore; deeper
    specs are consulted after shallower ones so the deepest matching
    pattern (including negations) decides.
    """

    def __init__(self) -> None:
        self._specs: list[tuple[str, GitIgnoreSpec]] = []

    def push(self, base_rel_posix: str, gitignore: Path) -> None:
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        self._specs.append((base_rel_posix, GitIgnoreSpec.from_lines(lines)))

    def ignored(self, rel_posix: str, *, is_dir: bool = False) -> bool:
        candidate = rel_posix + "/" if is_dir else rel_posix
        decision = False
        for base, spec in self._specs:
            if base == "":
                sub = candidate
            elif candidate.startswith(base + "/"):
                sub = candidate[len(base) + 1 :]
            else:
                continue
            result = spec.check_file(sub)
            if result.include is not None:
                decision = result.include
        return decision


def discover_files(
    root: str | Path,
    config: DiscoveryConfig | None = None,
    *,
    errors: DiscoveryErrors | None = None,
) -> list[str]:
    """Return sorted, POSIX-style relative paths of indexable files under *root*.

    When *errors* is given, transient stat/read/scandir failures are recorded
    there instead of vanishing silently — callers doing change detection need
    them to tell "file errored" from "file deleted". ``errors=None`` keeps the
    historical silent-skip behavior."""
    cfg = config or DiscoveryConfig()
    root_path = Path(root).resolve()

    def _walk_error(exc: OSError) -> None:
        # A directory deleted mid-walk is a genuine deletion (same
        # discrimination as H7's file case); any other scandir failure means
        # part of the tree went unseen and must be surfaced when asked.
        if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
            return
        if errors is None:
            return
        rel = "<root>"
        if exc.filename:
            try:
                rel = PurePosixPath(
                    Path(exc.filename).relative_to(root_path)
                ).as_posix()
            except ValueError:
                rel = str(exc.filename)
            if rel == ".":
                rel = "<root>"
        errors.dirs.append((rel, str(exc) or type(exc).__name__))

    ignores = _IgnoreStack()
    extra_spec = (
        GitIgnoreSpec.from_lines(cfg.extra_ignore_patterns)
        if cfg.extra_ignore_patterns
        else None
    )
    results: list[str] = []
    # When following symlinks, os.walk has no cycle/duplicate guard. Track
    # directory identity (st_dev, st_ino) and prune any dir already walked so
    # a self-referencing link cannot loop forever and a link into an
    # already-walked subtree cannot index files twice.
    visited: set[tuple[int, int]] = set()

    for dirpath, dirnames, filenames in os.walk(
        root_path, topdown=True, followlinks=cfg.follow_symlinks, onerror=_walk_error
    ):
        if cfg.follow_symlinks:
            try:
                st = os.stat(dirpath)
                visited.add((st.st_dev, st.st_ino))
            except OSError:
                pass
        dir_rel = PurePosixPath(Path(dirpath).relative_to(root_path)).as_posix()
        if dir_rel == ".":
            dir_rel = ""

        gitignore = Path(dirpath) / ".gitignore"
        if gitignore.is_file():
            ignores.push(dir_rel, gitignore)

        kept_dirs = []
        for d in sorted(dirnames):
            if d in EXCLUDED_DIRS:
                continue
            child_rel = f"{dir_rel}/{d}" if dir_rel else d
            if ignores.ignored(child_rel, is_dir=True):
                continue
            if not cfg.follow_symlinks and (Path(dirpath) / d).is_symlink():
                continue
            if cfg.follow_symlinks:
                try:
                    cst = os.stat(Path(dirpath) / d)
                except OSError:
                    kept_dirs.append(d)
                    continue
                if (cst.st_dev, cst.st_ino) in visited:
                    continue
                visited.add((cst.st_dev, cst.st_ino))
            kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for name in filenames:
            rel = f"{dir_rel}/{name}" if dir_rel else name
            full = Path(dirpath) / name
            try:
                if not cfg.follow_symlinks and full.is_symlink():
                    continue
                if ignores.ignored(rel):
                    continue
                if is_secret_path(rel):
                    continue
                if _GENERATED_SPEC.match_file(rel):
                    continue
                if extra_spec is not None and extra_spec.match_file(rel):
                    continue
                if (
                    cfg.include_languages is not None
                    and detect_language(rel) not in cfg.include_languages
                ):
                    continue
                st = full.stat()
                # Non-regular files are skipped before anything opens them.
                # A FIFO passes the size gate (st_size is 0) and then
                # `_is_binary`'s open() blocks until some process opens the
                # write end — forever, in practice. That hangs discovery
                # inside its worker thread, so the run never finishes and
                # nothing raises for the OSError handler below to catch.
                # Device and socket nodes are skipped here too, explicitly
                # rather than by accident.
                if not stat.S_ISREG(st.st_mode):
                    continue
                if st.st_size > cfg.max_file_bytes:
                    continue
                if _is_binary(full):
                    continue
            except (FileNotFoundError, NotADirectoryError):
                continue  # genuinely vanished mid-walk — a real deletion (H7)
            except OSError as exc:
                # Still present but unscreened (EACCES, EIO, ESTALE): never
                # index it on unverified data, but record the failure so
                # callers don't mistake its absence for a deletion.
                if errors is not None:
                    errors.files.append((rel, str(exc) or type(exc).__name__))
                continue
            results.append(rel)

    return sorted(results)
