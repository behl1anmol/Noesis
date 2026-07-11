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
from dataclasses import dataclass
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
    root: str | Path, config: DiscoveryConfig | None = None
) -> list[str]:
    """Return sorted, POSIX-style relative paths of indexable files under *root*."""
    cfg = config or DiscoveryConfig()
    root_path = Path(root).resolve()
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
        root_path, topdown=True, followlinks=cfg.follow_symlinks
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
                if full.stat().st_size > cfg.max_file_bytes:
                    continue
                if _is_binary(full):
                    continue
            except OSError:
                continue
            results.append(rel)

    return sorted(results)
