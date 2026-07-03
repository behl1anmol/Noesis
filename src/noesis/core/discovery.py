"""File discovery: walk a project tree and yield indexable files.

Filters, in order: excluded directories, .gitignore (git semantics, nested
files, negation), the secret skip-list, symlinks, size cap, binary sniff.
The secret skip-list is defense-in-depth on top of .gitignore — a secret
file is skipped even when no .gitignore mentions it, so it can never enter
the index (a retrievable surface) or M5 structural-search results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

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
    ".aws/**",
    ".ssh/**",
)

_SECRET_SPEC = GitIgnoreSpec.from_lines(SECRET_SKIP_PATTERNS)

_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class DiscoveryConfig:
    max_file_bytes: int = 1_048_576
    follow_symlinks: bool = False


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


def discover_files(root: str | Path, config: DiscoveryConfig | None = None) -> list[str]:
    """Return sorted, POSIX-style relative paths of indexable files under *root*."""
    cfg = config or DiscoveryConfig()
    root_path = Path(root).resolve()
    ignores = _IgnoreStack()
    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(
        root_path, topdown=True, followlinks=cfg.follow_symlinks
    ):
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
                if full.stat().st_size > cfg.max_file_bytes:
                    continue
                if _is_binary(full):
                    continue
            except OSError:
                continue
            results.append(rel)

    return sorted(results)
