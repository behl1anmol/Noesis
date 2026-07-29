"""File discovery: walk a project tree and yield indexable files.

Filters, in order: excluded directories, .gitignore (git semantics, nested
files, negation), the secret skip-list, the generated-lockfile skip-list,
symlinks, size cap, binary sniff.
The secret skip-list is defense-in-depth on top of .gitignore — a secret
file is skipped even when no .gitignore mentions it, so it can never enter
the index (a retrievable surface) or M5 structural-search results.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from .languages import detect_language

logger = logging.getLogger(__name__)

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

    ``entries_seen`` counts every file entry the walk enumerated, *before*
    any filter runs. It is not an error, but it lives here because only the
    caller that cares about the errors cares about it: an empty result with
    ``entries_seen == 0`` means the walk found nothing at all (an unmounted
    or emptied root), while an empty result with ``entries_seen > 0`` means
    the tree is readable and populated and the filters — an ADR-42 scope
    narrowing, .gitignore, the skip-lists — removed everything. Those two
    are indistinguishable from the returned list alone, and change
    detection must not treat the first as deletion.
    """

    files: list[tuple[str, str]] = field(default_factory=list)
    dirs: list[tuple[str, str]] = field(default_factory=list)
    entries_seen: int = 0


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
        except OSError as exc:
            # The second declared exception to "no filesystem fault vanishes
            # without a trace" (ADR-51), and the one whose consequence points
            # the opposite way from every other error in this module. Losing a
            # .gitignore does not risk over-deleting — it risks
            # UNDER-IGNORING: the rules that should have excluded this
            # directory's files never load, so files the user asked to keep
            # out can enter the index, which is a retrievable surface.
            #
            # Skipping is still what happens, for now, because the
            # alternative is failing the whole walk over one unreadable
            # ignore file. Which of those is right is a taxonomy decision
            # tracked in issue #26 alongside the sibling `is_file()` call
            # three lines below; this only stops the failure being invisible
            # while that is decided (PR #24 round-8 review).
            #
            # Bounded, and worth knowing: the secret and generated-file
            # skip-lists are applied independently of this stack, so a lost
            # .gitignore is never a secret-exposure path.
            logger.debug(
                "discovery: could not read %s (%s) — its ignore rules are "
                "not applied, so this directory may be under-ignored",
                gitignore,
                exc,
            )
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


def _cycle_guard_identity(path: str | Path) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` for the ``follow_symlinks`` cycle guard, or None.

    A failure here is deliberately NOT recorded in ``DiscoveryErrors.dirs``.
    ``os.walk`` has already scandir'd this directory successfully — its files
    are enumerated and reach ``results`` — so this stat is evidence about
    directory *identity*, never about whether a file exists. Routing it to
    ``errors.dirs`` would suppress deletion and orphan pruning for a subtree
    that was fully walked and block the git anchor (ADR-51, narrowed by
    ADR-54), on a question it did not answer: exactly the over-correction
    ADR-54 exists to undo, reached from the other direction.

    What actually degrades is the duplicate guard, and only under
    ``follow_symlinks=True``. An unseeded directory is not pruned, so a symlink
    pointing back at it re-walks that subtree and yields the same files a
    second time under a different rel path. Logged rather than dropped: after
    ADR-51 this module's contract is that no filesystem fault vanishes without
    a trace, and a reader is entitled to see the one place where the trace is a
    log line instead of a recorded error (PR #24 round-7 review).
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        logger.debug(
            "discovery: cycle-guard stat failed for %s (%s) — symlink "
            "duplicate-detection degraded for this directory",
            path,
            exc,
        )
        return None
    return (st.st_dev, st.st_ino)


def discover_files(
    root: str | Path,
    config: DiscoveryConfig | None = None,
    *,
    errors: DiscoveryErrors | None = None,
) -> list[str]:
    """Return sorted, POSIX-style relative paths of indexable files under *root*.

    When *errors* is given, transient stat/read/scandir failures are recorded
    there instead of vanishing silently — callers doing change detection need
    them to tell "file errored" from "file deleted"; ``errors.entries_seen``
    additionally reports how many file entries the walk enumerated before
    filtering, which is the only way to tell an unreadable root from a fully
    filtered one. ``errors=None`` keeps the historical silent-skip
    behavior."""
    cfg = config or DiscoveryConfig()
    root_path = Path(root).resolve()

    def _walk_error(exc: OSError) -> None:
        # A *subdirectory* deleted mid-walk is a genuine deletion (same
        # discrimination as H7's file case); any other scandir failure means
        # part of the tree went unseen and must be surfaced when asked.
        #
        # The root is the exception. os.walk routes the walk root's own
        # scandir failure through this hook too, and an unmounted or renamed
        # root arrives as FileNotFoundError — so treating it as "genuine
        # absence" would report an empty tree with no error recorded, which
        # every caller reads as "every tracked file was deleted". A missing
        # root is never evidence that files were deleted; it is evidence the
        # scan could not run. An unattributable failure (no filename) is
        # treated the same way, since it cannot be proven to be a subtree.
        failed = Path(exc.filename) if exc.filename else None
        at_root = failed is None or failed == root_path
        if isinstance(exc, (FileNotFoundError, NotADirectoryError)) and not at_root:
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
            ident = _cycle_guard_identity(dirpath)
            if ident is not None:
                visited.add(ident)
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
                child_ident = _cycle_guard_identity(Path(dirpath) / d)
                if child_ident is None:
                    # Keep walking it. Skipping an unstattable directory would
                    # hide whatever it holds and hand the deletion path a set
                    # of "absent" files that were never actually looked for —
                    # purge on doubt, which ADR-51 refuses. It simply walks
                    # unguarded.
                    kept_dirs.append(d)
                    continue
                if child_ident in visited:
                    continue
                visited.add(child_ident)
            kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for name in filenames:
            # Counted before every filter: the question this answers is "did
            # the walk see anything?", not "did anything survive screening".
            if errors is not None:
                errors.entries_seen += 1
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
