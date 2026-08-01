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

    The two lists above are about a path's EXISTENCE. The two below are not:
    the walk reached the directory and enumerated it fine, and every file in
    it is accounted for — what failed is an input to *screening*, so the
    result may be the wrong SHAPE rather than incomplete (ADR-58).

    ``unscreened``: a directory whose .gitignore could not be tested for or
    read, so its ignore rules were never applied. Unlike every other failure
    in this module this one runs in BOTH directions. The obvious half is
    under-ignoring — excluded files reach the index, a retrievable surface.
    The half that is easy to miss, and the reason this is not merely
    cosmetic: git's last-match-wins semantics mean a deeper .gitignore can
    NEGATE a shallower exclusion (``*.log`` at the root, ``!important.log``
    in ``logs/``). Lose the deeper spec and the outer exclusion stands
    unopposed, the file drops out of the results, and "absent from
    discovery" is read downstream as deletion. So this list DOES suppress
    deletion — under its own directory prefix only, via the ADR-54
    machinery — even though nothing about the walk was incomplete.

    ``unidentified``: a directory whose identity could not be established —
    the ``follow_symlinks`` cycle-guard ``stat``, or the per-child
    ``is_symlink`` test. This one can only make the walk OVER-inclusive (a
    subtree walked twice under two rel paths, or a symlink descended into
    that would normally be skipped); it can never remove a path from the
    results, so it is never deletion evidence and suppresses nothing.
    Recorded anyway, so that no filesystem fault in this module is invisible.

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
    # Keyed on the root-relative directory, using the same ``<root>`` sentinel
    # ``dirs`` uses, so `indexer.is_attributable_prefix` classifies all three
    # lists with one rule.
    unscreened: list[tuple[str, str]] = field(default_factory=list)
    unidentified: list[tuple[str, str]] = field(default_factory=list)


def is_secret_path(rel_posix: str) -> bool:
    """True if a project-relative POSIX path matches the secret skip-list."""
    return bool(_SECRET_SPEC.match_file(rel_posix))


def _dir_key(dir_rel: str) -> str:
    """Collector key for a directory: its rel path, or the root sentinel.

    The walk root's rel path is the empty string, which describes no subtree
    and would silently match nothing in `indexer._under`. ``<root>`` is the
    sentinel `_walk_error` already records for the same situation, and
    `indexer.is_attributable_prefix` already rejects it — so reusing it here
    means the new lists need no new classification rule.
    """
    return dir_rel or "<root>"


def _record_degraded(
    bucket: list[tuple[str, str]] | None,
    key: str,
    path: str | Path,
    exc: OSError,
    summary: str,
) -> None:
    """Report a screening fault: always logged, recorded when asked (ADR-58).

    The DEBUG log fires unconditionally, including for the ``errors=None``
    callers (structural search, the registration preview) that keep the
    historical silent-skip contract — "silent" was never meant to include
    "leaves no trace at all".

    Only *summary* and the errno reach the collector, never *path*: the
    recorded pairs are persisted to `run_file_errors` and rendered on the
    dashboard, and the key is the directory rel, so an absolute filesystem
    path in the message would be both redundant and a wider exposure surface
    than the rest of the table (ADR-25). The log line, which stays local and
    is already at DEBUG for that reason, carries the full path an operator
    needs to go fix it.
    """
    detail = str(exc) or type(exc).__name__
    logger.debug("discovery: %s — %s (%s)", summary, path, detail)
    if bucket is not None:
        bucket.append((key, f"{summary} ({detail})"))


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
        """Load one .gitignore onto the stack.

        Lets ``OSError`` propagate deliberately (issue #26, ADR-58). It used
        to be swallowed here with a bare ``return``, which put the only
        record of the failure out of reach of the collector; `discover_files`
        is the single caller and owns every other fault report in this
        module, so the handling belongs there next to the sibling
        ``is_file()`` test that fails the same way for the same reasons.
        """
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
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


def _cycle_guard_identity(
    path: str | Path,
    *,
    bucket: list[tuple[str, str]] | None = None,
    key: str = "<root>",
) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` for the ``follow_symlinks`` cycle guard, or None.

    A failure here is recorded in ``DiscoveryErrors.unidentified`` and
    deliberately NOT in ``dirs`` or ``files``. ``os.walk`` has already
    scandir'd this directory successfully — its files are enumerated and reach
    ``results`` — so this stat is evidence about directory *identity*, never
    about whether a file exists. Routing it to ``errors.dirs`` would suppress
    deletion and orphan pruning for a subtree that was fully walked and block
    the git anchor (ADR-51, narrowed by ADR-54), on a question it did not
    answer: exactly the over-correction ADR-54 exists to undo, reached from
    the other direction. ``unidentified`` exists precisely so the failure can
    be reported without being mistaken for evidence (ADR-58).

    What actually degrades is the duplicate guard, and only under
    ``follow_symlinks=True``. An unseeded directory is not pruned, so a symlink
    pointing back at it re-walks that subtree and yields the same files a
    second time under a different rel path. That can only ADD paths to the
    result, which is why it is safe to suppress nothing.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        _record_degraded(
            bucket,
            key,
            path,
            exc,
            "cycle-guard stat failed, so symlink duplicate-detection is "
            "degraded for this directory — it may be walked twice under two "
            "rel paths, never omitted",
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

    # `errors=None` keeps the historical silent-skip contract for the callers
    # that do no change detection (structural search, the ADR-42 registration
    # preview). `_record_degraded` still logs in that case — these are the
    # buckets, not the reporting switch.
    unscreened = None if errors is None else errors.unscreened
    unidentified = None if errors is None else errors.unidentified

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
        dir_rel = PurePosixPath(Path(dirpath).relative_to(root_path)).as_posix()
        if dir_rel == ".":
            dir_rel = ""
        # Computed before the cycle guard purely so the guard has a key to
        # report against; it reads nothing from the filesystem.
        dir_key = _dir_key(dir_rel)

        if cfg.follow_symlinks:
            ident = _cycle_guard_identity(dirpath, bucket=unidentified, key=dir_key)
            if ident is not None:
                visited.add(ident)

        # Both .gitignore faults land in `unscreened` and both are non-fatal.
        # Before issue #26 the first one RAISED — CPython's `_ignore_error`
        # covers only (ENOENT, ENOTDIR, EBADF, ELOOP), so `is_file()` re-raises
        # EACCES and ESTALE — and it ran unconditionally once per walked
        # directory, so a single unstattable path took down the whole run, and
        # with it `structural_search` (an unhandled 500) and the registration
        # preview. The second was swallowed with a bare `return`.
        #
        # Continuing is not the same as ignoring: the directory prefix is
        # recorded, and the indexer suppresses deletion under it (ADR-58),
        # because a lost .gitignore can drop a file OUT of the results as well
        # as let one in — a deeper spec's negation of a shallower exclusion
        # stops applying, and downstream "absent from the walk" means deleted.
        gitignore = Path(dirpath) / ".gitignore"
        try:
            has_gitignore = gitignore.is_file()
        except OSError as exc:
            has_gitignore = False
            _record_degraded(
                unscreened,
                dir_key,
                gitignore,
                exc,
                "could not test whether this directory holds a .gitignore, so "
                "any rules it holds were not applied",
            )
        if has_gitignore:
            try:
                ignores.push(dir_rel, gitignore)
            except OSError as exc:
                _record_degraded(
                    unscreened,
                    dir_key,
                    gitignore,
                    exc,
                    "this directory's .gitignore exists but could not be read, "
                    "so its rules were not applied",
                )

        kept_dirs = []
        for d in sorted(dirnames):
            if d in EXCLUDED_DIRS:
                continue
            child_rel = f"{dir_rel}/{d}" if dir_rel else d
            if ignores.ignored(child_rel, is_dir=True):
                continue
            if not cfg.follow_symlinks:
                try:
                    is_link = (Path(dirpath) / d).is_symlink()
                except OSError as exc:
                    # Same re-raise mechanism as the .gitignore test above, and
                    # before issue #26 this also killed the run. Treated as
                    # "not a symlink" so the directory is still walked: pruning
                    # it would hide whatever it holds from the deletion path,
                    # handing downstream a set of "absent" files nobody looked
                    # for — purge on doubt, which ADR-51 refuses. That is the
                    # same call the `follow_symlinks=True` branch below already
                    # makes for an unstattable child.
                    #
                    # Recorded in `unidentified`, which suppresses nothing,
                    # because the cost runs only one way: `os.walk` may now
                    # descend into a link it would normally skip, so the result
                    # can gain paths. It cannot lose any.
                    is_link = False
                    _record_degraded(
                        unidentified,
                        child_rel,
                        Path(dirpath) / d,
                        exc,
                        "could not determine whether this child directory is a "
                        "symlink, so it is walked unguarded",
                    )
                if is_link:
                    continue
            if cfg.follow_symlinks:
                child_ident = _cycle_guard_identity(
                    Path(dirpath) / d, bucket=unidentified, key=child_rel
                )
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
