"""Structural search over live files via ast-grep (§3.5, ADR-21/22).

Bypasses Qdrant, SQLite chunk state, and the embedder entirely (Finding 5):
results are pattern matches against the filesystem as it is *right now*, so
they are never stale by construction. It reuses exactly two core surfaces —
the project registry (``project_id`` → ``root_path``) and the discovery
filter — so structural search can never surface a file that indexing would
have excluded (.gitignore, secret skip-list, size caps). Search only:
ast-grep's rewrite capability is never exposed (ADR-22).

The scan is blocking I/O plus native Rust compute, so ``structural_search``
runs it via ``run_in_executor`` on the default thread pool — never on the
embedder/reranker workers, which must not queue behind a scan (§3.5).
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from sqlite3 import Connection
from typing import Any, TypedDict

from ast_grep_py import SgRoot

from noesis.core import state
from noesis.core.config import StructuralSettings
from noesis.core.discovery import DiscoveryConfig, discover_files
from noesis.core.indexer import discovery_config_for_project
from noesis.core.languages import LANGUAGE_MAP, detect_language

# Metavariable names are extracted from the pattern text because ast-grep-py
# exposes captures only by name (get_match / get_multiple_matches), not as a
# collection. $$$NAME is a multi metavar, $NAME single; anonymous $$$ captures
# nothing nameable. Same lexical rules as ast-grep: uppercase, _ and digits.
_MULTI_METAVAR = re.compile(r"\$\$\$([A-Z_][A-Z0-9_]*)")
_SINGLE_METAVAR = re.compile(r"(?<!\$)\$([A-Z_][A-Z0-9_]*)")


class StructuralMatch(TypedDict):
    file_path: str  # project-relative POSIX path
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    matched_text: str
    meta_vars: dict[str, Any]  # single → str | None, multi → list[str]


class StructuralResult(TypedDict):
    matches: list[StructuralMatch]
    scanned_files: int
    truncated: bool  # scan stopped at max_results; more matches may exist
    timed_out: bool  # stopped at the timeout_s budget; matches are partial


class StructuralSearchError(Exception):
    """Structured, adapter-mappable failure. ``error_type`` is one of
    ``unknown_project`` / ``unsupported_language`` / ``pattern_error`` /
    ``invalid_path``; ``message`` carries the diagnostic (for pattern
    errors, ast-grep's own — agents iterate on patterns, keep that cheap)."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def _validate_pattern(pattern: str, ast_grep_lang: str) -> None:
    """Reject patterns ast-grep cannot compile, with its diagnostic.

    ast-grep is lenient: most malformed patterns compile and simply match
    nothing (that is a legitimate empty result, not an error). Only patterns
    the engine refuses outright raise — probe once against empty source so
    the failure surfaces before any file is read.

    Runs on the caller's thread (the event loop): parsing empty source is
    microseconds. If this probe ever grows real work, move it into the
    executor with the scan.
    """
    try:
        SgRoot("", ast_grep_lang).root().find_all(pattern=pattern)
    except RuntimeError as exc:
        raise StructuralSearchError("pattern_error", str(exc)) from exc


def _normalize_paths(paths: list[str] | None) -> list[str] | None:
    """Normalize subtree restrictions to project-relative POSIX prefixes."""
    if not paths:
        return None
    normalized = []
    for p in paths:
        if Path(p).is_absolute() or ".." in Path(p).parts:
            raise StructuralSearchError(
                "invalid_path",
                f"paths must be project-relative without '..': {p!r}",
            )
        stripped = p.strip("/")
        if not stripped:  # "" or "/" would silently match nothing
            raise StructuralSearchError(
                "invalid_path",
                f"empty path restriction {p!r}; omit paths to scan the whole project",
            )
        normalized.append(stripped)
    return normalized


def _meta_var_names(pattern: str) -> tuple[list[str], list[str]]:
    multi = _MULTI_METAVAR.findall(pattern)
    stripped = _MULTI_METAVAR.sub("", pattern)
    single = [n for n in _SINGLE_METAVAR.findall(stripped) if n not in multi]
    return single, multi


def _scan_sync(
    root_path: str,
    pattern: str,
    language: str,
    paths: list[str] | None,
    max_results: int,
    timeout_s: float,
    discovery_config: DiscoveryConfig | None,
) -> StructuralResult:
    mapping = LANGUAGE_MAP[language]
    single_vars, multi_vars = _meta_var_names(pattern)
    deadline = time.monotonic() + timeout_s

    candidates = [
        rel
        for rel in discover_files(root_path, discovery_config)
        if detect_language(rel) == language
        and (paths is None or any(rel == p or rel.startswith(p + "/") for p in paths))
    ]

    matches: list[StructuralMatch] = []
    scanned = 0
    truncated = False
    timed_out = False
    for rel in candidates:
        # Budget granularity is per file: a single file can overrun the
        # deadline by its own parse+match time, which discovery bounds at
        # max_file_bytes (1 MB) — small enough that we skip mid-file checks.
        if time.monotonic() > deadline:
            timed_out = True
            break
        try:
            src = (Path(root_path) / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # deleted/unreadable mid-scan — live filesystem, skip
        scanned += 1
        for node in SgRoot(src, mapping.ast_grep).root().find_all(pattern=pattern):
            if len(matches) >= max_results:
                break
            rng = node.range()
            meta: dict[str, Any] = {}
            for name in single_vars:
                cap = node.get_match(name)
                meta[name] = cap.text() if cap is not None else None
            for name in multi_vars:
                meta[name] = [
                    n.text() for n in node.get_multiple_matches(name) if n.is_named()
                ]
            matches.append(
                StructuralMatch(
                    file_path=rel,
                    start_line=rng.start.line + 1,
                    end_line=rng.end.line + 1,
                    matched_text=node.text(),
                    meta_vars=meta,
                )
            )
        if len(matches) >= max_results:
            # Cap reached — stop scanning. More matches may exist in files
            # (or nodes) never examined, so this is always reported truncated.
            truncated = True
            break

    return StructuralResult(
        matches=matches, scanned_files=scanned, truncated=truncated, timed_out=timed_out
    )


async def structural_search(
    conn: Connection,
    project_id: str,
    pattern: str,
    language: str,
    *,
    paths: list[str] | None = None,
    max_results: int | None = None,
    settings: StructuralSettings | None = None,
) -> StructuralResult:
    """Pattern-match ASTs of live, discovery-filtered files (§3.5).

    *max_results* is clamped to the configured cap; *paths* optionally
    restricts the scan to project-relative subtrees.
    """
    cfg = settings or StructuralSettings()
    project = state.get_project(conn, project_id)
    if project is None:
        raise StructuralSearchError(
            "unknown_project", f"unknown project_id {project_id!r}"
        )
    cfg_discovery = discovery_config_for_project(conn, project_id)
    if language not in LANGUAGE_MAP:
        raise StructuralSearchError(
            "unsupported_language",
            f"{language!r} is not supported for structural search; "
            f"supported: {', '.join(sorted(LANGUAGE_MAP))}",
        )
    _validate_pattern(pattern, LANGUAGE_MAP[language].ast_grep)
    norm_paths = _normalize_paths(paths)
    # Requests may lower the configured cap, never raise it; a non-positive
    # value (REST validates ge=1, but core has other callers — MCP in M6)
    # clamps to 1 rather than silently returning nothing.
    limit = (
        cfg.max_results if max_results is None else min(max_results, cfg.max_results)
    )
    limit = max(1, limit)

    return await asyncio.get_running_loop().run_in_executor(
        None,
        _scan_sync,
        project["root_path"],
        pattern,
        language,
        norm_paths,
        limit,
        cfg.timeout_s,
        cfg_discovery,
    )
