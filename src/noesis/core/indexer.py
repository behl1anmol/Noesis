"""Indexing pipeline: discover → hash-diff → chunk → embed → upsert (§3.2).

Orchestrates the M1 spine (discovery, hashdiff, state) with the M2 pieces
(chunker, Embedder, VectorStore). Dense channel only in M2 — the sparse/BM25
channel and RRF fusion land in M3; the git fast-path narrows the candidate
set in M7. Embedding batches go through the Embedder Protocol at LOW
priority so live queries preempt indexing (§3.8).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

from . import gitfast, hashdiff, state
from .chunker import chunk_file
from .discovery import DiscoveryConfig, discover_files
from .embedder import Embedder
from .languages import detect_language
from .vectorstore import VectorStore


@dataclass(frozen=True)
class IndexResult:
    project_id: str
    run_id: str
    files_total: int
    files_indexed: int
    files_deleted: int
    chunks_written: int
    fast_path_used: bool = False
    candidate_count: int | None = None
    files_failed: int = 0
    # Paths whose per-file processing failed (ADR-41): callers that clear
    # pending_changes must keep these pending or auto-reindex never retries.
    failed_paths: tuple[str, ...] = ()


# Live-progress callback: (files_done, files_to_index, chunks_written).
# Called after each processed file — consumers (jobs.py) keep it in memory
# for the dashboard; nothing here touches the DB per-file beyond upsert_file.
ProgressFn = Callable[[int, int, int], None]


def discovery_config_for_project(
    conn: sqlite3.Connection, project_id: str
) -> DiscoveryConfig | None:
    """Build a project's persisted index scope (ADR-42) into a
    DiscoveryConfig. Returns None for an unknown project (caller falls
    back to the default walk). NULL columns map to DiscoveryConfig
    defaults, so an unconfigured project walks exactly as before."""
    row = state.get_project(conn, project_id)
    if row is None:
        return None
    kwargs: dict = {"follow_symlinks": bool(row["follow_symlinks"])}
    if row["max_file_bytes"] is not None:
        kwargs["max_file_bytes"] = row["max_file_bytes"]
    if row["index_languages"]:
        kwargs["include_languages"] = frozenset(json.loads(row["index_languages"]))
    if row["extra_ignores"]:
        kwargs["extra_ignore_patterns"] = tuple(json.loads(row["extra_ignores"]))
    return DiscoveryConfig(**kwargs)


def prepare_run(
    conn: sqlite3.Connection, embedder: Embedder, root_path: str
) -> tuple[str, str]:
    """Register (or re-open) the project and open a run row.

    Split from :func:`execute_run` so the API can hand back
    ``202 Accepted + run_id`` before indexing starts (§3.2).
    """
    project_id = state.register_project(conn, root_path, embedder.model_id)
    run_id = state.start_run(conn, project_id)
    return project_id, run_id


async def index_project(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    root_path: str,
    *,
    batch_size: int = 32,
    discovery_config: DiscoveryConfig | None = None,
    git_fast_path: bool = True,
) -> IndexResult:
    """Register, open a run, and index in one call (tests / CLI use)."""
    project_id, run_id = prepare_run(conn, embedder, root_path)
    return await execute_run(
        conn,
        store,
        embedder,
        root_path,
        project_id,
        run_id,
        batch_size=batch_size,
        discovery_config=discovery_config,
        git_fast_path=git_fast_path,
    )


async def execute_run(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    root_path: str,
    project_id: str,
    run_id: str,
    *,
    batch_size: int = 32,
    discovery_config: DiscoveryConfig | None = None,
    git_fast_path: bool = True,
    paths: Sequence[str] | None = None,
    on_progress: ProgressFn | None = None,
) -> IndexResult:
    """Index changes for an already-registered project under an open run.

    Idempotent: chunk ids are content-derived and file state is only written
    after that file's chunks are safely in Qdrant, so an interrupted run
    re-processes only what is still out of date (Overview §5).

    *paths* scopes the run to an explicit candidate set (the watcher's
    pending files, ADR-40): only those files (plus anything unknown to
    stored state) are hashed; deletions are still detected discovery-wide.
    A scoped run disables the git fast path and NEVER advances
    ``last_indexed_commit`` — the anchor may only move when the candidate
    set was derived from a git diff against it, otherwise the next fast
    path would silently skip files the watcher never saw (§3.2 rule 1).
    """
    if paths is not None:
        git_fast_path = False
    try:
        # Git fast-path (§3.2): capture HEAD and the candidate set BEFORE
        # discovery/hashing — anything committed after this point is simply
        # re-examined next run (the safe direction). Every fallback is
        # silent here and logged in gitfast; hash stays the truth.
        head: str | None = None
        git_info: "gitfast.GitCandidates | None" = None
        candidates: "gitfast.CandidatePathSet | frozenset[str] | None" = None
        if paths is not None:
            candidates = frozenset(paths)
        if git_fast_path:
            anchor = None
            project = state.get_project(conn, project_id)
            if project is not None:
                anchor = project["last_indexed_commit"]
            git_info = gitfast.compute_candidates(root_path, anchor) if anchor else None
            if git_info is not None:
                head = git_info.head_commit
                candidates = git_info.candidates
            else:
                # Full walk this run, but still record HEAD (on success) so
                # the next run has an anchor to fast-path from (rule 4).
                head = gitfast.resolve_head(root_path)

        # H1: re-admit files that were working-tree-dirty at the last anchor
        # advance. If such a file was reverted to HEAD since, neither the diff
        # nor `git status` surfaces it now, yet its stored hash is the stale
        # dirty content — carrying it forward forever. Unioning the persisted
        # dirty set only widens the candidate set (rule 1 safe) and forces a
        # re-hash that detects the revert.
        if candidates is not None:
            prior_dirty = state.get_dirty_paths(conn, project_id)
            if prior_dirty:
                candidates = gitfast.CandidatePathSet(
                    set(candidates) | set(prior_dirty)
                )

        fast_path_active = git_fast_path and candidates is not None

        # No explicit config → honor the project's persisted index scope
        # (ADR-42), so manual/watcher/reindex runs all apply the same filter.
        if discovery_config is None:
            discovery_config = discovery_config_for_project(conn, project_id)
        discovered = discover_files(root_path, discovery_config)
        stored = state.get_file_states(conn, project_id)
        diff = hashdiff.partition(root_path, discovered, stored, candidates=candidates)

        chunks_written = 0
        file_errors: list[tuple[str, str]] = []
        to_index = [*diff.new, *diff.changed]
        for done, rel in enumerate(to_index, start=1):
            # Per-file failure containment (ADR-41): a bad file is recorded
            # and skipped — its old chunks keep serving, its state row stays
            # out of date so the next run retries it. Exception, not
            # BaseException: cancellation must still abort the run.
            try:
                text = _read_text(root_path, rel)
                language = detect_language(rel)
                file_hash = diff.hashes[rel]
                chunks = chunk_file(
                    text, language=language, file_path=rel, file_hash=file_hash
                )
                if chunks:
                    vectors: list[list[float]] = []
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]
                        vectors.extend(
                            await embedder.embed_documents([c.text for c in batch])
                        )
                    store.upsert_chunks(
                        project_id, chunks, vectors, embedding_model=embedder.model_id
                    )
                # New points first, stale points after: chunk ids embed the file
                # hash, so old content lives at different ids and must be pruned —
                # but only once the replacement is searchable. A failure above
                # leaves the old chunks serving; a failure below leaves brief
                # duplicates that the next (self-healing) run prunes.
                store.delete_file_chunks(project_id, [rel], exclude_file_hash=file_hash)
                state.upsert_file(
                    conn,
                    project_id,
                    rel,
                    file_hash,
                    language=language,
                    chunk_count=len(chunks),
                )
                chunks_written += len(chunks)
            except Exception as exc:  # noqa: BLE001 — contained per ADR-41
                file_errors.append((rel, str(exc) or type(exc).__name__))
            if on_progress is not None:
                on_progress(done, len(to_index), chunks_written)

        if diff.deleted:
            store.delete_file_chunks(project_id, diff.deleted)
            state.delete_files(conn, project_id, diff.deleted)

        # Hash-time failures (H7 carry-forward) are per-file failures too:
        # the file's true state is unknown this run, its stored hash may be
        # stale, and — unlike chunk/embed failures — it never entered
        # to_index, so without this it would be invisible to every guard
        # below and silently stranded by the next fast path.
        hash_errors = [(path, f"hash failed: {msg}") for path, msg in diff.errored]
        if file_errors or hash_errors:
            state.record_file_errors(conn, run_id, [*file_errors, *hash_errors])
        # Partial failure is still a completed run (ADR-41); a run where
        # every file failed is not "done" by any honest reading — likely an
        # infrastructure problem (store/embedder down) wearing per-file dress.
        all_failed = bool(to_index) and len(file_errors) == len(to_index)
        state.finish_run(
            conn,
            run_id,
            "failed" if all_failed else "done",
            files_total=len(discovered),
            files_changed=len(to_index),
            chunks_written=chunks_written,
            fast_path_used=fast_path_active,
            candidate_count=len(candidates)
            if fast_path_active and candidates is not None
            else None,
            files_failed=len(file_errors) + len(hash_errors),
            error=f"all {len(file_errors)} files failed" if all_failed else None,
        )
        # A run with failed files must not advance the git anchor either:
        # the failed files' state rows are stale, and anchoring past them
        # would let the next fast path carry them forward as unchanged.
        if head is not None and not file_errors and not hash_errors:
            # Persist the working-tree-dirty set with the anchor (H1). The
            # fast path already captured it at run start (git_info); a
            # full-walk run re-queries `git status` here.
            new_dirty = (
                git_info.dirty_paths
                if git_info is not None
                else (gitfast.status_dirty_paths(root_path) or frozenset())
            )
            state.set_last_indexed_commit(conn, project_id, head, dirty_paths=new_dirty)
        return IndexResult(
            project_id=project_id,
            run_id=run_id,
            files_total=len(discovered),
            files_indexed=len(to_index) - len(file_errors),
            files_deleted=len(diff.deleted),
            chunks_written=chunks_written,
            fast_path_used=fast_path_active,
            candidate_count=len(candidates)
            if fast_path_active and candidates is not None
            else None,
            files_failed=len(file_errors) + len(hash_errors),
            failed_paths=tuple(path for path, _ in (*file_errors, *hash_errors)),
        )
    except BaseException as exc:
        # BaseException: CancelledError (e.g. server shutdown) must also mark
        # the run failed, or it would sit "running" forever.
        state.finish_run(conn, run_id, "failed", error=str(exc) or type(exc).__name__)
        raise


def _read_text(root_path: str, rel: str) -> str:
    from pathlib import Path

    return (Path(root_path) / rel).read_text(encoding="utf-8", errors="replace")
