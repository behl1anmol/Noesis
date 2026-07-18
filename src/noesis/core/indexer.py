"""Indexing pipeline: discover → hash-diff → chunk → embed → upsert (§3.2).

Orchestrates the M1 spine (discovery, hashdiff, state) with the M2 pieces
(chunker, Embedder, VectorStore). Dense channel only in M2 — the sparse/BM25
channel and RRF fusion land in M3; the git fast-path narrows the candidate
set in M7. Embedding batches go through the Embedder Protocol at LOW
priority so live queries preempt indexing (§3.8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from . import gitfast, hashdiff, state
from .chunker import chunk_file
from .discovery import DiscoveryConfig, discover_files
from .embedder import Embedder
from .languages import detect_language
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


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

    Goes through :func:`state.try_start_run` — the same guard as
    jobs.launch_index_run — and raises RuntimeError when a live run
    already holds the project: returning that run's id would make this
    caller execute it concurrently with its owner, the exact race the
    guard exists to prevent.
    """
    project_id = state.register_project(conn, root_path, embedder.model_id)
    run_id, created = state.try_start_run(conn, project_id)
    if not created:
        raise RuntimeError(
            f"index run {run_id} already running for project {project_id}"
        )
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
        run_started = time.perf_counter()
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
            # Heavy sync spans (git subprocesses, tree walk, hashing, file
            # IO, Qdrant round-trips) run via to_thread throughout this
            # function: the run shares the event loop with live queries and
            # must never starve them (§3.8). Quick state.* calls stay on the
            # loop — the shared sqlite conn is serialized either way.
            git_info = (
                await asyncio.to_thread(gitfast.compute_candidates, root_path, anchor)
                if anchor
                else None
            )
            if git_info is not None:
                head = git_info.head_commit
                candidates = git_info.candidates
            else:
                # Full walk this run, but still record HEAD (on success) so
                # the next run has an anchor to fast-path from (rule 4).
                head = await asyncio.to_thread(gitfast.resolve_head, root_path)

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
        # Discovery walks the whole tree and binary-sniffs each file; on large
        # or network-mounted trees this alone is slow and, until now, silent.
        disc_started = time.perf_counter()
        discovered = await asyncio.to_thread(
            discover_files, root_path, discovery_config
        )
        logger.info(
            "index run %s: discovery took=%.1fs discovered=%d",
            run_id,
            time.perf_counter() - disc_started,
            len(discovered),
        )
        stored = state.get_file_states(conn, project_id)

        # Drift self-heal (ADR-49): the state DB can claim files are indexed
        # while Qdrant holds none of their points — an externally dropped or
        # recreated collection. Hash comparison then sees no content change,
        # the run writes nothing, and search stays empty for those files
        # permanently. Gate cheaply: one exact project-scoped point count vs
        # the stored chunk total. Only a mismatch pays for the per-file
        # scroll. The total gate can miss *balanced* drift (one file over by
        # N points, another under by N) — accepted to keep the steady state
        # at a single count query.
        drifted: set[str] = set()
        orphan_paths: list[str] = []
        stored_chunk_counts = state.get_file_chunk_counts(conn, project_id)
        expected_points = sum(stored_chunk_counts.values())
        actual_points = await asyncio.to_thread(
            store.count_project_points, project_id
        )
        if actual_points != expected_points:
            if paths is not None:
                # Scoped (watcher) run: keep its narrow candidate set and its
                # never-advance-anchor rule — surface the drift only. A full
                # reindex is what heals it.
                logger.warning(
                    "index run %s: drift detected project=%s expected=%d "
                    "actual=%d — scoped run, warning only; run a full reindex "
                    "to self-heal",
                    run_id,
                    project_id,
                    expected_points,
                    actual_points,
                )
            else:
                per_file = await asyncio.to_thread(
                    store.per_file_point_counts, project_id
                )
                discovered_set = set(discovered)
                drifted = {
                    rel
                    for rel, cc in stored_chunk_counts.items()
                    if per_file.get(rel, 0) != cc
                }
                # Points for a file_path neither tracked in state nor present
                # on disk: no other prune path covers them and they keep the
                # drift gate firing every run.
                orphan_paths = [
                    p
                    for p in per_file
                    if p not in stored_chunk_counts and p not in discovered_set
                ]
                logger.warning(
                    "index run %s: drift detected project=%s expected=%d "
                    "actual=%d — re-embedding %d drifted file(s), pruning %d "
                    "orphan path(s)",
                    run_id,
                    project_id,
                    expected_points,
                    actual_points,
                    len(drifted),
                    len(orphan_paths),
                )
                if candidates is not None and drifted:
                    # Widening-only, exactly like the H1 dirty-paths union
                    # above: forces partition to re-hash drifted files so the
                    # re-embed uses their true current hash (rule 1 safe).
                    # When candidates is None, partition already hashes the
                    # whole tree, so no union is needed.
                    candidates = gitfast.CandidatePathSet(
                        set(candidates) | drifted
                    )

        diff = await asyncio.to_thread(
            hashdiff.partition, root_path, discovered, stored, candidates=candidates
        )

        chunks_written = 0
        file_errors: list[tuple[str, str]] = []
        to_index = [*diff.new, *diff.changed]
        if drifted:
            # Drifted-but-unchanged files: content hash still matches state,
            # yet their points are missing/mismatched in Qdrant — exactly what
            # incremental indexing skips. Re-embed is idempotent (deterministic
            # point ids), so this restores the missing points. Files deleted
            # from disk are excluded: they never enter diff.hashes.
            present = set(to_index)
            to_index.extend(
                rel
                for rel in sorted(drifted)
                if rel in diff.hashes and rel not in present
            )
        # Start milestone: counts only, no paths or content (ADR-25).
        logger.info(
            "index run %s: project=%s to_index=%d (new=%d changed=%d deleted=%d "
            "drifted=%d) discovered=%d",
            run_id,
            project_id,
            len(to_index),
            len(diff.new),
            len(diff.changed),
            len(diff.deleted),
            len(drifted),
            len(discovered),
        )
        # Throttle progress lines: the larger of every-50-files and every-10%,
        # so a big repo logs ~10 lines, not thousands. Empty to_index skips the
        # loop entirely, so the /0 case never arises.
        progress_step = max(50, (len(to_index) + 9) // 10)
        for done, rel in enumerate(to_index, start=1):
            # Per-file failure containment (ADR-41): a bad file is recorded
            # and skipped — its old chunks keep serving, its state row stays
            # out of date so the next run retries it. Exception, not
            # BaseException: cancellation must still abort the run.
            try:
                text = await asyncio.to_thread(_read_text, root_path, rel)
                language = detect_language(rel)
                file_hash = diff.hashes[rel]
                chunks = await asyncio.to_thread(
                    chunk_file,
                    text,
                    language=language,
                    file_path=rel,
                    file_hash=file_hash,
                )
                if chunks:
                    vectors: list[list[float]] = []
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]
                        vectors.extend(
                            await embedder.embed_documents([c.text for c in batch])
                        )
                    # Shielded, then awaited on cancel: asyncio delivers
                    # CancelledError at the await while the worker thread runs
                    # on, so a plain `await asyncio.to_thread(...)` lets an
                    # abandoned write land after a concurrent delete_project
                    # wipe — orphaning those points under a dead project_id
                    # (nothing prunes them; every prune path is scoped to a
                    # live project). Cancellation still aborts the run; it
                    # just waits out the one write already in flight.
                    upsert = asyncio.ensure_future(
                        asyncio.to_thread(
                            store.upsert_chunks,
                            project_id,
                            chunks,
                            vectors,
                            embedding_model=embedder.model_id,
                        )
                    )
                    try:
                        await asyncio.shield(upsert)
                    except asyncio.CancelledError:
                        await asyncio.gather(upsert, return_exceptions=True)
                        raise
                # New points first, stale points after: chunk ids embed the file
                # hash, so old content lives at different ids and must be pruned —
                # but only once the replacement is searchable. A failure above
                # leaves the old chunks serving; a failure below leaves brief
                # duplicates that the next (self-healing) run prunes.
                await asyncio.to_thread(
                    store.delete_file_chunks,
                    project_id,
                    [rel],
                    exclude_file_hash=file_hash,
                )
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
                # Path at DEBUG only: it can reveal repo structure if logs are
                # pasted into a bug report (telemetry.py precedent). Never the
                # file contents.
                logger.debug("index run %s: file %s failed: %s", run_id, rel, exc)
            if on_progress is not None:
                on_progress(done, len(to_index), chunks_written)
            if done % progress_step == 0 or done == len(to_index):
                logger.info(
                    "index run %s: %d/%d files chunks=%d",
                    run_id,
                    done,
                    len(to_index),
                    chunks_written,
                )

        if diff.deleted:
            await asyncio.to_thread(store.delete_file_chunks, project_id, diff.deleted)
            state.delete_files(conn, project_id, diff.deleted)

        if orphan_paths:
            # Drift cleanup: points whose file_path is neither tracked in
            # state nor on disk. No state rows correspond to them, so this
            # only prunes Qdrant; without it the drift gate keeps firing.
            await asyncio.to_thread(
                store.delete_file_chunks, project_id, orphan_paths
            )

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
        # Two distinct total-failure shapes: every chunk/embed attempt failed
        # (existing check), or every hash attempt failed and nothing else in
        # the whole discovered set is in a known-good state (PR #14 review) —
        # a whole-tree permission/network-fs outage, where every candidate
        # lands in hash_errors and `to_index` stays empty, so the chunk/embed
        # check alone would never see it and the run would report "done"
        # despite indexing nothing. `verified + skipped == 0` (not just
        # `verified == 0`) is required: under the git fast path a single
        # transient failure on a narrow one-file candidate set also leaves
        # verified at 0, but the rest of the tree is legitimately healthy via
        # `skipped` (fast-path carry-forward, never in doubt) — that stays a
        # contained per-file failure, not a run-wide outage.
        #
        # The chunk/embed check needs the same remainder guard: every file in
        # to_index hashed successfully (each is counted in `verified`), so the
        # known-good remainder is `verified - len(to_index) + skipped`. With a
        # non-zero remainder, "every attempted file failed" is a scoped run's
        # one edited file hitting a transient embed/store error while the rest
        # of the tree is provably healthy — contained per ADR-41, not an
        # outage. Only when nothing outside to_index is in a known-good state
        # does total chunk/embed failure mean the infrastructure is down.
        known_good_remainder = diff.verified - len(to_index) + diff.skipped
        chunk_embed_all_failed = (
            bool(to_index)
            and len(file_errors) == len(to_index)
            and known_good_remainder == 0
        )
        hash_all_failed = bool(hash_errors) and diff.verified + diff.skipped == 0
        all_failed = chunk_embed_all_failed or hash_all_failed
        total_failed = len(file_errors) + len(hash_errors)
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
            files_failed=total_failed,
            error=f"all {total_failed} files failed" if all_failed else None,
        )
        # A run with failed files must not advance the git anchor either:
        # the failed files' state rows are stale, and anchoring past them
        # would let the next fast path carry them forward as unchanged.
        if head is not None and not file_errors and not hash_errors:
            # Persist the working-tree-dirty set with the anchor (H1). The
            # fast path already captured it at run start (git_info); a
            # full-walk run re-queries `git status` here.
            if git_info is not None:
                new_dirty = git_info.dirty_paths
            else:
                new_dirty = (
                    await asyncio.to_thread(gitfast.status_dirty_paths, root_path)
                    or frozenset()
                )
            state.set_last_indexed_commit(conn, project_id, head, dirty_paths=new_dirty)
        # Completion milestone: counts + duration, so a finished run is visible
        # even when the dashboard isn't being watched. errors is a count; the
        # failed paths themselves stay at DEBUG (ADR-25 exposure).
        logger.info(
            "index run %s finished: status=%s files_indexed=%d chunks=%d errors=%d "
            "took=%.1fs",
            run_id,
            "failed" if all_failed else "done",
            len(to_index) - len(file_errors),
            chunks_written,
            total_failed,
            time.perf_counter() - run_started,
        )
        if total_failed:
            logger.debug(
                "index run %s failed paths: %s",
                run_id,
                ", ".join(path for path, _ in (*file_errors, *hash_errors)),
            )
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
            files_failed=total_failed,
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
