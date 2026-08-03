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
import os.path
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from . import gitfast, hashdiff, state
from .chunker import chunk_file
from .discovery import DiscoveryConfig, DiscoveryErrors, discover_files
from .embedder import Embedder
from .languages import detect_language
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


def is_attributable_prefix(dir_rel: str) -> bool:
    """True when *dir_rel* is a root-relative prefix ``_under`` can test.

    False for the three shapes that describe no subtree of this project: the
    empty string, the ``<root>`` sentinel, and a path outside the root (an
    absolute name, or one reached by ``../``) — which discovery records when
    ``relative_to`` fails. Public because the promotion policy in ``jobs`` asks
    the same question of a stored ledger row: an unattributable failure leaves
    the *whole* index unverified, which no scoped run can make progress on.
    """
    return bool(
        dir_rel
        and dir_rel != "<root>"
        and not os.path.isabs(dir_rel)
        and not dir_rel.startswith("../")
    )


def _error_prefixes(
    reported: Sequence[tuple[str, str]], *, follow_symlinks: bool
) -> tuple[str, ...] | None:
    """Root-relative prefixes under which *reported* voids deletion evidence.

    ``()`` means nothing is blocked. ``None`` means the failures cannot be
    attributed to a subtree and the whole run's deletion evidence is void:

    * the ``<root>`` sentinel — the walk root itself failed, or the empty-scan
      guard fired, so nothing at all was seen;
    * a path outside the root (``relative_to`` failed, so discovery recorded
      the absolute name) — no prefix of the project's rel paths matches it;
    * ``follow_symlinks`` — the same directory is reachable under rel paths
      that are not under its own, so prefix containment stops implying
      "unseen".

    Shared by the two collector lists that suppress deletion, so they cannot
    drift apart on any of those three edge cases.
    """
    if not reported:
        return ()
    if follow_symlinks:
        return None
    prefixes: list[str] = []
    for dir_rel, _ in reported:
        if not is_attributable_prefix(dir_rel):
            return None
        prefixes.append(dir_rel)
    return tuple(prefixes)


def _unwalked_prefixes(
    disc_errors: DiscoveryErrors, *, follow_symlinks: bool
) -> tuple[str, ...] | None:
    """Root-relative prefixes whose contents this walk never saw.

    A stored path under one of them cannot be proven deleted; every other
    stored path can, because os.walk only loses the failing directory's own
    subtree and carries on elsewhere. Returning the prefixes instead of a
    single project-wide "suppress everything" flag keeps a permanently
    unreadable directory — a root-owned dir, a dead mount — from freezing
    deletion processing and orphan pruning for the whole project forever
    (PR #24 review).

    Reads ``dirs`` ONLY. ``unscreened`` blocks deletion too but is not the same
    question and is deliberately kept out: this value also drives the ADR-56
    ledger, quarantine, and the recovery reset, and a directory that was walked
    perfectly well must never be counted as unwalkable there (ADR-58).
    """
    return _error_prefixes(disc_errors.dirs, follow_symlinks=follow_symlinks)


def _under(path: str, prefixes: tuple[str, ...]) -> bool:
    """True if *path* is one of *prefixes* or lives inside one."""
    return any(path == p or path.startswith(f"{p}/") for p in prefixes)


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
    # Paths the next run must re-examine: callers that clear pending_changes
    # must keep these pending or auto-reindex never retries them. Two
    # populations, despite the name — files whose own processing failed
    # (ADR-41), and files a directory-level discovery failure left unverified
    # (ADR-51/54), which did not fail at all; the run simply never reached
    # them. So this is NOT bounded by "how many files went wrong": when the
    # failure cannot be attributed to a subtree (a vanished root), every
    # stored path is unverified and this is the size of the index.
    #
    # That matters because the caller re-pends it with one `executemany` on
    # the event loop (jobs.py). Deliberate: `state.*` writes run on the
    # loop-owning thread throughout — the shared conn is loop-owned, and a
    # worker-thread write can interleave inside another operation's open
    # BEGIN IMMEDIATE (see jobs.index_status). At index scale the write is the
    # same order as `delete_files` and `clear_pending_changes` on either side
    # of it, so it is the established cost of that rule, not an exception to
    # it. Whether a whole-index re-pend should happen at all is the open
    # question in issue #25, not a property of this field.
    failed_paths: tuple[str, ...] = ()
    # Paths this run left OUT of `failed_paths` because the directory hiding
    # them is quarantined (ADR-56): it has failed to be walked for
    # `unwalkable_quarantine_runs` consecutive runs, so re-pending them every
    # run was a backlog that could never drain. They are still not deleted —
    # quarantine bounds the RETRY, never the deletion policy — and the run that
    # finds the directory walkable again re-hashes them all. Reported here so
    # "we stopped retrying N paths" is never a silent drop.
    unverified_suppressed: int = 0
    # Directories whose SCREENING inputs failed this run (ADR-58): a .gitignore
    # that could not be read or tested for, or a directory whose symlink
    # identity could not be established. Not a file count and not a failure
    # count — deliberately absent from `files_failed`, because nothing failed
    # to be indexed. It is the "this run's results may be the wrong shape"
    # signal: `unscreened` entries additionally hold back deletions under their
    # own prefix, which is reported separately in the run log.
    discovery_degraded: int = 0


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
    force: bool = False,
    unwalkable_quarantine_runs: int = 5,
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

    *force* accepts a discovery result of zero files as real deletion evidence
    even when state still tracks files. Off by default: nothing observable
    distinguishes an emptied root from an unmounted one, so the assertion
    has to come from an operator (ADR-55). It relaxes nothing else — a
    directory that failed to scan still suppresses deletions under it.

    *unwalkable_quarantine_runs* is how many consecutive runs a directory must
    fail to be walked before the stored paths it hides stop being re-queued for
    retry (ADR-56). ``0`` disables quarantine. It relaxes nothing about
    deletion either: a quarantined subtree's files are still never purged, and
    the run that finds the directory readable again re-hashes every one of
    them. All it bounds is the ``pending_changes`` backlog, which under a
    permanently unreadable directory could otherwise never drain (issue #25).
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
        # Collect discovery failures instead of letting them vanish: a file
        # (or whole subtree) that errored transiently is absent from
        # `discovered`, and without the collector hashdiff would read that
        # absence as deletion and purge live chunks.
        disc_errors = DiscoveryErrors()
        discovered = await asyncio.to_thread(
            discover_files, root_path, discovery_config, errors=disc_errors
        )
        logger.info(
            "index run %s: discovery took=%.1fs discovered=%d",
            run_id,
            time.perf_counter() - disc_started,
            len(discovered),
        )
        stored = state.get_file_states(conn, project_id)
        # A scan that returned NOTHING while state tracks files is not
        # evidence of mass deletion — it is evidence the scan did not run: an
        # unmounted or renamed root, or a mountpoint not populated yet. The
        # walk-root discrimination in discovery catches the errno cases, but a
        # root that exists and is simply empty raises nothing at all, so this
        # cannot be expressed as an error check — the emptiness itself is the
        # signal. Recorded as a directory-level error so it takes the same
        # path as an unwalked subtree: deletions and orphan pruning are
        # skipped, the anchor cannot advance, and the run is marked failed.
        #
        # Two escapes, because a guard that can never be satisfied is a guard
        # that strands the project (PR #24 review):
        #
        # * ``entries_seen`` — the walk enumerated files and the filters
        #   removed all of them. The root is demonstrably readable and
        #   populated, so the emptiness is a scope decision (an ADR-42
        #   narrowing, a new .gitignore), not a missing mount, and the
        #   now-out-of-scope files SHOULD leave the index.
        # * ``force`` — an operator asserting the empty root is real. Nothing
        #   the process can observe distinguishes "every file deleted" from
        #   "root not mounted", so that call is the operator's to make.
        #
        # ``force`` deliberately does NOT relax the directory-error
        # suppression below: an unwalked subtree is a different, still-open
        # proof problem, not something the operator asserted anything about.
        #
        # Both branches stand down when discovery already reported a directory
        # failure (PR #24 review). This guard exists for the emptiness that
        # raises nothing; when something DID raise, that error is the diagnosis
        # and it is already recorded. Synthesizing a second ``<root>`` row on
        # top of it would overwrite the real errno — ``record_file_errors`` is
        # INSERT OR REPLACE keyed on (run_id, path) — costing the operator the
        # EACCES/ESTALE they need to fix the mount, and double-counting one
        # failure in ``files_failed``. Under ``force`` the same condition
        # otherwise logs a purge that the deletion block then refuses to
        # perform. Deletion safety is unchanged either way: the recorded
        # directory error drives `unwalked` on its own.
        if (
            stored
            and not discovered
            and not force
            and not disc_errors.entries_seen
            and not disc_errors.dirs
        ):
            disc_errors.dirs.append(
                (
                    "<root>",
                    f"discovery returned no files while state tracks "
                    f"{len(stored)} file(s) — treating as an unreadable root, "
                    f"not as deletion",
                )
            )
            logger.warning(
                "index run %s: discovery returned 0 files but state tracks %d "
                "— refusing to treat that as deletion (root unreadable or "
                "not mounted?)",
                run_id,
                len(stored),
            )
        elif stored and not discovered and not disc_errors.dirs:
            # The purge the guard exists to prevent, now deliberate. Loud
            # either way: an operator reading the log after the fact must be
            # able to see which escape emptied the project — so it may only
            # fire where a purge can actually follow, and it may only claim
            # what the walk actually did. With no directory error recorded,
            # `_unwalked_prefixes` below returns `()` and nothing is
            # suppressed; with one, it returns a subtree or None and this line
            # would name an escape that emptied nothing.
            #
            # `entries_seen` counts every enumerated entry, including ones that
            # then failed to be screened — and a file that ERRORED was not
            # "filtered out": hashdiff carries it forward, so it is never
            # deleted. Report the two populations separately rather than
            # crediting the filters with a permission error.
            logger.warning(
                "index run %s: discovery returned 0 files while state tracks "
                "%d — accepting the empty scan as real (%s); what that deletes "
                "is still decided per file, so an unreadable path is carried "
                "forward, not purged",
                run_id,
                len(stored),
                "forced by caller"
                if force
                else f"walk enumerated {disc_errors.entries_seen} file entry/"
                f"entries ({disc_errors.entries_seen - len(disc_errors.files)} "
                f"filtered out, {len(disc_errors.files)} unreadable) — root is "
                f"readable",
            )

        # Which stored paths this walk could not see. Computed once, after
        # the empty-scan guard has had its chance to add `<root>`, and used
        # by both purge paths below (orphan pruning and deletion).
        # `discovery_config` may still be None here (no persisted scope row);
        # discover_files defaults it the same way, so read the flag from the
        # same default rather than assuming what it is.
        follow_symlinks = (discovery_config or DiscoveryConfig()).follow_symlinks
        unwalked = _unwalked_prefixes(disc_errors, follow_symlinks=follow_symlinks)
        # Prefixes where the walk saw everything but could not apply the ignore
        # rules (ADR-58). These block deletion for a reason that is the reverse
        # of `unwalked`'s: nothing went unseen, but a lost .gitignore can push a
        # file OUT of the results — git is last-match-wins, so a deeper spec's
        # negation of a shallower exclusion stops applying — and downstream
        # every absence reads as a deletion.
        #
        # Kept in its own value rather than merged into `unwalked`, because
        # `unwalked` also feeds the ADR-56 ledger, the quarantine test and the
        # recovery reset. A directory that walked fine must not accumulate
        # consecutive-failure counts toward quarantine, and a ledger row must
        # not be denied its reset because an unrelated .gitignore was degraded.
        unscreened = _error_prefixes(
            disc_errors.unscreened, follow_symlinks=follow_symlinks
        )

        # Directory-failure ledger (ADR-56). READ here, WRITTEN after the file
        # loop — the order is load-bearing. Reconciling early would delete a
        # recovered directory's row before the re-admission below had run, and
        # a crash in between would lose the only record that its files were
        # ever unverified: the ledger would say healthy, the files would carry
        # stale hashes, and nothing would re-hash them.
        ledger = state.list_unwalkable_dirs(conn, project_id)
        reported_now = {dir_rel for dir_rel, _ in disc_errors.dirs}
        prior_counts = {row["dir_path"]: row["consecutive_runs"] for row in ledger}
        # Quarantine is judged on the count this run WILL have, not the stored
        # one, so `unwalkable_quarantine_runs = N` means "suppressed on the Nth
        # consecutive failure" rather than the N+1'th. The reconcile below
        # applies the same >= test to the same post-increment number, so the
        # decision here and the `quarantined_at` an operator reads agree.
        quarantined: frozenset[str] = (
            frozenset()
            if unwalkable_quarantine_runs <= 0
            else frozenset(
                key
                for key in set(prior_counts) | reported_now
                if prior_counts.get(key, 0) + (1 if key in reported_now else 0)
                >= unwalkable_quarantine_runs
            )
        )
        # Recorded directories this walk proved walkable again. Requires
        # `unwalked is not None`: when the failure cannot be attributed to a
        # subtree the walk proved nothing about anything, so nothing clears.
        # A recorded path that is not under what went unseen this run was
        # reached — whether it is readable again, was deleted, or has been
        # filtered out of scope. All three are correct resets, and none of them
        # is a fault.
        recovered = (
            ()
            if unwalked is None
            else tuple(key for key in prior_counts if not _under(key, unwalked))
        )
        # Re-admit what the recovered subtrees hold, in the very run that
        # observes the recovery. This is what makes it safe to stop re-pending
        # a quarantined subtree: a file edited while its directory was
        # unreadable lost its watcher event, so without this nothing would ever
        # retry it. Widening-only, exactly like the H1 and drift unions
        # (§3.2 rule 1). Unattributable rows are deliberately excluded — no
        # prefix describes them, and unioning "everything" would silently turn
        # a scoped run into a whole-index hash; those force a full walk through
        # the promotion policy in `jobs` instead.
        recovered_prefixes = tuple(k for k in recovered if is_attributable_prefix(k))
        if recovered_prefixes and candidates is not None:
            readmit = {p for p in discovered if _under(p, recovered_prefixes)}
            if readmit:
                logger.info(
                    "index run %s: %d director(ies) walkable again — re-hashing "
                    "%d file(s) under them to catch edits made while they were "
                    "unreadable",
                    run_id,
                    len(recovered_prefixes),
                    len(readmit),
                )
                candidates = gitfast.CandidatePathSet(set(candidates) | readmit)

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
        actual_points = await asyncio.to_thread(store.count_project_points, project_id)
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
                #
                # Discovery failures make "absent from disk" unprovable, so
                # they must not feed this either — same purge-on-doubt rule as
                # the deletion guard below. A path missing from `discovered_set`
                # because its stat or its parent's scandir failed says nothing
                # about whether the file exists, and a file live on disk whose
                # state row is missing (a crash between `upsert_chunks` and
                # `upsert_file`) is exactly the shape that lands here: pruning
                # it deletes searchable points for a live file. A dir-level
                # failure hides its whole subtree and cannot enumerate what it
                # hid, so every path under it is spared; file-level failures
                # name their paths, so only those are. Paths elsewhere were
                # walked normally and stay provable. Cost of skipping is one
                # more drift-gate firing next run; cost of pruning wrongly is
                # content gone from search.
                #
                # `unscreened` is spared on the same rule (ADR-58): under a
                # directory whose ignore rules never loaded, "not in
                # discovered_set" can mean "wrongly excluded", not "gone".
                errored_files = {p for p, _ in disc_errors.files}
                orphan_paths = (
                    []
                    if unwalked is None or unscreened is None
                    else [
                        p
                        for p in per_file
                        if p not in stored_chunk_counts
                        and p not in discovered_set
                        and p not in errored_files
                        and not _under(p, unwalked)
                        and not _under(p, unscreened)
                    ]
                )
                if disc_errors.dirs:
                    logger.warning(
                        "index run %s: %d directory discovery error(s) — orphan "
                        "point pruning %s (absence is not evidence); a clean "
                        "run prunes them",
                        run_id,
                        len(disc_errors.dirs),
                        "skipped for the whole run"
                        if unwalked is None
                        else "skipped under the failed director(ies)",
                    )
                if disc_errors.unscreened:
                    # Reported separately, and not folded into the line above,
                    # because a screening fault can skip pruning with NO
                    # directory error to explain it — that line would never
                    # fire and the skip was the one thing about this run an
                    # operator could not see anywhere (PR #31 review). The
                    # screening warning further down talks about deletions held
                    # back, which is a different decision on a different set.
                    logger.warning(
                        "index run %s: %d director(ies) had unloadable ignore "
                        "rules — orphan point pruning %s, because a file that "
                        "the missing rules wrongly excluded is absent from the "
                        "walk without being absent from disk; a run that loads "
                        "them prunes normally",
                        run_id,
                        len(disc_errors.unscreened),
                        "skipped for the whole run"
                        if unscreened is None
                        else "skipped under those director(ies)",
                    )
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
                    candidates = gitfast.CandidatePathSet(set(candidates) | drifted)

        diff = await asyncio.to_thread(
            hashdiff.partition,
            root_path,
            discovered,
            stored,
            candidates=candidates,
            discovery_errored=disc_errors.files,
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

        # A directory-level discovery error means part of the tree was never
        # walked, so "absent from discovery" is not deletion evidence *there*.
        # Stale chunks beat purged live files; a clean next run detects real
        # deletions again. Scoped to the unwalked subtrees rather than the
        # whole project: a directory that is permanently unreadable (root-owned,
        # dead mount) would otherwise freeze deletion processing for every
        # other path forever, and deletions outside it were provable all along
        # — os.walk loses only the failing directory's subtree. When the
        # failures cannot be attributed to a subtree (`unwalked is None`) the
        # old project-wide suppression is exactly right and still applies
        # (ADR-54).
        files_deleted = 0
        # Suppressed deletions are unverified files, not deleted ones: their
        # stored hash may be stale and this run never read them. They must be
        # re-queued individually (the caller re-pends `failed_paths`) — a
        # scoped retry matches its candidate set EXACTLY, with no prefix
        # expansion, so re-pending only the errored directory would make the
        # retry carry every stored child forward as unchanged and clear the
        # pending row, stranding the edit until a manual full run. Any path in
        # here that really was deleted is simply re-detected and deleted by the
        # next run, which then clears its pending row.
        unverified: tuple[str, ...] = ()
        # Deletions held back because the ignore rules that decide membership
        # could not be loaded (ADR-58). Kept OUT of `unverified` on purpose:
        # that name feeds the re-pend and the ADR-56 quarantine, and these
        # paths need neither. Discovery walks the entire tree on every run, and
        # deletion is recomputed from that walk — so the first run that reads
        # the .gitignore again reaches the right answer on its own, with no
        # retry queued and no backlog to drain. What it does drive is the
        # anchor: a stored file wrongly excluded this run is a file whose hash
        # may now be stale, which is exactly what the anchor must not skip past.
        screening_held: tuple[str, ...] = ()
        if diff.deleted:
            if unwalked is None:
                provable: tuple[str, ...] = ()
                unverified = tuple(diff.deleted)
            elif unwalked:
                provable = tuple(p for p in diff.deleted if not _under(p, unwalked))
                unverified = tuple(p for p in diff.deleted if _under(p, unwalked))
            else:
                provable = tuple(diff.deleted)
            # Second filter, applied only to what the first one already cleared
            # — so a path blocked by both is counted once, as unverified.
            if unscreened is None:
                screening_held = provable
                provable = ()
            elif unscreened:
                screening_held = tuple(p for p in provable if _under(p, unscreened))
                provable = tuple(p for p in provable if not _under(p, unscreened))
            if provable:
                await asyncio.to_thread(store.delete_file_chunks, project_id, provable)
                state.delete_files(conn, project_id, provable)
                files_deleted = len(provable)

        # Which unverified paths the caller should re-pend (ADR-56). Note what
        # this block does NOT touch: `provable`, `files_deleted`, and the
        # deletion above already happened on ADR-51/54's terms alone. Quarantine
        # decides only whether a path is queued for RETRY — never whether it is
        # deleted — so a quarantined subtree's chunks keep serving search
        # forever, exactly as they did before this change.
        repend = unverified
        if unverified and quarantined:
            if unwalked is None:
                # No prefix describes what went unseen, so every stored path is
                # unverified and the choice is all-or-nothing. Stand down only
                # when EVERY directory reported this run is quarantined: one
                # fresh failure means this run's doubt is new, and new doubt is
                # exactly what the retry exists for.
                if disc_errors.dirs and reported_now <= quarantined:
                    repend = ()
            else:
                blocked = tuple(p for p in unwalked if p in quarantined)
                if blocked:
                    repend = tuple(p for p in unverified if not _under(p, blocked))
        suppressed = len(unverified) - len(repend)
        if suppressed:
            logger.warning(
                "index run %s: %d path(s) under %d quarantined director(ies) are "
                "no longer being re-queued for retry — their indexed content is "
                "KEPT and still searchable, not deleted, but it is no longer "
                "being re-verified either. Fix the director(ies) (or narrow the "
                "project's index scope) and the next run re-hashes them "
                "automatically",
                run_id,
                suppressed,
                len(reported_now & quarantined),
            )

        if orphan_paths:
            # Drift cleanup: points whose file_path is neither tracked in
            # state nor on disk. No state rows correspond to them, so this
            # only prunes Qdrant; without it the drift gate keeps firing.
            await asyncio.to_thread(store.delete_file_chunks, project_id, orphan_paths)

        # Hash-time failures (H7 carry-forward) are per-file failures too:
        # the file's true state is unknown this run, its stored hash may be
        # stale, and — unlike chunk/embed failures — it never entered
        # to_index, so without this it would be invisible to every guard
        # below and silently stranded by the next fast path.
        #
        # `diff.errored` carries both stages: hashdiff's own read failures and
        # the file-level discovery failures handed to it as
        # `discovery_errored`. They are disjoint (a file discovery could not
        # screen never reaches `discovered`, so it is never hashed), and the
        # stage belongs in the message — an operator chasing a permission
        # error should not be pointed at hashing when the stat failed.
        discovery_failed = {path for path, _ in disc_errors.files}
        hash_errors = [
            (
                path,
                f"{'discovery' if path in discovery_failed else 'hash'} failed: {msg}",
            )
            for path, msg in diff.errored
        ]
        # Dir-level discovery errors are run failures too: an unseen subtree
        # leaves its files' true state unknown, so they must block anchor
        # advance exactly like hash failures. (File-level discovery errors
        # already arrive here via diff.errored / discovery_errored.)
        #
        # Kept in their OWN list rather than folded into hash_errors, because
        # they are the only entries here that are not file paths — a directory
        # rel, or the `<root>` sentinel. They belong in run_file_errors (an
        # operator needs to see which subtree failed) but must never reach
        # `failed_paths`, which the caller re-pends as file rows: a scoped
        # retry matches its candidate set exactly, so a directory candidate
        # matches nothing, indexes nothing, and then clears its own pending
        # row. The stored files under that subtree are already re-queued the
        # right way, as themselves, via `unverified`.
        #
        # A quarantined directory says so here, so the run row explains its own
        # backlog: an operator looking at "why is this failing every run and
        # why did the awaiting-reindex list empty out?" gets both halves in one
        # place, rather than having to correlate a log line with a table.
        dir_errors = [
            (
                dir_rel,
                f"discovery failed: {msg}"
                + (
                    " — quarantined after "
                    f"{prior_counts.get(dir_rel, 0) + 1} consecutive failure(s):"
                    " stored paths under it are no longer re-queued for retry."
                    " They are NOT deleted; a run that walks this directory"
                    " again re-hashes them"
                    if dir_rel in quarantined
                    else ""
                ),
            )
            for dir_rel, msg in disc_errors.dirs
        ]
        # Screening faults (ADR-58). Recorded so an operator can see them, and
        # kept out of `total_failed` / `all_failed` / `run_error`, because
        # nothing failed to be indexed — the run's results may just be the
        # wrong shape.
        #
        # The synthetic key prefixes are load-bearing, not decoration.
        # `record_file_errors` is INSERT OR REPLACE on (run_id, path), and the
        # r-without-x case that motivates this whole change produces a REAL
        # `errors.files` row for `<dir>/.gitignore` at the same time as the
        # screening row for `<dir>` — different questions, and on a root
        # .gitignore the keys would otherwise be one character apart. Round 6
        # of PR #24 already lost an operator's errno to exactly this collision,
        # with a generic sentinel overwriting the real one. A real rel path can
        # never begin with `<`, so these cannot collide with anything.
        screening_errors = [
            (f"<screening>:{dir_rel}", f"screening degraded: {msg}")
            for dir_rel, msg in disc_errors.unscreened
        ] + [
            (f"<identity>:{dir_rel}", f"screening degraded: {msg}")
            for dir_rel, msg in disc_errors.unidentified
        ]
        if file_errors or hash_errors or dir_errors or screening_errors:
            state.record_file_errors(
                conn,
                run_id,
                [*file_errors, *hash_errors, *dir_errors, *screening_errors],
            )
        if screening_errors:
            logger.warning(
                "index run %s: %d discovery screening fault(s) — this run's "
                "results may be over- or under-inclusive under %s. %d deletion(s) "
                "held back as a result; indexed content is untouched and a run "
                "that loads the rules again reaches the right answer on its own",
                run_id,
                len(screening_errors),
                ", ".join(
                    sorted(
                        {
                            dir_rel or "<root>"
                            for dir_rel, _ in (
                                *disc_errors.unscreened,
                                *disc_errors.unidentified,
                            )
                        }
                    )
                ),
                len(screening_held),
            )
        # Fold this run's directory result into the ledger. Deliberately AFTER
        # the file loop and the re-admission above: a crash before this point
        # leaves the ledger untouched, so the next run redoes the union rather
        # than believing a recovery that never got applied.
        state.reconcile_unwalkable_dirs(
            conn,
            project_id,
            reported=disc_errors.dirs,
            cleared=recovered,
            quarantine_after=unwalkable_quarantine_runs,
        )
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
        # dir_errors counts here as well as hash_errors: a root that could not
        # be scanned at all produces ONLY a dir error, and that is the exact
        # whole-tree outage this check exists to catch.
        hash_all_failed = (
            bool(hash_errors) or bool(dir_errors)
        ) and diff.verified + diff.skipped == 0
        all_failed = chunk_embed_all_failed or hash_all_failed
        total_failed = len(file_errors) + len(hash_errors) + len(dir_errors)
        # `total_failed` counts directory entries too, so the file-count
        # wording lies for the shape that produces ONLY dir errors — an
        # unreadable or empty root reported "all 1 files failed" with zero
        # files failed (PR #24 review).
        if not all_failed:
            run_error = None
        elif dir_errors and not file_errors and not hash_errors:
            run_error = (
                f"discovery failed for {len(dir_errors)} location(s); "
                "no files could be verified"
            )
        else:
            run_error = f"all {total_failed} files failed"
        # Computed BEFORE `finish_run` so the run row can carry the verdict
        # rather than leave a reader to reconstruct it (ADR-58). Both drains
        # below are gated on it, and so is `state.last_full_run_was_clean`,
        # which used to re-derive it as `files_failed == 0` — true only while
        # every term here was a file-level failure, which stopped being true
        # when dir errors were added (PR #30 round 2) and again with
        # `screening_held`, which `files_failed` deliberately excludes because
        # nothing failed to index. A degraded run therefore reported
        # `files_failed == 0` while draining nothing, and the promotion policy
        # counted a dirty set it could not shrink (PR #31 review). One writer,
        # one persisted answer.
        #
        # `screening_held` earns its place here for the same reason as the
        # others: a stored file that a lost .gitignore pushed out of the
        # results is a file this run did not hash, so its stored hash may be
        # stale, and anchoring past it would leave the next fast path with no
        # reason to look at it again. The test is on what was actually held
        # back rather than on the fault itself, so an unreadable .gitignore
        # that excludes nothing currently indexed — the common case — still
        # advances the anchor instead of latching the project onto full walks.
        clean_run = (
            not file_errors
            and not hash_errors
            and not dir_errors
            and not screening_held
        )
        # ADR-60. `clean_run` above is the run's HEALTH verdict and keeps every
        # term it had. The two drains below ask a different question — "which
        # files is this run unsure about?" — and its honest answer is a SET,
        # not a boolean.
        #
        # Every stored path lands in exactly one bucket, and this names the
        # ones left in doubt: hashed-then-failed-to-index (`file_errors`),
        # failed to hash or to be screened (`hash_errors`, which carries the
        # stale stored hash forward), never seen because a directory could not
        # be walked (`unverified`), and held back from deletion because the
        # ignore rules that decide membership never loaded (`screening_held`).
        # The enumeration is exhaustive rather than hopeful: `diff.deleted` is
        # `stored - seen`, and the deletion block above partitions all of it
        # into `unverified` / `screening_held` / `provable`, the last of which
        # leaves `stored` entirely.
        unresolved = frozenset(p for p, _ in (*file_errors, *hash_errors)).union(
            unverified, screening_held
        )
        # So the gate becomes "advance when the doubt can be NAMED", not
        # "advance when there is none". What the old gate protected — the next
        # fast path must never skip a file whose state is unknown — is
        # delivered by recording those paths in `dirty_paths`, which already
        # means "owed re-admission next run" and which the H1 union above feeds
        # straight back into the candidate set. Blocking instead froze
        # `last_indexed_commit` for the whole project on one unreadable
        # directory (or one unreadable FILE, which leaves no ledger row at
        # all), and `compute_candidates` then diffed against an ever-older
        # commit forever (issue #28).
        #
        # Two conditions still block completely, and they are the point:
        #
        # * `unwalked is None` — the `<root>` sentinel, a path outside the
        #   root, or `follow_symlinks`. `unverified` does technically enumerate
        #   the paths even then, so this is a judgement rather than a
        #   necessity: under `<root>` that enumeration is the ENTIRE index, so
        #   advancing would write every path in the project into a JSON column
        #   on every run to buy a next run that hashes everything anyway. The
        #   walk proved nothing; ADR-57's trigger 1 already forces a full walk
        #   for these projects.
        # * `all_failed` — §3.2 rule 4 is untouched. A run that finishes
        #   `failed` never advances the anchor.
        #
        # Note this SUBSUMES the old rule rather than sitting beside it: when
        # `clean_run` is true, all four terms are empty and `unwalked` is `()`,
        # so `unresolved` is empty and both branches do exactly what they did
        # before. A healthy project cannot observe this change.
        may_advance = not all_failed and unwalked is not None
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
            clean=clean_run,
            error=run_error,
        )
        if paths is not None:
            # Scoped runs never advance the anchor, so the H1 dirty-set write
            # below never fires for them — persist what they DID index here,
            # or content indexed only by scoped runs is invisible to the H1
            # re-admission and a revert-while-unwatched stays stale forever.
            # Union-only: can only widen the next fast path's candidate set
            # (§3.2 rule-1 safe).
            failed = {path for path, _ in file_errors}
            indexed_ok = [rel for rel in to_index if rel not in failed]
            if indexed_ok:
                state.add_dirty_paths(conn, project_id, indexed_ok)
        # The anchor may advance past a failure whose paths this run can name,
        # because those paths ride along in the same write (ADR-60). It may not
        # advance past one it cannot: `may_advance` is computed above, next to
        # the `unresolved` set it travels with.
        #
        # Tracked rather than re-derived for the log below: `may_advance` says
        # the run was ALLOWED to record a position, not that it did. A scoped
        # run has no anchor to write (`head is None`) and is not a full walk
        # (`candidates is not None`), so it takes neither branch however clean
        # it was — claiming otherwise would be a log line that lies about the
        # commonest run in the service.
        position_recorded = False
        if head is not None and may_advance:
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
            state.set_last_indexed_commit(
                conn, project_id, head, dirty_paths=set(new_dirty) | unresolved
            )
            position_recorded = True
        elif candidates is None and may_advance:
            # FULL walk with no anchor to record — a non-git root, or git
            # unavailable. `set_last_indexed_commit` is the only other writer
            # that shrinks the H1 dirty set, and it needs a commit, so without
            # this the union in `add_dirty_paths` only ever grows: on a non-git
            # watched project every scoped run adds what it indexed and nothing
            # ever removes it, and indexer's H1 re-admission unions the whole
            # accumulation back into the next scoped candidate set. It would
            # creep toward "re-hash every file ever indexed" on every watcher
            # run.
            #
            # Safe here and only here because a full walk hashed every
            # discovered file — so the only stored hashes still in doubt are
            # the ones it could not reach, which is precisely `unresolved`.
            # Writing that (rather than clearing outright) is what lets this
            # branch run at all on a project with a contained fault; before
            # ADR-60 it was gated on `clean_run` and a single permanently
            # unreadable path pinned the set forever (issue #28).
            state.set_dirty_paths(conn, project_id, unresolved)
            position_recorded = True
        if unresolved and position_recorded:
            # The visibility half of issue #28: the old behaviour was a
            # `last_indexed_commit` that silently stopped moving, with nothing
            # anywhere saying why. Say plainly that the position advanced and
            # what it is carrying, so "it moved despite errors" is something an
            # operator can read rather than infer. Count only — the paths are
            # at DEBUG below, per ADR-25 exposure.
            #
            # Deliberately NOT reported against `total_failed`: screening-held
            # deletions are carried here but excluded from that count (nothing
            # failed to index), so pairing the two printed "despite 0
            # failure(s)" on exactly the run this line exists to explain. The
            # carried count is the honest one — it is the number of paths this
            # sentence is actually about.
            logger.info(
                "index run %s: index position recorded with %d path(s) still "
                "unverified — they are carried forward for re-hashing, so none "
                "of them is skipped and the next run re-examines every one",
                run_id,
                len(unresolved),
            )
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
                ", ".join(
                    path for path, _ in (*file_errors, *hash_errors, *dir_errors)
                ),
            )
        return IndexResult(
            project_id=project_id,
            run_id=run_id,
            files_total=len(discovered),
            files_indexed=len(to_index) - len(file_errors),
            files_deleted=files_deleted,
            chunks_written=chunks_written,
            fast_path_used=fast_path_active,
            candidate_count=len(candidates)
            if fast_path_active and candidates is not None
            else None,
            files_failed=total_failed,
            # `unverified` is a retry list, not an error list: it carries no
            # run_file_errors row and is not counted in files_failed (nothing
            # failed on those paths — the run simply never reached them).
            # `dir_errors` is deliberately absent: every entry here is re-pended
            # as a file row by the caller, and a directory or the `<root>`
            # sentinel can never match a scoped run's exact candidate set.
            failed_paths=(
                *(path for path, _ in (*file_errors, *hash_errors)),
                *repend,
            ),
            unverified_suppressed=suppressed,
            discovery_degraded=len(screening_errors),
        )
    except BaseException as exc:
        # BaseException: CancelledError (e.g. server shutdown) must also mark
        # the run failed, or it would sit "running" forever.
        state.finish_run(conn, run_id, "failed", error=str(exc) or type(exc).__name__)
        raise


def _read_text(root_path: str, rel: str) -> str:
    from pathlib import Path

    return (Path(root_path) / rel).read_text(encoding="utf-8", errors="replace")
