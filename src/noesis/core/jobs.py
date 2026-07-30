"""Job manager — one place both adapters launch and inspect index runs.

Part of the Core Engine's "Job/State manager" box (§3.1). REST
(`POST /projects`, `POST /projects/{id}/reindex`) and the MCP `reindex`
tool must behave identically (two thin adapters over one core), so this
module owns the single launch path: open a run row, hand back ids
immediately, and index in a background task tracked in ``ctx.jobs`` so the
app lifespan can cancel orphans on shutdown.

M8 additions (ADR-40): watcher-scoped launches (``paths`` narrows the
candidate set, ``triggered_by`` records provenance), in-memory live
progress per run (``ctx.progress``, read by the dashboard — deliberately
not persisted: live progress of a dead process is meaningless), and
pending-change clearing once a run has examined the files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from noesis.core import state
from noesis.core.config import IndexingSettings
from noesis.core.indexer import execute_run, is_attributable_prefix

logger = logging.getLogger(__name__)


class _ContextLike(Protocol):
    """The AppContext attributes this module needs (duck-typed to avoid an
    import cycle: app.py builds the MCP server, which uses this module)."""

    conn: Any
    store: Any
    embedder: Any
    jobs: dict[str, asyncio.Task]
    progress: dict[str, dict[str, Any]]
    git_fast_path: bool
    embed_batch_size: int
    indexing: IndexingSettings


def _promotion_reason(
    ctx: _ContextLike, project_id: str, paths: Sequence[str]
) -> str | None:
    """Why this scoped run should become a full walk instead — or None.

    Every automated trigger in the service is scoped (``watcher``'s two, and
    the dashboard's two), and only a FULL run drains the H1 dirty set or
    advances the git anchor: ``clear_dirty_paths`` needs ``candidates is None``
    and ``set_last_indexed_commit`` needs a commit, neither of which a scoped
    run can produce (ADR-40). So without a promotion rule a watched project
    never gets a full walk again after its first one, and both the dirty set
    and any unwalkable-directory state stay latched (issue #25).

    Three triggers, cheapest test first. Each is independently disableable with
    ``0`` — the whole policy off restores exactly the pre-ADR-57 behaviour.
    """
    cfg = getattr(ctx, "indexing", None) or IndexingSettings()
    unwalkable = state.list_unwalkable_dirs(ctx.conn, project_id)
    # 1. An unattributable discovery failure (a vanished root, a dead mount)
    #    leaves EVERY stored path unverified. A scoped run cannot make progress
    #    on that — its candidate set is by definition narrower than the damage
    #    — so the only useful next run is a full one.
    if any(not is_attributable_prefix(row["dir_path"]) for row in unwalkable):
        return "an unattributable discovery failure leaves the whole index unverified"
    # 2. Enough scoped runs have gone by that the drains are overdue.
    if cfg.promote_after_scoped_runs > 0:
        since = state.scoped_runs_since_full(ctx.conn, project_id)
        if since >= cfg.promote_after_scoped_runs:
            return f"{since} scoped run(s) since the last completed full walk"
    # 3. The candidate set is no longer narrow enough to be worth scoping.
    #    Discovery already stats and binary-sniffs every file on every run, so
    #    past this point a scoped run costs about what the full walk costs
    #    while delivering none of its drains.
    if cfg.promote_candidate_fraction > 0:
        indexed = state.indexed_file_count(ctx.conn, project_id)
        if indexed:
            # The H1 dirty set counts toward the threshold only when a
            # promotion could actually shrink it (PR #30 review). Both drains
            # are gated on `clean_run`, so after a run that reported ANY
            # failure the set is pinned — and counting a pinned set latches
            # this trigger permanently at the first crossing, promoting every
            # launch thereafter on a number promotion is structurally incapable
            # of moving. That is the same "a guard that can fire forever has no
            # exit" shape this module exists to remove, reached from the other
            # side.
            #
            # Keyed on whether the last FULL walk was clean, NOT on whether a
            # directory is unwalkable. `clean_run` has three terms and the
            # ledger only records one of them: a persistently unreadable *file*
            # pins the set identically with no ledger row, and latched this
            # trigger just the same until round 2 of the review caught it.
            #
            # "Full walk" and not "last run": a scoped run never hashes what is
            # outside its candidate set, so a permanently unhashable file leaves
            # every scoped run looking clean while the fault is still there.
            # Keying on the last run of any kind made this alternate — clean
            # scoped run promotes, the full walk hits the fault, next one does
            # not, and so on. Measured at shipped defaults over 12 launches:
            # healthy 1/12, broken directory 0/12, broken file 5/12 before any
            # guard, 3/12 keyed on the last run, 1/12 keyed on the last full
            # walk — matching the healthy baseline, which is the point.
            #
            # `pending` is still counted: those paths really did change, a
            # promotion really does clear them, and the cost argument holds.
            # Triggers 1 and 2 keep re-probing on their own cadence, so nothing
            # stops being revisited. Draining a pinned set at all is issue #28.
            candidates = set(paths)
            if state.last_full_run_was_clean(ctx.conn, project_id):
                candidates |= state.get_dirty_paths(ctx.conn, project_id)
            effective = len(candidates)
            if effective >= cfg.promote_candidate_fraction * indexed:
                return (
                    f"effective candidate set is {effective} of {indexed} indexed "
                    f"file(s)"
                )
    return None


def launch_index_run(
    ctx: _ContextLike,
    root_path: str,
    *,
    paths: Sequence[str] | None = None,
    triggered_by: str = "manual",
    force: bool = False,
) -> dict[str, str]:
    """Register (or re-open) the project and start indexing in the
    background. Returns the 202-style acceptance body shared verbatim by
    REST and MCP.

    Fails fast with ValueError before touching state when *root_path* is
    missing or not a directory (an agent gets the answer now instead of
    polling a background failure), and on the mixed-model guard
    (state.register_project). If a run is already ``running`` for this
    project, launches nothing (a second concurrent index would race on the
    same collection and state rows) and returns that run's id with
    ``status: "already_running"`` — a distinct status so a scoped/watcher
    caller can tell its launch was skipped and re-arm its retry (H3). A
    real launch returns ``status: "accepted"``.

    *force* is passed straight to :func:`execute_run`: it lets a discovery
    result of zero files count as deletion evidence when state still tracks
    files (ADR-55). Only an operator can make that call — nothing observable
    separates an emptied root from an unmounted one — so it is never set by
    the watcher, and the REST route is the only caller that exposes it. The
    MCP `reindex` tool deliberately does not: its caller is an agent."""
    if not os.path.isdir(root_path):
        raise ValueError(f"root_path is not an existing directory: {root_path!r}")
    project_id = state.register_project(ctx.conn, root_path, ctx.embedder.model_id)
    # Promote a scoped run to a full walk when the scoped path has stopped
    # paying for itself (ADR-57). Decided HERE, before the run row is opened,
    # because `paths` feeds both `execute_run` and the `clear_pending_changes`
    # below — setting it to None in one place keeps the run and its pending
    # clear describing the same thing. Doing it inside `execute_run` instead
    # would leave the caller clearing only the old scoped set after a walk that
    # examined everything.
    #
    # Never promoted under `force`: that flag lets an empty discovery result
    # purge the index, and applying it to a whole tree the caller had
    # deliberately scoped is not what the operator asserted. (Unreachable
    # today — the only `force` caller passes no paths — but the guard is the
    # cheap half of that pairing.)
    if paths is not None and not force:
        reason = _promotion_reason(ctx, project_id, paths)
        if reason is not None:
            logger.info(
                "project %s: promoting a %d-path scoped run to a full walk (%s)",
                project_id,
                len(paths),
                reason,
            )
            paths = None
    # Atomic check-and-insert (state.try_start_run, BEGIN IMMEDIATE): a plain
    # read-then-insert is race-free on one event loop but not across the
    # dual-transport deployment (HTTP + stdio MCP sharing this DB), where two
    # near-simultaneous launches could both pass the check and race two index
    # runs onto the same collection.
    run_id, created = state.try_start_run(
        ctx.conn, project_id, triggered_by=triggered_by, scoped=paths is not None
    )
    if not created:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "already_running",
        }
    # Cut point for the pending clear: rows re-dirtied after this survive.
    started_iso = datetime.now(timezone.utc).isoformat()
    ctx.progress[run_id] = {
        "files_done": None,
        "files_to_index": None,
        "chunks_written": 0,
        "monotonic_start": time.monotonic(),
    }

    def _on_progress(done: int, total: int, chunks: int) -> None:
        entry = ctx.progress.get(run_id)
        if entry is not None:
            entry["files_done"] = done
            entry["files_to_index"] = total
            entry["chunks_written"] = chunks

    async def _run() -> None:
        try:
            result = await execute_run(
                ctx.conn,
                ctx.store,
                ctx.embedder,
                root_path,
                project_id,
                run_id,
                batch_size=ctx.embed_batch_size,
                git_fast_path=ctx.git_fast_path,
                paths=paths,
                force=force,
                unwalkable_quarantine_runs=(
                    getattr(ctx, "indexing", None) or IndexingSettings()
                ).unwalkable_quarantine_runs,
                on_progress=_on_progress,
            )
        except Exception:
            # execute_run already marked the run failed; log for the operator.
            logger.exception("index run %s failed", run_id)
        else:
            # The run examined these files — clear their pending rows, but
            # only up to the launch timestamp so a file re-dirtied mid-run
            # stays pending (resolved toward re-examination, ADR-40).
            state.clear_pending_changes(
                ctx.conn, project_id, paths=paths, before=started_iso
            )
            # Files that failed per-file containment (ADR-41) go straight
            # back to pending: their state rows are stale, and without a
            # pending row the watcher would never auto-retry them — only a
            # manual full run would (PR #10 review). No hot loop: a retry
            # fires only on the next event/quiet cycle or manual action.
            if result.failed_paths:
                state.upsert_pending_changes(
                    ctx.conn,
                    project_id,
                    [(p, "modified") for p in result.failed_paths],
                )

    task = asyncio.create_task(_run())
    ctx.jobs[run_id] = task

    def _cleanup(_t: asyncio.Task) -> None:
        ctx.jobs.pop(run_id, None)
        ctx.progress.pop(run_id, None)

    task.add_done_callback(_cleanup)
    return {"project_id": project_id, "run_id": run_id, "status": "accepted"}


def run_progress(ctx: _ContextLike, run_id: str) -> dict[str, Any] | None:
    """Live progress for a running run: percent, counts, elapsed, ETA.
    None when the run is not tracked (finished, or another process's).
    ETA is naive linear extrapolation over files processed — honest enough
    for a progress bar, no smoothing pretence."""
    entry = ctx.progress.get(run_id)
    if entry is None:
        return None
    done = entry["files_done"]
    total = entry["files_to_index"]
    elapsed = time.monotonic() - entry["monotonic_start"]
    percent: float | None = None
    eta: float | None = None
    if done is not None and total is not None:
        percent = 100.0 if total == 0 else round(done / total * 100.0, 1)
        if done > 0 and total > done:
            eta = round(elapsed / done * (total - done), 1)
        elif total == done:
            eta = 0.0
    return {
        "files_done": done,
        "files_to_index": total,
        "chunks_written": entry["chunks_written"],
        "percent": percent,
        "elapsed_s": round(elapsed, 1),
        "eta_s": eta,
    }


async def index_status(ctx: _ContextLike, project_id: str) -> dict[str, Any]:
    """Latest run for a project, shaped identically for REST and MCP.

    A registered project with no runs yet reports ``never_indexed`` with
    the run fields nulled — a stable shape beats a shape-shifting one for
    agent consumers.

    Async so the quick ``state.*`` reads stay on the event loop (the shared
    sqlite conn is loop-owned; a worker-thread read can interleave inside
    another op's open BEGIN IMMEDIATE transaction) while only the
    synchronous Qdrant count round-trip is pushed to a worker thread.

    ``drift`` is reported only when the expected chunk total held still
    across that round-trip and still disagreed with the store. The cost of
    that rule is worth stating: while a run is actively committing, ``drift``
    reads False even if the store genuinely lost data — the first call after
    the run settles reports it. A false negative for a few seconds beats a
    false positive on every status poll during every index run.

    ``unwalkable_dirs`` / ``quarantined_dirs`` (ADR-56) are the coverage half
    of the same health picture: directories discovery could not walk, and the
    subset that has failed long enough for their contents to stop being
    re-queued. Both zero is the healthy state. Non-zero does not mean anything
    was deleted — nothing under an unwalked directory is ever purged — it means
    what is indexed there cannot be proven current."""
    # Index health: what the state DB expects vs what Qdrant actually holds.
    # A mismatch is drift — a vector store that lost data (external collection
    # wipe) while state still reports the files indexed. Surfaced so agents
    # and the dashboard can see it without a reindex; a full reindex heals it.
    expected_chunks = state.expected_chunk_total(ctx.conn, project_id)
    vector_count = await asyncio.to_thread(ctx.store.count_project_points, project_id)
    # Re-read after the count: the Qdrant round-trip is a scheduling point, so
    # a live index run can commit chunks between the two readings. A total
    # that MOVED across that window means neither reading is a stable
    # expectation, so there is nothing to compare the count against — the two
    # readings must agree with each other before a disagreement with the count
    # counts as evidence. (Asking only whether either reading happens to equal
    # the count is not the same test and does not catch the in-flight case:
    # `upsert_chunks` lands points before `upsert_file` commits the row, so
    # mid-run the count sits ABOVE the expected total and both readings differ
    # from it.) Report the fresher figure.
    expected_after = state.expected_chunk_total(ctx.conn, project_id)
    drift = expected_chunks == expected_after and expected_after != vector_count
    expected_chunks = expected_after
    # Coverage health, alongside the store health above (ADR-56). `drift` says
    # the store lost content it should hold; these say part of the TREE was
    # never read, so what is indexed for it may be stale and cannot be proven
    # otherwise. An agent searching a project is entitled to know that before
    # trusting an answer, which is why this rides the shared REST/MCP shape and
    # not just the dashboard.
    unwalkable, quarantined = state.count_unwalkable_dirs(ctx.conn, project_id)
    run = state.get_latest_run(ctx.conn, project_id)
    if run is None:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "never_indexed",
            "files_total": None,
            "files_changed": None,
            "chunks_written": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "expected_chunks": expected_chunks,
            "vector_count": vector_count,
            "drift": drift,
            "unwalkable_dirs": unwalkable,
            "quarantined_dirs": quarantined,
        }
    return {
        "project_id": project_id,
        "run_id": run["id"],
        "status": run["status"],
        "files_total": run["files_total"],
        "files_changed": run["files_changed"],
        "chunks_written": run["chunks_written"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "error": run["error"],
        "expected_chunks": expected_chunks,
        "vector_count": vector_count,
        "drift": drift,
        "unwalkable_dirs": unwalkable,
        "quarantined_dirs": quarantined,
    }
