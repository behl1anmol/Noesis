"""Machine-wide index-run cap (ADR-85): the cross-process guarantee, the
boundary, and the five call sites that were left untested.

``00db0aa`` shipped the cap at ``state.try_start_run`` with one test behind
it (``test_api.py::test_register_at_index_capacity_still_reports_the_registered_project``,
which pins only ``POST /projects``). This file covers the rest.

What this proves:

* **The cap is enforced in the database, not in a process.** This is the
  whole reason it lives inside ``try_start_run``'s ``BEGIN IMMEDIATE``
  transaction rather than in an ``asyncio.Semaphore``: the documented
  deployment runs an HTTP server and a stdio MCP process against ONE state
  DB, so an in-process counter would bound one of them and let the other
  proceed unaware — no cap at all. Two *separate* ``sqlite3`` connections to
  the same file are opened and the second is shown seeing the limit the
  first filled, and then seeing a slot the first freed. The same workload
  through the same two connections with ``max_concurrent=None`` is admitted,
  so the refusal is the cap and not the plumbing.
* **The boundary, probed on both sides with an explicit margin.**
  ``running == limit - 1`` admits (and the admitted run is the limit-th),
  ``running == limit`` refuses, and the refusal carries the live counts.
  Written against a named ``LIMIT`` so the numbers move together — a bare
  literal could be passing on an off-by-one that happens to line up.
* **A project's own live run still reads as ``already_running``**, not a
  refusal, even when the cap is met. Deliberate: it is a truthful answer
  about work already in flight, and the contrast against a *different*
  project refused at the same instant is asserted in one test so the two
  cannot drift apart.
* **Every remaining call site translates the refusal.** ``watcher._launch``
  re-arms and does not count (both halves asserted — the commit's stated
  reason for a typed exception is precisely that the ``already_running``
  STRING path would have counted a run that never started); REST
  ``/reindex`` and the dashboard's ``reindex-pending`` answer 429 with
  ``Retry-After``; MCP ``reindex`` raises ``ToolError`` naming
  ``[indexing] max_concurrent_index_runs``; and the two best-effort paths —
  ``core.dashboard.register_project(index_now=True)`` and
  ``set_project_flags``' catch-up run — swallow it, so the registration and
  the operator's flag change land with no run reported.

Why this tier: nothing here indexes anything. A capacity refusal happens
*before* any file is walked, so every test registers projects and occupies
slots directly through ``state`` and asserts on the refusal — no model, no
Qdrant round-trip beyond the in-memory client the ``AppContext`` needs, no
``_wait_done`` polling. FakeEmbedder + in-memory Qdrant, same offline
harness as ``test_api.py``.

Deliberately NOT tested here:

* ``POST /projects`` at capacity (202 + ``status: "capacity_reached"``,
  NOT 429, because it registers before it launches). Already covered by
  ``test_api.py``; duplicating it here would give two places to update and
  invite someone to "fix" the deliberate asymmetry.
* A genuine two-*process* race. These tests use two connections in one
  process, which exercises the mechanism that matters (SQLite's write lock
  and a shared ``index_runs`` table) but shares ``state._OWNER``, so the
  dead-owner liveness probe cannot be driven from here. Owner liveness is
  ``fail_orphaned_runs``' territory and is covered in ``test_state.py``.
* The default value of 4. The commit documents it as a judgment call, not
  a measurement; asserting it here would pin a number no measurement backs
  and would fail the moment an operator's config is read differently.
  ``test_config.py`` owns the config plumbing.
* Whether a refused caller eventually succeeds after real runs finish.
  That is the scheduler's behaviour over time, not the cap's, and the
  slot-freeing half of the cross-connection test already pins the only
  claim the cap makes about it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from qdrant_client import QdrantClient

from noesis.app import AppContext, create_app
from noesis.core import dashboard as core_dashboard
from noesis.core import state
from noesis.core.config import IndexingSettings
from noesis.core.embedder import FakeEmbedder
from noesis.core.state import IndexCapacityReached
from noesis.core.vectorstore import VectorStore
from noesis.core.watcher import WatcherManager
from noesis.mcp import build_mcp

MODEL = "fake-embedder-v1"


def _make_ctx(tmp_path) -> AppContext:
    """Offline context, same shape as ``test_api.make_client``'s."""
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    embedder = FakeEmbedder(dim=8)
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    return AppContext(conn=conn, store=store, embedder=embedder)


def _project(conn, tmp_path, name: str) -> tuple[str, str]:
    """Register a real (empty) directory; returns (project_id, root_path)."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return state.register_project(conn, root, MODEL), str(root)


def _live_runs(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM index_runs WHERE status = 'running'"
    ).fetchone()[0]


def _auto_runs(conn, project_id: str) -> int:
    return conn.execute(
        "SELECT COALESCE(SUM(auto_runs), 0) FROM watcher_stats WHERE project_id = ?",
        (project_id,),
    ).fetchone()[0]


@pytest.fixture()
def ctx(tmp_path):
    context = _make_ctx(tmp_path)
    yield context
    # A context owns its telemetry writer (ADR-59); close_runtime_context
    # does this in production, a fixture here.
    context.telemetry.close(timeout=5.0)


@pytest.fixture()
def client(ctx):
    with TestClient(create_app(ctx=ctx)) as tc:
        yield tc


def _cap_at_one(ctx, tmp_path) -> str:
    """Set the cap to a single run and occupy that slot with ANOTHER
    project's run, so what refuses the caller is the machine-wide cap and
    not ``try_start_run``'s per-project guard. Returns the occupying
    project's id. Same approach as
    ``test_api.py::test_register_at_index_capacity_still_reports_the_registered_project``.
    """
    ctx.indexing = IndexingSettings(max_concurrent_index_runs=1)
    other_id, _ = _project(ctx.conn, tmp_path, "occupier")
    run_id, created = state.try_start_run(ctx.conn, other_id)
    assert created, "the occupying run must really be live for the cap to bite"
    assert _live_runs(ctx.conn) == ctx.indexing.max_concurrent_index_runs
    return other_id


# -- the cross-process guarantee -----------------------------------------------


def test_cap_is_shared_across_separate_connections_to_one_db(tmp_path):
    """The cap must hold between two connections to the same DB file.

    This is the entire reason it is enforced inside ``try_start_run``'s
    ``BEGIN IMMEDIATE`` transaction instead of an ``asyncio.Semaphore``:
    the documented deployment runs an HTTP server and a stdio MCP process
    against one state DB, and an in-process counter would bound one of them
    while the other kept launching. Both directions are asserted — writer A
    fills the limit and writer B is refused, then A frees a slot and B can
    use it.

    Made non-vacuous by replaying the refused call through the SAME
    connection with ``max_concurrent=None``: it is admitted, so the refusal
    above came from the cap and not from a locked or mis-wired connection.
    """
    db = tmp_path / "shared.sqlite"
    conn_a = state.connect(db)
    state.init_db(conn_a)
    conn_b = state.connect(db)  # a second, independent connection
    try:
        LIMIT = 2
        p1, _ = _project(conn_a, tmp_path, "p1")
        p2, _ = _project(conn_a, tmp_path, "p2")
        p3, _ = _project(conn_a, tmp_path, "p3")

        # Connection A fills the limit.
        run1, created1 = state.try_start_run(conn_a, p1, max_concurrent=LIMIT)
        run2, created2 = state.try_start_run(conn_a, p2, max_concurrent=LIMIT)
        assert created1 and created2
        assert _live_runs(conn_b) == LIMIT, "B must see A's committed rows"

        # Connection B is refused by a cap it never counted itself.
        with pytest.raises(IndexCapacityReached) as excinfo:
            state.try_start_run(conn_b, p3, max_concurrent=LIMIT)
        assert excinfo.value.running == LIMIT
        assert excinfo.value.limit == LIMIT
        assert _live_runs(conn_a) == LIMIT, "a refusal must not insert a row"

        # Non-vacuous: the identical call on the identical connection is
        # admitted when uncapped, so the refusal was the cap.
        uncapped, created3 = state.try_start_run(conn_b, p3, max_concurrent=None)
        assert created3, "uncapped, the same workload on the same conn is admitted"
        state.finish_run(conn_b, uncapped, "done")  # back to LIMIT live

        # A slot freed through A becomes usable from B.
        assert _live_runs(conn_b) == LIMIT
        state.finish_run(conn_a, run1, "done")
        assert _live_runs(conn_b) == LIMIT - 1
        run3, created4 = state.try_start_run(conn_b, p3, max_concurrent=LIMIT)
        assert created4, "the slot A freed must be usable from B"
        assert run3 != run2
        assert _live_runs(conn_a) == LIMIT
    finally:
        conn_a.close()
        conn_b.close()


# -- the boundary --------------------------------------------------------------


def test_boundary_admits_the_limith_run_and_refuses_the_next(tmp_path):
    """``running == limit - 1`` admits; ``running == limit`` refuses.

    Probed on both sides with an explicit one-run margin and written
    against a named ``LIMIT``, never a bare literal: an off-by-one that
    happened to line up with a hard-coded 3 would be invisible.
    """
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    try:
        LIMIT = 3
        projects = [_project(conn, tmp_path, f"p{i}")[0] for i in range(LIMIT + 1)]

        # Fill to exactly one below the limit.
        for pid in projects[: LIMIT - 1]:
            _, created = state.try_start_run(conn, pid, max_concurrent=LIMIT)
            assert created
        assert _live_runs(conn) == LIMIT - 1, "the margin below the boundary"

        # Below side: at running == LIMIT - 1 the next call is ADMITTED, and
        # it is the limit-th run.
        _, created = state.try_start_run(
            conn, projects[LIMIT - 1], max_concurrent=LIMIT
        )
        assert created, "the limit-th run must be admitted, not refused"
        assert _live_runs(conn) == LIMIT

        # Above side: at running == LIMIT the next call is REFUSED.
        with pytest.raises(IndexCapacityReached) as excinfo:
            state.try_start_run(conn, projects[LIMIT], max_concurrent=LIMIT)
        assert excinfo.value.running == LIMIT
        assert excinfo.value.limit == LIMIT
        assert _live_runs(conn) == LIMIT, "the refusal inserted nothing"
        # The message is what an operator reads out of a log line.
        assert str(excinfo.value) == (
            f"index capacity reached: {LIMIT} runs already in flight, limit {LIMIT}"
        )
    finally:
        conn.close()


def test_own_live_run_reads_as_already_running_not_a_refusal(tmp_path):
    """At capacity, a project asking about ITS OWN live run gets
    ``(run_id, False)`` — "already running" — while a different project
    asking at the same instant is refused.

    Deliberate (ADR-85): "already running" is a truthful answer about work
    that is genuinely in flight, and turning it into a refusal would make
    a harmless duplicate request look like an overload. The two halves are
    asserted together so they cannot drift apart.
    """
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    try:
        LIMIT = 1
        mine, _ = _project(conn, tmp_path, "mine")
        theirs, _ = _project(conn, tmp_path, "theirs")

        run_id, created = state.try_start_run(conn, mine, max_concurrent=LIMIT)
        assert created
        assert _live_runs(conn) == LIMIT, "the cap is met"

        # Own project: no exception, and the id of the run already in flight.
        same_id, created_again = state.try_start_run(conn, mine, max_concurrent=LIMIT)
        assert (same_id, created_again) == (run_id, False)

        # A different project at the same instant: refused.
        with pytest.raises(IndexCapacityReached):
            state.try_start_run(conn, theirs, max_concurrent=LIMIT)
        assert _live_runs(conn) == LIMIT
    finally:
        conn.close()


# -- call site: the watcher ----------------------------------------------------


async def test_watcher_rearms_and_does_not_count_a_capacity_refusal(ctx, tmp_path):
    """``watcher._launch`` must treat a capacity refusal exactly like
    ``already_running``: re-arm the quiet-period trigger AND leave
    ``auto_runs`` alone.

    Both halves matter and are asserted separately. The commit that added
    the cap explains why it is a typed exception rather than a third status
    string: ``_launch`` decides whether to re-arm by matching the literal
    ``"already_running"``, so a new status would have fallen straight
    through — bumping ``auto_runs`` for a run that never started and leaving
    the pending files parked until the user happened to touch another file.
    A test that checked only the re-arm would miss exactly that.
    """
    occupier = _cap_at_one(ctx, tmp_path)
    pid, root = _project(ctx.conn, tmp_path, "watched")
    (tmp_path / "watched" / "a.py").write_text("x = 1\n")
    state.set_project_flags(ctx.conn, pid, watch_enabled=True, auto_reindex=True)
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])

    manager = WatcherManager(ctx, debounce_s=0.05, quiet_s=0.2)
    ctx.watcher = manager
    # No observer needed: _launch is the unit under test, driven directly
    # the same way _maybe_auto_reindex drives it.
    assert pid not in manager._last_event

    manager._launch(pid, ["a.py"])

    # Half 1: re-armed, so the quiet-period trigger will retry.
    assert pid in manager._last_event, (
        "a capacity refusal must re-arm the quiet-period trigger, or the "
        "pending files sit until the user touches another file"
    )
    # Half 2: nothing was counted, because nothing started.
    assert _auto_runs(ctx.conn, pid) == 0, "no run started, so auto_runs must not move"
    assert not ctx.conn.execute(
        "SELECT 1 FROM index_runs WHERE project_id = ?", (pid,)
    ).fetchall(), "the refused launch must not have opened a run row"
    # The occupying run is untouched, and the pending work is still pending.
    assert _live_runs(ctx.conn) == 1
    assert [p["path"] for p in state.list_pending_changes(ctx.conn, pid)] == ["a.py"]
    assert _auto_runs(ctx.conn, occupier) == 0


# -- call site: REST -----------------------------------------------------------


def test_reindex_at_capacity_answers_429_with_retry_after(ctx, client, tmp_path):
    """``POST /projects/{id}/reindex`` at capacity → 429 + ``Retry-After``.

    429 rather than 503 (ADR-84/85): the machine is busy and nothing failed,
    and a client that cannot tell "busy" from "down" retries the wrong way.
    Unlike ``POST /projects`` this endpoint commits nothing before it
    launches, so a bare refusal is the honest answer.
    """
    _cap_at_one(ctx, tmp_path)
    pid, _ = _project(ctx.conn, tmp_path, "target")

    resp = client.post(f"/projects/{pid}/reindex")

    assert resp.status_code == 429, resp.text
    assert resp.headers["Retry-After"] == str(
        IndexCapacityReached(running=1, limit=1).retry_after_seconds
    )
    assert "index capacity reached" in resp.json()["detail"]
    assert _live_runs(ctx.conn) == 1, "no run was started for the refused project"


def test_dashboard_reindex_pending_at_capacity_answers_429(ctx, client, tmp_path):
    """The dashboard's ``/api/projects/{id}/reindex-pending`` is the
    Reindex button; it launches nothing before it fails, so it answers 429
    with ``Retry-After`` like the REST route it mirrors.

    Contrast with ``core.dashboard.set_project_flags``' catch-up run and
    ``register_project(index_now=True)``, which swallow the same exception:
    those have an operator-visible side effect to preserve, this one does
    not.
    """
    _cap_at_one(ctx, tmp_path)
    pid, _ = _project(ctx.conn, tmp_path, "target")
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])
    (tmp_path / "target" / "a.py").write_text("x = 1\n")

    resp = client.post(f"/api/projects/{pid}/reindex-pending")

    assert resp.status_code == 429, resp.text
    assert resp.headers["Retry-After"] == str(
        IndexCapacityReached(running=1, limit=1).retry_after_seconds
    )
    assert "index capacity reached" in resp.json()["detail"]
    assert _live_runs(ctx.conn) == 1
    # The pending work survives the refusal — nothing examined it.
    assert [p["path"] for p in state.list_pending_changes(ctx.conn, pid)] == ["a.py"]


# -- call site: MCP ------------------------------------------------------------


async def test_mcp_reindex_at_capacity_raises_a_tool_error_naming_the_setting(
    ctx, tmp_path
):
    """MCP has no status codes, so the message IS the interface.

    The agent must learn three things it cannot infer: nothing was started
    (so it retries rather than polling for a run id that does not exist),
    the live counts, and the config key to raise. That last one is asserted
    verbatim — an agent-facing refusal that does not name
    ``[indexing] max_concurrent_index_runs`` leaves the operator guessing.
    """
    _cap_at_one(ctx, tmp_path)
    pid, _ = _project(ctx.conn, tmp_path, "target")
    mcp = build_mcp(lambda: ctx)

    async with Client(mcp) as mcp_client:
        with pytest.raises(ToolError) as excinfo:
            await mcp_client.call_tool("reindex", {"project_id": pid})

    message = str(excinfo.value)
    assert "[indexing] max_concurrent_index_runs" in message
    assert "nothing was started" in message
    assert "1 index runs are already in flight against a limit of 1" in message
    assert _live_runs(ctx.conn) == 1


# -- call site: the flags catch-up run -----------------------------------------


async def test_enabling_auto_reindex_at_capacity_still_sets_the_flag(ctx, tmp_path):
    """``core.dashboard.set_project_flags`` swallows the refusal too.

    Turning auto_reindex on with pending changes fires a catch-up run, and
    that run is best effort by design: the flag change the operator asked
    for must land even when the machine is busy elsewhere. The pending rows
    stay pending, so the watcher's quiet-period trigger picks them up later
    — nothing is lost, only deferred.

    Async only so that a broken cap fails this on the assertion it was
    written for (a second live run) rather than on ``launch_index_run``
    hitting ``asyncio.create_task`` with no loop.
    """
    _cap_at_one(ctx, tmp_path)
    pid, _ = _project(ctx.conn, tmp_path, "target")
    (tmp_path / "target" / "a.py").write_text("x = 1\n")
    state.upsert_pending_changes(ctx.conn, pid, [("a.py", "modified")])

    summary = core_dashboard.set_project_flags(ctx, pid, auto_reindex=True)

    assert summary is not None
    assert state.get_project(ctx.conn, pid)["auto_reindex"] == 1, (
        "the operator's flag change must survive a busy machine"
    )
    assert _live_runs(ctx.conn) == 1, "no catch-up run was started"
    assert [p["path"] for p in state.list_pending_changes(ctx.conn, pid)] == ["a.py"]


# -- call site: register with index_now ----------------------------------------


async def test_register_with_index_now_at_capacity_stays_best_effort(ctx, tmp_path):
    """``core.dashboard.register_project(index_now=True)`` must still
    register.

    By the time the cap is hit the registration has already committed, so
    failing the whole request would leave the operator with a registered
    project and an error. It reports no run instead — the same outcome as
    ``index_now=False``, which the UI already renders, and the Reindex
    button covers the retry.

    Async because the uncapped companion below launches a real run, and
    both must run on the same kind of context for the comparison to mean
    anything.
    """
    _cap_at_one(ctx, tmp_path)
    root = tmp_path / "added"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")

    result = core_dashboard.register_project(ctx, str(root), index_now=True)

    assert result["run"] is None, "no run started, and none is claimed"
    project_id = result["project"]["id"]
    assert project_id, "the registration is real, so it is reported"
    listed = {row["id"] for row in state.list_projects(ctx.conn)}
    assert project_id in listed
    assert _live_runs(ctx.conn) == 1, "the occupier's run is still the only one"
    assert not ctx.conn.execute(
        "SELECT 1 FROM index_runs WHERE project_id = ?", (project_id,)
    ).fetchall()


async def test_register_with_index_now_below_capacity_does_launch(ctx, tmp_path):
    """Companion to the test above, and the thing that makes it mean
    something: with one slot still free, the identical call DOES report an
    accepted run. Without this, ``run is None`` would pass just as well
    against a ``register_project`` that never launched anything at all.
    """
    ctx.indexing = IndexingSettings(max_concurrent_index_runs=2)
    occupier, _ = _project(ctx.conn, tmp_path, "occupier")
    state.try_start_run(ctx.conn, occupier)
    assert _live_runs(ctx.conn) == ctx.indexing.max_concurrent_index_runs - 1
    root = tmp_path / "added"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")

    result = core_dashboard.register_project(ctx, str(root), index_now=True)

    assert result["run"] is not None, "a free slot must produce a run"
    assert result["run"]["status"] == "accepted"
    # Drain the launched task so the run does not outlive the test.
    await ctx.jobs[result["run"]["run_id"]]
