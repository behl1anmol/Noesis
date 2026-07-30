"""SQLite state store — projects, per-file state, index runs.

WAL-mode single-file DB per the approved Overview §7 schema plus the
`last_indexed_commit` column from the expanded doc §3.7. All mutating
functions commit before returning; callers never manage transactions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id                  TEXT PRIMARY KEY,
  root_path           TEXT NOT NULL UNIQUE,
  embedding_model     TEXT NOT NULL,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL,
  last_indexed_commit TEXT
);

CREATE TABLE IF NOT EXISTS files (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES projects(id),
  path            TEXT NOT NULL,
  language        TEXT,
  content_hash    TEXT NOT NULL,
  chunk_count     INTEGER NOT NULL DEFAULT 0,
  last_indexed_at TEXT,
  UNIQUE(project_id, path)
);

CREATE TABLE IF NOT EXISTS index_runs (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES projects(id),
  status          TEXT NOT NULL CHECK(status IN ('queued','running','done','failed')),
  files_total     INTEGER,
  files_changed   INTEGER,
  chunks_written  INTEGER,
  fast_path_used  INTEGER,
  candidate_count INTEGER,
  started_at      TEXT,
  finished_at     TEXT,
  error           TEXT,
  triggered_by    TEXT,
  files_failed    INTEGER
);

CREATE TABLE IF NOT EXISTS pending_changes (
  project_id  TEXT NOT NULL REFERENCES projects(id),
  path        TEXT NOT NULL,
  event_type  TEXT NOT NULL CHECK(event_type IN ('created','modified','deleted')),
  detected_at TEXT NOT NULL,
  PRIMARY KEY (project_id, path)
);

CREATE TABLE IF NOT EXISTS run_file_errors (
  run_id TEXT NOT NULL REFERENCES index_runs(id),
  path   TEXT NOT NULL,
  error  TEXT NOT NULL,
  PRIMARY KEY (run_id, path)
);

CREATE TABLE IF NOT EXISTS query_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           TEXT NOT NULL,
  interface    TEXT NOT NULL CHECK(interface IN ('rest','mcp')),
  kind         TEXT NOT NULL CHECK(kind IN ('search','structural')),
  project_id   TEXT,
  channel      TEXT,
  reranked     INTEGER,
  latency_ms   REAL,
  result_count INTEGER
);

CREATE TABLE IF NOT EXISTS watcher_stats (
  project_id       TEXT NOT NULL REFERENCES projects(id),
  day              TEXT NOT NULL,
  events_seen      INTEGER NOT NULL DEFAULT 0,
  events_coalesced INTEGER NOT NULL DEFAULT 0,
  auto_runs        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (project_id, day)
);

CREATE TABLE IF NOT EXISTS app_settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Directories discovery could not walk, and for how long (ADR-56). One row
-- per failing location, keyed on exactly what DiscoveryErrors.dirs reports:
-- a root-relative prefix, the '<root>' sentinel, or an absolute path when the
-- failure could not be attributed to a subtree at all.
--
-- A table rather than a JSON column like projects.dirty_paths: this needs a
-- per-row counter and timestamps, has to be queryable for the dashboard, and
-- must not be a read-modify-write blob on the event loop.
CREATE TABLE IF NOT EXISTS unwalkable_dirs (
  project_id       TEXT NOT NULL REFERENCES projects(id),
  dir_path         TEXT NOT NULL,
  consecutive_runs INTEGER NOT NULL DEFAULT 0,
  first_seen_at    TEXT NOT NULL,
  last_seen_at     TEXT NOT NULL,
  last_error       TEXT NOT NULL,
  -- Set once consecutive_runs crosses the threshold: the stored paths this
  -- directory hides stop being re-queued for retry. NEVER a licence to delete
  -- them (ADR-51/54 still own that decision) — only to stop re-pending.
  quarantined_at   TEXT,
  PRIMARY KEY (project_id, dir_path)
);
"""

# Additive column migrations for DBs created by earlier milestones —
# CREATE TABLE IF NOT EXISTS never alters an existing table.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("projects", "last_indexed_commit", "TEXT"),
    ("index_runs", "fast_path_used", "INTEGER"),
    ("index_runs", "candidate_count", "INTEGER"),
    # M8: per-project watcher flags (both default off, ADR-40) and run
    # provenance/partial-failure fields (ADR-41).
    ("projects", "watch_enabled", "INTEGER NOT NULL DEFAULT 0"),
    ("projects", "auto_reindex", "INTEGER NOT NULL DEFAULT 0"),
    ("index_runs", "triggered_by", "TEXT"),
    ("index_runs", "files_failed", "INTEGER"),
    # ADR-42: per-project index config set at registration. NULL columns
    # mean "use the default" (index_languages NULL = all languages,
    # max_file_bytes NULL = DiscoveryConfig default). JSON-encoded lists.
    ("projects", "index_languages", "TEXT"),
    ("projects", "max_file_bytes", "INTEGER"),
    ("projects", "follow_symlinks", "INTEGER NOT NULL DEFAULT 0"),
    ("projects", "extra_ignores", "TEXT"),
    # H1: working-tree-dirty paths as of the last anchor advance (JSON list).
    # Re-admitted as candidates on the next run so a file dirty at run N is
    # re-examined at run N+1 even after being reverted to HEAD.
    ("projects", "dirty_paths", "TEXT"),
    # M7: which process owns a run, so crash recovery fails only DEAD runs —
    # a second process sharing the DB must not mark the first's live run
    # failed (and thereby disarm the concurrent-run guard).
    ("index_runs", "owner", "TEXT"),
    # ADR-57: 1 for a run scoped to an explicit candidate set, 0 for a full
    # walk. Needed to count scoped runs since the last full walk (the
    # promotion trigger) and, incidentally, to make the scoped path visible at
    # all: `candidate_count` is NULL for scoped runs, so the run row recorded
    # nothing about them. A NEW column deliberately — `candidate_count` and
    # `fast_path_used` already carry meanings the dashboard and usage page
    # depend on. NULL means "recorded before this column existed", which
    # `scoped_runs_since_full` reads as not-scoped.
    ("index_runs", "scoped", "INTEGER"),
)


def _boot_token() -> str:
    """A token stable within one OS boot and different across reboots, so an
    owner identity survives PID recycling across reboots. Linux exposes it
    directly; elsewhere we fall back to empty and rely on PID liveness alone
    (local single-machine service, ADR-25)."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _proc_start_time(pid: int) -> str:
    """Process start time (clock ticks since boot) from ``/proc/<pid>/stat``
    field 22 — distinguishes a recycled PID from the original one (PR review).
    Empty when unavailable (process gone, or non-Linux): liveness then
    degrades to a bare PID probe with a documented residual race."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    # comm (field 2) is parenthesised and may contain spaces/parens, so split
    # the tail after the final ')': tail[0] is state (field 3), and starttime
    # (field 22) is therefore tail[19].
    rparen = data.rfind(b")")
    if rparen == -1:
        return ""
    tail = data[rparen + 2 :].split()
    if len(tail) < 20:
        return ""
    return tail[19].decode("ascii", "replace")


# Per-process owner identity: <boot>:<pid>:<starttime>. Computed once — all
# three components are stable for the life of the process. starttime closes
# same-boot PID recycling: a reused PID belongs to a process with a different
# start time, so its stamped runs read as dead.
_OWNER = f"{_boot_token()}:{os.getpid()}:{_proc_start_time(os.getpid())}"


def _owner_alive(owner: str) -> bool:
    """True if the process that stamped *owner* is still running (M7). A
    different boot token means a reboot happened → dead. Same boot → probe the
    PID with signal 0, then confirm the live process's start time matches, so
    a PID recycled within one boot is not mistaken for the original owner."""
    parts = owner.split(":")
    if len(parts) < 2:
        return False
    boot, pid_s = parts[0], parts[1]
    start = parts[2] if len(parts) > 2 else ""
    if boot != _boot_token():
        return False
    try:
        pid = int(pid_s)
    except ValueError:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists but owned by another user — /proc/<pid>/stat still readable
    except OSError:
        return False
    # PID exists: guard against same-boot recycling. If we recorded a start
    # time and the live process reports a different one, the original owner is
    # dead and its PID was reused → treat as dead. If either is unavailable
    # (non-Linux), fall back to "alive" (the documented residual race).
    if start:
        current = _proc_start_time(pid)
        if current and current != start:
            return False
    return True


class MixedModelError(ValueError):
    """A project is already indexed with a different embedding model.
    Typed so adapters can map it to 409 without matching message text
    (PR #10 review)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Migrations run single-writer: the dual-transport deployment (HTTP +
    # stdio MCP on one DB) can call init_db concurrently, and an unguarded
    # check-then-ALTER races — the loser's ALTER raises 'duplicate column
    # name', a schema error busy_timeout cannot retry, aborting that
    # process's startup. BEGIN IMMEDIATE serializes the loop (the second
    # writer's table_info re-read then sees the committed column and
    # skips); the except is a backstop for a pre-lock writer running older
    # code, where skipping is correct — the column exists.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table, column, ddl_type in _MIGRATIONS:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc):
                        raise
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def register_project(
    conn: sqlite3.Connection, root_path: str | Path, embedding_model: str
) -> str:
    resolved = str(Path(root_path).resolve())
    row = conn.execute(
        "SELECT id, embedding_model FROM projects WHERE root_path = ?", (resolved,)
    ).fetchone()
    if row is not None:
        if row["embedding_model"] != embedding_model:
            raise MixedModelError(
                f"project at {resolved} is indexed with model "
                f"{row['embedding_model']!r}; refusing to serve mixed-model state. "
                f"Re-register requires a full re-index with {embedding_model!r}."
            )
        return row["id"]
    project_id = uuid.uuid4().hex
    now = _now()
    try:
        conn.execute(
            "INSERT INTO projects (id, root_path, embedding_model, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (project_id, resolved, embedding_model, now, now),
        )
        conn.commit()
        return project_id
    except sqlite3.IntegrityError:
        # Concurrent registration of the same root_path lost the UNIQUE race
        # (L4). Return the winner's id idempotently instead of surfacing a raw
        # IntegrityError, re-checking the model just as the fast path does.
        conn.rollback()
        row = conn.execute(
            "SELECT id, embedding_model FROM projects WHERE root_path = ?", (resolved,)
        ).fetchone()
        if row is None:
            raise
        if row["embedding_model"] != embedding_model:
            raise MixedModelError(
                f"project at {resolved} is indexed with model "
                f"{row['embedding_model']!r}; refusing to serve mixed-model state. "
                f"Re-register requires a full re-index with {embedding_model!r}."
            ) from None
        return row["id"]


def get_project(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()


def list_projects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM projects ORDER BY created_at").fetchall()


def set_last_indexed_commit(
    conn: sqlite3.Connection,
    project_id: str,
    commit: str,
    *,
    dirty_paths: Iterable[str] | None = None,
) -> None:
    """Record the git fast-path anchor. Callers must invoke this only after
    a run completes successfully (§3.2 rule 4) — an anchor from a failed
    run would let the next fast path skip files the failed run never
    finished indexing.

    *dirty_paths* (when provided) records the working-tree-dirty set as of
    this anchor so the next run re-admits them as candidates (H1). Passing
    an empty iterable clears the stored set; None leaves it untouched."""
    if dirty_paths is None:
        conn.execute(
            "UPDATE projects SET last_indexed_commit = ?, updated_at = ? WHERE id = ?",
            (commit, _now(), project_id),
        )
    else:
        conn.execute(
            "UPDATE projects SET last_indexed_commit = ?, dirty_paths = ?,"
            " updated_at = ? WHERE id = ?",
            (commit, json.dumps(sorted(dirty_paths)), _now(), project_id),
        )
    conn.commit()


def add_dirty_paths(
    conn: sqlite3.Connection, project_id: str, paths: Iterable[str]
) -> None:
    """Union *paths* into the persisted dirty set (H1) WITHOUT touching
    ``last_indexed_commit``. Scoped (watcher) runs never advance the anchor,
    so :func:`set_last_indexed_commit` never fires for them — yet the dirty
    content they indexed must still be re-admitted by the next fast path.
    Union-only: it can only widen the candidate set (§3.2 rule-1 safe).

    The read and the write run in one ``BEGIN IMMEDIATE`` transaction. In
    practice ``try_start_run`` already serializes runs per project, even
    across processes, and this is only called from inside a run — but that
    invariant lives in another function, and a lost union here silently
    reopens the H1 gap this exists to close. Self-protecting beats
    depending on a caller's lock.

    Runs on the caller's thread — ``execute_run`` calls it directly from the
    event loop — and ``BEGIN IMMEDIATE`` takes SQLite's write lock at the
    first statement rather than at commit. Under a foreign writer (the other
    transport mid-run) it can therefore block the loop for up to the shared
    connection's ``busy_timeout`` of 5s, once per scoped run. Accepted
    deliberately: the alternative is a dedicated short-timeout connection
    like telemetry's, which DROPS its write when the lock is contended — and
    unlike a telemetry row, a dropped union is a correctness loss that
    silently reopens the H1 gap. A bounded stall beats silent data loss. It
    must not be moved to a worker thread either: the shared connection is
    loop-owned, and a worker-thread write interleaving inside another
    operation's open transaction is the hazard ``jobs.index_status``
    documents.
    """
    new = set(paths)
    if not new:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT dirty_paths FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            conn.rollback()
            return
        current: set[str] = (
            set(json.loads(row["dirty_paths"])) if row["dirty_paths"] else set()
        )
        conn.execute(
            "UPDATE projects SET dirty_paths = ?, updated_at = ? WHERE id = ?",
            (json.dumps(sorted(current | new)), _now(), project_id),
        )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def clear_dirty_paths(conn: sqlite3.Connection, project_id: str) -> None:
    """Empty the persisted H1 dirty set without touching the anchor.

    :func:`set_last_indexed_commit` is the only other writer that shrinks it,
    and it requires a commit — so a project with no git anchor (non-git root,
    or git unavailable) had no way to shrink the set at all and grew it
    without bound. Callers must have completed a clean FULL walk: that
    re-hashes every discovered file, so nothing is left needing H1
    re-admission. A blind write, unlike :func:`add_dirty_paths`'s
    read-modify-write, so it needs no explicit transaction."""
    conn.execute(
        "UPDATE projects SET dirty_paths = ?, updated_at = ? WHERE id = ?",
        (json.dumps([]), _now(), project_id),
    )
    conn.commit()


def get_dirty_paths(conn: sqlite3.Connection, project_id: str) -> frozenset[str]:
    """The dirty-path set persisted at the last anchor advance (H1). Empty
    when unset or the project is unknown."""
    row = conn.execute(
        "SELECT dirty_paths FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    if row is None or row["dirty_paths"] is None:
        return frozenset()
    return frozenset(json.loads(row["dirty_paths"]))


def get_file_states(conn: sqlite3.Connection, project_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT path, content_hash FROM files WHERE project_id = ?", (project_id,)
    ).fetchall()
    return {row["path"]: row["content_hash"] for row in rows}


def get_file_chunk_counts(conn: sqlite3.Connection, project_id: str) -> dict[str, int]:
    """Stored per-file chunk_count for a project — the number of points the
    vector store should hold for each file. Drift self-heal compares this,
    file by file, against the live per-file point counts in Qdrant."""
    rows = conn.execute(
        "SELECT path, chunk_count FROM files WHERE project_id = ?", (project_id,)
    ).fetchall()
    return {row["path"]: int(row["chunk_count"]) for row in rows}


def expected_chunk_total(conn: sqlite3.Connection, project_id: str) -> int:
    """Sum of stored per-file chunk_count for a project — the number of
    points the vector store should hold in total. COALESCE so a
    never-indexed project reads 0, never NULL."""
    row = conn.execute(
        "SELECT COALESCE(SUM(chunk_count), 0) AS n FROM files WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    return int(row["n"])


def indexed_file_total(conn: sqlite3.Connection) -> int:
    """Total tracked files across all projects — the wipe-signature check
    (a vector collection created while state already tracks indexed files)."""
    row = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()
    return int(row["n"])


def indexed_file_count(conn: sqlite3.Connection, project_id: str) -> int:
    """Files tracked for one project — the denominator the promotion trigger
    measures a scoped run's effective candidate set against (ADR-57)."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE project_id = ?", (project_id,)
    ).fetchone()
    return int(row["n"])


def upsert_file(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
    content_hash: str,
    language: str | None = None,
    chunk_count: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO files (id, project_id, path, language, content_hash,
                           chunk_count, last_indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id, path) DO UPDATE SET
          language = excluded.language,
          content_hash = excluded.content_hash,
          chunk_count = excluded.chunk_count,
          last_indexed_at = excluded.last_indexed_at
        """,
        (
            uuid.uuid4().hex,
            project_id,
            path,
            language,
            content_hash,
            chunk_count,
            _now(),
        ),
    )
    conn.commit()


def delete_files(
    conn: sqlite3.Connection, project_id: str, paths: Iterable[str]
) -> None:
    conn.executemany(
        "DELETE FROM files WHERE project_id = ? AND path = ?",
        [(project_id, p) for p in paths],
    )
    conn.commit()


def fail_orphaned_runs(conn: sqlite3.Connection) -> int:
    """Mark 'running' runs whose OWNING PROCESS IS DEAD as failed. Called at
    process startup to clear crash leftovers — the launch guard would
    otherwise return a dead run id forever, silently no-opping every future
    launch (PR #10 review).

    Owner-gated (M7): in the documented two-process deployment (HTTP + stdio
    MCP sharing the DB, m6 guide) a starting process must NOT fail the other
    process's LIVE run — that both loses the run and disarms the
    concurrent-run guard. Rows with no owner (pre-M7 DBs) predate ownership
    tracking and are treated as dead."""
    rows = conn.execute(
        "SELECT id, owner FROM index_runs WHERE status = 'running'"
    ).fetchall()
    dead = [
        row["id"] for row in rows if not row["owner"] or not _owner_alive(row["owner"])
    ]
    if not dead:
        return 0
    now = _now()
    conn.executemany(
        "UPDATE index_runs SET status = 'failed', error = 'interrupted',"
        " finished_at = ? WHERE id = ?",
        [(now, run_id) for run_id in dead],
    )
    conn.commit()
    return len(dead)


def start_run(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    triggered_by: str = "manual",
    scoped: bool = False,
) -> str:
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO index_runs (id, project_id, status, started_at, triggered_by,"
        " owner, scoped) VALUES (?, ?, 'running', ?, ?, ?, ?)",
        (run_id, project_id, _now(), triggered_by, _OWNER, int(scoped)),
    )
    conn.commit()
    return run_id


def try_start_run(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    triggered_by: str = "manual",
    scoped: bool = False,
) -> tuple[str, bool]:
    """Atomically open a run — or return the one already running.

    ``BEGIN IMMEDIATE`` takes SQLite's write lock up front, so the
    running-check and the insert are a single unit even across processes:
    the documented dual-transport deployment (HTTP server + stdio MCP on
    one DB) cannot both pass a read-then-insert guard and race two runs
    onto the same collection. Stale ``running`` rows whose owning process
    is dead are failed here (same owner probe as ``fail_orphaned_runs``),
    so a crashed co-process can never jam the guard until a restart.

    Returns ``(run_id, created)`` — ``created`` False means a live run
    already exists and ``run_id`` is that run's id.

    *scoped* records whether this run was given an explicit candidate set,
    written at INSERT rather than at ``finish_run`` so a run that crashes still
    counts toward the promotion trigger (ADR-57) — otherwise a project that
    keeps dying mid-run would never promote.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT id, owner FROM index_runs WHERE project_id = ?"
            " AND status = 'running' ORDER BY started_at DESC, rowid DESC",
            (project_id,),
        ).fetchall()
        alive: list[sqlite3.Row] = []
        dead: list[sqlite3.Row] = []
        for row in rows:
            (alive if (row["owner"] and _owner_alive(row["owner"])) else dead).append(
                row
            )
        if dead:  # dead owners: fail them now, exactly like fail_orphaned_runs —
            # even when an alive run also exists, or a crashed co-process's row
            # would sit "running" until the next restart's fail_orphaned_runs
            now = _now()
            conn.executemany(
                "UPDATE index_runs SET status = 'failed', error = 'interrupted',"
                " finished_at = ? WHERE id = ?",
                [(now, r["id"]) for r in dead],
            )
        if alive:
            if dead:
                conn.commit()  # persist the dead-row cleanup; releases the lock
            else:
                conn.rollback()  # nothing written; release the lock
            return alive[0]["id"], False
        run_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO index_runs (id, project_id, status, started_at,"
            " triggered_by, owner, scoped) VALUES (?, ?, 'running', ?, ?, ?, ?)",
            (run_id, project_id, _now(), triggered_by, _OWNER, int(scoped)),
        )
        conn.commit()
        return run_id, True
    except BaseException:
        conn.rollback()
        raise


def get_latest_run(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
    """Most recent index run for a project — the M6 ``get_index_status``
    surface. Ordered by started_at (ISO-8601 UTC, lexicographically
    sortable); ties broken by rowid (insertion order)."""
    return conn.execute(
        "SELECT * FROM index_runs WHERE project_id = ?"
        " ORDER BY started_at DESC, rowid DESC LIMIT 1",
        (project_id,),
    ).fetchone()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    *,
    files_total: int | None = None,
    files_changed: int | None = None,
    chunks_written: int | None = None,
    fast_path_used: bool | None = None,
    candidate_count: int | None = None,
    files_failed: int | None = None,
    error: str | None = None,
) -> None:
    """Close out a run row: status, counters, finish time, error.

    The counters are written with ``COALESCE(?, col)`` so passing ``None``
    leaves the stored value alone instead of erasing it. That matters because
    this can legitimately be called **twice** for one run: ``execute_run``
    writes ``"done"`` with a full set of counters, and only afterwards does its
    remaining bookkeeping — ``add_dirty_paths`` (which takes SQLite's write
    lock under ``BEGIN IMMEDIATE`` and raises ``OperationalError`` past the
    5s ``busy_timeout``), the anchor write, and a ``git status`` subprocess.
    Any of those raising lands in the ``except BaseException`` handler, which
    calls this again with a status and an error and nothing else. Under the
    old unconditional ``SET``, that second call blanked every counter the
    first had committed — a run whose content was fully indexed showed up on
    the dashboard as failed with no numbers at all (PR #24 round-8 review).

    ``status``, ``error`` and ``finished_at`` stay unconditional: the whole
    point of the second call is to overwrite those three, and ``error`` must
    be able to move back to NULL as well as to a message.
    """
    conn.execute(
        """
        UPDATE index_runs SET
          status = ?,
          files_total     = COALESCE(?, files_total),
          files_changed   = COALESCE(?, files_changed),
          chunks_written  = COALESCE(?, chunks_written),
          fast_path_used  = COALESCE(?, fast_path_used),
          candidate_count = COALESCE(?, candidate_count),
          files_failed    = COALESCE(?, files_failed),
          finished_at = ?, error = ?
        WHERE id = ?
        """,
        (
            status,
            files_total,
            files_changed,
            chunks_written,
            None if fast_path_used is None else int(fast_path_used),
            candidate_count,
            files_failed,
            _now(),
            error,
            run_id,
        ),
    )
    conn.commit()


def set_project_flags(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    watch_enabled: bool | None = None,
    auto_reindex: bool | None = None,
) -> None:
    """Update the per-project watcher flags (ADR-40). None leaves a flag
    untouched."""
    sets, params = [], []
    if watch_enabled is not None:
        sets.append("watch_enabled = ?")
        params.append(int(watch_enabled))
    if auto_reindex is not None:
        sets.append("auto_reindex = ?")
        params.append(int(auto_reindex))
    if not sets:
        return
    sets.append("updated_at = ?")
    params.extend([_now(), project_id])
    conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def set_index_config(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    index_languages: list[str] | None = None,
    max_file_bytes: int | None = None,
    follow_symlinks: bool = False,
    extra_ignores: list[str] | None = None,
) -> None:
    """Persist a project's index scope (ADR-42). ``index_languages`` /
    ``extra_ignores`` are JSON-encoded; an empty or None list stores NULL
    (meaning 'no filter' / 'all languages'). ``max_file_bytes`` None stores
    NULL (DiscoveryConfig default applies)."""
    langs_json = json.dumps(index_languages) if index_languages else None
    ignores_json = json.dumps(extra_ignores) if extra_ignores else None
    conn.execute(
        "UPDATE projects SET index_languages = ?, max_file_bytes = ?,"
        " follow_symlinks = ?, extra_ignores = ?, updated_at = ? WHERE id = ?",
        (
            langs_json,
            max_file_bytes,
            int(follow_symlinks),
            ignores_json,
            _now(),
            project_id,
        ),
    )
    conn.commit()


def delete_project(conn: sqlite3.Connection, project_id: str) -> None:
    """Remove a project and all its state rows, child-first so the FK
    constraints hold (ADR-43). ``query_log`` rows are deliberately kept:
    metadata-only aggregates with no FK — deleting them would silently
    rewrite usage history."""
    conn.execute(
        "DELETE FROM run_file_errors WHERE run_id IN"
        " (SELECT id FROM index_runs WHERE project_id = ?)",
        (project_id,),
    )
    conn.execute("DELETE FROM index_runs WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM pending_changes WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM watcher_stats WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM unwalkable_dirs WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM files WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()


def watched_projects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM projects WHERE watch_enabled = 1 ORDER BY created_at"
    ).fetchall()


def upsert_pending_changes(
    conn: sqlite3.Connection,
    project_id: str,
    changes: Iterable[tuple[str, str]],
) -> None:
    """Record watcher-detected dirty files: (path, event_type) pairs. A
    re-event on the same path refreshes event_type and detected_at, so the
    row survives a clear cut at an earlier timestamp (the reindex race is
    resolved toward re-examination, never toward silent loss)."""
    now = _now()
    conn.executemany(
        """
        INSERT INTO pending_changes (project_id, path, event_type, detected_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(project_id, path) DO UPDATE SET
          -- created + later modified is still a creation: the file has never
          -- been indexed. Deletion (or anything else) overwrites.
          event_type = CASE
            WHEN pending_changes.event_type = 'created'
                 AND excluded.event_type = 'modified'
            THEN 'created' ELSE excluded.event_type END,
          detected_at = excluded.detected_at
        """,
        [(project_id, path, event_type, now) for path, event_type in changes],
    )
    conn.commit()


def list_pending_changes(
    conn: sqlite3.Connection, project_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT path, event_type, detected_at FROM pending_changes"
        " WHERE project_id = ? ORDER BY detected_at DESC, path",
        (project_id,),
    ).fetchall()


def clear_pending_changes(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    paths: Iterable[str] | None = None,
    before: str | None = None,
) -> None:
    """Clear pending rows after a run examined them. *paths* limits the
    clear to a scoped run's candidate set (None = full run cleared all);
    *before* keeps rows re-dirtied while the run was executing."""
    where = "project_id = ?"
    if before is not None:
        where += " AND detected_at <= ?"
    if paths is None:
        params = [project_id] + ([before] if before is not None else [])
        conn.execute(f"DELETE FROM pending_changes WHERE {where}", params)
    else:
        base = [project_id] + ([before] if before is not None else [])
        conn.executemany(
            f"DELETE FROM pending_changes WHERE {where} AND path = ?",
            [(*base, p) for p in paths],
        )
    conn.commit()


def record_file_errors(
    conn: sqlite3.Connection, run_id: str, errors: Iterable[tuple[str, str]]
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO run_file_errors (run_id, path, error) VALUES (?, ?, ?)",
        [(run_id, path, err) for path, err in errors],
    )
    conn.commit()


def list_file_errors(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT path, error FROM run_file_errors WHERE run_id = ? ORDER BY path",
        (run_id,),
    ).fetchall()


def list_unwalkable_dirs(
    conn: sqlite3.Connection, project_id: str
) -> list[sqlite3.Row]:
    """Directories discovery could not walk, longest-standing first (ADR-56).

    Empty is the healthy state. A row here does NOT mean anything was deleted
    or will be — it means part of the tree went unseen, so nothing under it can
    be proven absent."""
    return conn.execute(
        "SELECT dir_path, consecutive_runs, first_seen_at, last_seen_at,"
        " last_error, quarantined_at FROM unwalkable_dirs WHERE project_id = ?"
        " ORDER BY first_seen_at, dir_path",
        (project_id,),
    ).fetchall()


def count_unwalkable_dirs(conn: sqlite3.Connection, project_id: str) -> tuple[int, int]:
    """``(unwalkable, quarantined)`` counts for the status surfaces. The second
    is a subset of the first: ``COUNT(col)`` skips NULLs."""
    row = conn.execute(
        "SELECT COUNT(*) AS n, COUNT(quarantined_at) AS q FROM unwalkable_dirs"
        " WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    return int(row["n"]), int(row["q"])


def reconcile_unwalkable_dirs(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    reported: Iterable[tuple[str, str]],
    cleared: Iterable[str],
    quarantine_after: int,
) -> None:
    """Fold one run's directory-level discovery result into the ledger (ADR-56).

    *reported* are this run's ``DiscoveryErrors.dirs`` pairs — each bumps its
    row's consecutive-failure count. *cleared* are recorded paths this run
    proved walkable; their rows are deleted, which is what resets the count.
    The caller decides what "proved walkable" means, because that is path
    reasoning (``indexer._under`` against the run's unwalked prefixes) and this
    module owns SQL only.

    Deletes run before inserts. They are disjoint by construction — a path
    reported this run is by definition under this run's failures — but if that
    ever stopped holding, this order re-inserts at count 1 and merely DELAYS a
    quarantine. The other order would drop a row that had just been counted.
    Erring toward retrying too long is the same fail-safe direction as ADR-51.

    *quarantine_after* is the consecutive-run threshold; ``0`` disables
    quarantine entirely and leaves ``quarantined_at`` NULL, so the retry
    behaviour reverts to what shipped in PR #24. ``quarantined_at`` is only
    ever set, never cleared: recovery deletes the whole row.

    One ``BEGIN IMMEDIATE`` transaction for the same reason
    :func:`add_dirty_paths` uses one — a half-applied reconcile would either
    quarantine on a stale count or lose a recovery.
    """
    reported = list(reported)
    cleared = [p for p in cleared]
    if not reported and not cleared:
        return
    now = _now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if cleared:
            conn.executemany(
                "DELETE FROM unwalkable_dirs WHERE project_id = ? AND dir_path = ?",
                [(project_id, p) for p in cleared],
            )
        if reported:
            conn.executemany(
                """
                INSERT INTO unwalkable_dirs (project_id, dir_path,
                    consecutive_runs, first_seen_at, last_seen_at, last_error)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(project_id, dir_path) DO UPDATE SET
                  consecutive_runs = unwalkable_dirs.consecutive_runs + 1,
                  last_seen_at = excluded.last_seen_at,
                  last_error = excluded.last_error
                """,
                [(project_id, path, now, now, err) for path, err in reported],
            )
            if quarantine_after > 0:
                conn.execute(
                    "UPDATE unwalkable_dirs SET quarantined_at = ?"
                    " WHERE project_id = ? AND quarantined_at IS NULL"
                    " AND consecutive_runs >= ?",
                    (now, project_id, quarantine_after),
                )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def scoped_runs_since_full(conn: sqlite3.Connection, project_id: str) -> int:
    """Scoped runs started since the last *completed full walk* (ADR-57).

    What this measures is "has a full walk happened recently", **not** "has a
    drain happened". The two come apart, and the distinction is deliberate.

    A full run that finishes ``done`` with contained per-file failures
    (ADR-41) is not a *clean* run, so it takes neither drain branch —
    ``clean_run`` gates both ``clear_dirty_paths`` and the anchor write, and it
    is exactly ``files_failed == 0``. It still resets this counter. That looks
    wrong until you price the alternative: requiring a clean full run means a
    project with a **permanently** unreadable directory can never reset, so
    every launch past the threshold promotes. Measured at threshold 3 over 12
    watcher runs on such a project: 3 full walks under this rule versus 9 under
    a clean-only rule — every run, on the population issue #25 is about, where
    the walk is slowest (9p/network mounts, ADR-45).

    Nor would that buy a drain. The drain is blocked by the discovery error
    itself, not by the promotion cadence, so promoting harder just re-walks a
    tree that still cannot be drained. That a latched error blocks the drain
    forever is a real gap, and it is issue #28's, not this function's.

    The cost of this rule is bounded and self-healing: a *transient* failure on
    a full run delays the drain by one promotion cycle, and the next full run
    is clean and drains. A permanent one is #28 either way.

    A **failed** full run is different and does not reset: it may have stopped
    before walking anything, so it is not evidence a full walk happened at all.

    ``scoped IS NULL`` rows predate the column. They are read as full walks for
    the baseline and never counted as scoped, so the counter simply starts from
    zero on an upgraded DB rather than inheriting a fabricated history."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM index_runs
        WHERE project_id = ? AND scoped = 1 AND started_at > COALESCE(
              (SELECT MAX(started_at) FROM index_runs
               WHERE project_id = ? AND status = 'done'
                 AND (scoped = 0 OR scoped IS NULL)), '')
        """,
        (project_id, project_id),
    ).fetchone()
    return int(row["n"])


def log_query(
    conn: sqlite3.Connection,
    *,
    interface: str,
    kind: str,
    project_id: str | None,
    channel: str | None = None,
    reranked: bool | None = None,
    latency_ms: float | None = None,
    result_count: int | None = None,
) -> None:
    """Metadata-only usage telemetry (ADR-40): never stores query text."""
    conn.execute(
        "INSERT INTO query_log (ts, interface, kind, project_id, channel,"
        " reranked, latency_ms, result_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _now(),
            interface,
            kind,
            project_id,
            channel,
            None if reranked is None else int(reranked),
            latency_ms,
            result_count,
        ),
    )
    conn.commit()


def bump_watcher_stats(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    events_seen: int = 0,
    events_coalesced: int = 0,
    auto_runs: int = 0,
) -> None:
    day = _now()[:10]
    conn.execute(
        """
        INSERT INTO watcher_stats (project_id, day, events_seen, events_coalesced, auto_runs)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(project_id, day) DO UPDATE SET
          events_seen = events_seen + excluded.events_seen,
          events_coalesced = events_coalesced + excluded.events_coalesced,
          auto_runs = auto_runs + excluded.auto_runs
        """,
        (project_id, day, events_seen, events_coalesced, auto_runs),
    )
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM app_settings WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else row["value"]


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_settings (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
