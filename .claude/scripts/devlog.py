#!/usr/bin/env python3
"""CLI over dev/devlog.sqlite — the sole write path for the session/lesson/checkpoint store.

Hooks and slash commands shell out to this script; nothing else touches the
DB file directly (enforced by the `deny` rule on Edit(dev/devlog.sqlite) in
.claude/settings.json). Schema is created idempotently on every invocation,
so a fresh clone self-bootstraps on first use — `init` exists only as an
explicit, visible first-run step for humans.

See architecture-docs/code-indexer-expanded-architecture.md §5.2/§5.6 for the
design rationale (sessions/decisions/milestones/lessons) and the plan file
that added the `checkpoints` table on top of it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "dev" / "devlog.sqlite"
LESSONS_MD_PATH = REPO_ROOT / "dev" / "LESSONS.md"
# Retired and promoted lessons, committed but NEVER injected. §5.6 says a
# non-active lesson "stays in the DB for audit" — which was only true on the
# machine that retired it, since the DB is gitignored, so a fresh clone lost
# the record of every lesson the project had ever decided against or graduated.
# They cannot go in LESSONS.md: `session_start.py` injects that file verbatim,
# so a retired lesson there would keep influencing sessions after being
# retired, and the archive would grow the injected surface the 15-cap exists to
# bound. A second committed file gets durability without either cost.
LESSONS_ARCHIVE_PATH = REPO_ROOT / "dev" / "LESSONS-archive.md"

LESSON_CAP = 15

CHECKPOINT_TRIGGERS = {
    "precompact_auto",
    "precompact_manual",
    "stopfailure_rate_limit",
    "stopfailure_billing_error",
    "stopfailure_overloaded",
    "manual_cmd",
}

MILESTONE_STATUSES = {"pending", "in_progress", "done", "blocked"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id          TEXT PRIMARY KEY,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  milestone   TEXT,
  summary     TEXT,
  next_steps  TEXT,
  blockers    TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT REFERENCES sessions(id),
  decided_at  TEXT NOT NULL,
  title       TEXT NOT NULL,
  decision    TEXT NOT NULL,
  rationale   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS milestones (
  id             TEXT PRIMARY KEY,
  status         TEXT NOT NULL DEFAULT 'pending',
  exit_criterion TEXT,
  evidence       TEXT,
  updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at  TEXT NOT NULL,
  session_id  TEXT REFERENCES sessions(id),
  category    TEXT NOT NULL,
  mistake     TEXT NOT NULL,
  lesson      TEXT NOT NULL,
  rationale   TEXT NOT NULL,
  occurrences INTEGER NOT NULL DEFAULT 1,
  status      TEXT NOT NULL DEFAULT 'active'
);

-- Not in the original doc schema: added so `lessons import` (rehydrating a
-- fresh clone's DB from the committed dev/LESSONS.md) is idempotent to
-- re-run, and so a duplicate mistake can never silently double-insert
-- outside the /lesson command's own bump-instead-of-duplicate convention.
CREATE UNIQUE INDEX IF NOT EXISTS idx_lessons_category_mistake
  ON lessons(category, mistake);

CREATE TABLE IF NOT EXISTS checkpoints (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id      TEXT REFERENCES sessions(id),
  created_at      TEXT NOT NULL,
  trigger         TEXT NOT NULL,
  transcript_path TEXT,
  cwd             TEXT,
  state_snapshot  TEXT NOT NULL,
  notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session_created
  ON checkpoints(session_id, created_at DESC);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# sessions / decisions / milestones
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    connect().close()
    print(f"devlog initialized at {DB_PATH}")


def cmd_session_start(args: argparse.Namespace) -> None:
    conn = connect()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
        (args.session_id, now()),
    )
    conn.commit()
    print(f"session {args.session_id} started")


def cmd_session_end(args: argparse.Namespace) -> None:
    conn = connect()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
        (args.session_id, now()),
    )
    conn.execute(
        """UPDATE sessions SET ended_at = ?, summary = ?, next_steps = ?,
           blockers = COALESCE(?, blockers) WHERE id = ?""",
        (now(), args.summary, args.next, args.blockers, args.session_id),
    )
    conn.commit()
    print(f"session {args.session_id} ended")


def _print_milestone_board(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, status, exit_criterion, evidence, updated_at FROM milestones ORDER BY id"
    ).fetchall()
    if not rows:
        print("Milestone board: (empty — no milestones set yet)")
        return
    print("Milestone board:")
    for r in rows:
        print(
            f"  {r['id']}: {r['status']}"
            + (f" — {r['exit_criterion']}" if r["exit_criterion"] else "")
        )


def _print_checkpoint(row: sqlite3.Row | None, *, label: str) -> None:
    if row is None:
        print(f"{label}: none found")
        return
    print(f"{label}: trigger={row['trigger']} at={row['created_at']}")
    try:
        snapshot = json.loads(row["state_snapshot"])
        print(json.dumps(snapshot, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(row["state_snapshot"])
    if row["notes"]:
        print(f"notes: {row['notes']}")


def cmd_latest(args: argparse.Namespace) -> None:
    conn = connect()
    row = conn.execute(
        """SELECT * FROM sessions WHERE ended_at IS NOT NULL
           ORDER BY ended_at DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        print("No closed session found yet.")
    else:
        print(f"Last session: {row['id']} (ended {row['ended_at']})")
        print(f"Summary: {row['summary'] or '(none recorded)'}")
        print(f"Next steps: {row['next_steps'] or '(none recorded)'}")
        print(f"Blockers: {row['blockers'] or '(none)'}")
    _print_milestone_board(conn)

    if args.include_dangling:
        if args.exclude_session:
            dangling = conn.execute(
                "SELECT * FROM sessions WHERE ended_at IS NULL AND id != ? ORDER BY started_at DESC",
                (args.exclude_session,),
            ).fetchall()
        else:
            dangling = conn.execute(
                "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC"
            ).fetchall()
        if not dangling:
            print("No dangling (uncleanly-ended) sessions.")
        for d in dangling:
            print(
                f"\nWARNING: session {d['id']} (started {d['started_at']}) has no SessionEnd —"
                " possible interruption."
            )
            cp = conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (d["id"],),
            ).fetchone()
            if cp is None:
                print(
                    "  No checkpoint exists for this session — nothing beyond the plain"
                    " transcript survived (see CLAUDE.md rule 8 on checkpoint coverage)."
                )
            else:
                _print_checkpoint(cp, label="  Last checkpoint")


def cmd_session_get(args: argparse.Namespace) -> None:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (args.session_id,)
    ).fetchone()
    if row is None:
        print(json.dumps({"id": args.session_id, "found": False}))
        return
    print(
        json.dumps(
            {
                "id": row["id"],
                "found": True,
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "summary": row["summary"],
                "next_steps": row["next_steps"],
                "blockers": row["blockers"],
            }
        )
    )


def cmd_decision_add(args: argparse.Namespace) -> None:
    conn = connect()
    cur = conn.execute(
        "INSERT INTO decisions (session_id, decided_at, title, decision, rationale) VALUES (?, ?, ?, ?, ?)",
        (args.session, now(), args.title, args.decision, args.rationale),
    )
    conn.commit()
    print(f"decision #{cur.lastrowid} recorded")


def cmd_milestone_set(args: argparse.Namespace) -> None:
    conn = connect()
    existing = conn.execute(
        "SELECT * FROM milestones WHERE id = ?", (args.milestone_id,)
    ).fetchone()
    exit_criterion = (
        args.exit_criterion
        if args.exit_criterion is not None
        else (existing["exit_criterion"] if existing else None)
    )
    evidence = (
        args.evidence
        if args.evidence is not None
        else (existing["evidence"] if existing else None)
    )
    conn.execute(
        """INSERT INTO milestones (id, status, exit_criterion, evidence, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET status=excluded.status,
             exit_criterion=excluded.exit_criterion, evidence=excluded.evidence,
             updated_at=excluded.updated_at""",
        (args.milestone_id, args.status, exit_criterion, evidence, now()),
    )
    conn.commit()
    print(f"milestone {args.milestone_id} -> {args.status}")


# ---------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------


def cmd_lesson_add(args: argparse.Namespace) -> None:
    conn = connect()
    try:
        cur = conn.execute(
            """INSERT INTO lessons (created_at, session_id, category, mistake, lesson, rationale)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now(),
                args.session,
                args.category,
                args.mistake,
                args.lesson,
                args.rationale,
            ),
        )
        conn.commit()
        print(f"lesson #{cur.lastrowid} added")
    except sqlite3.IntegrityError:
        existing = conn.execute(
            "SELECT id FROM lessons WHERE category = ? AND mistake = ?",
            (args.category, args.mistake),
        ).fetchone()
        print(
            f"A lesson for this exact category+mistake already exists (#{existing['id']}) —"
            f" run `lesson bump {existing['id']}` instead of adding a duplicate."
        )
        sys.exit(1)


def cmd_lesson_bump(args: argparse.Namespace) -> None:
    conn = connect()
    conn.execute(
        "UPDATE lessons SET occurrences = occurrences + 1 WHERE id = ?",
        (args.lesson_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT occurrences FROM lessons WHERE id = ?", (args.lesson_id,)
    ).fetchone()
    if row is None:
        print(f"no lesson #{args.lesson_id}")
        sys.exit(1)
    print(f"lesson #{args.lesson_id} occurrences -> {row['occurrences']}")
    if row["occurrences"] >= 3:
        print(
            "Recurred >= 3 times — propose promoting this to a CLAUDE.md hard rule"
            " (`lesson promote`, human-approved edit)."
        )


def cmd_lesson_retire(args: argparse.Namespace) -> None:
    conn = connect()
    conn.execute(
        "UPDATE lessons SET status = 'retired' WHERE id = ?", (args.lesson_id,)
    )
    conn.commit()
    print(f"lesson #{args.lesson_id} retired")


def cmd_lesson_promote(args: argparse.Namespace) -> None:
    conn = connect()
    conn.execute(
        "UPDATE lessons SET status = 'promoted' WHERE id = ?", (args.lesson_id,)
    )
    conn.commit()
    print(
        f"lesson #{args.lesson_id} marked promoted — now add it to CLAUDE.md by hand"
        " (Edit(CLAUDE.md) is permission-gated by design; this command does not write it)."
    )


def _lessons_missing_from_db(conn: sqlite3.Connection) -> list[str]:
    """Lessons the committed file documents that the DB has never seen.

    The hydration check for :func:`cmd_render_lessons`, and deliberately NOT a
    count comparison. Retiring or promoting a lesson legitimately lowers the
    active count below the file's, so a count test would cry wolf on the two
    lifecycle operations §5.6 exists to encourage. What is never legitimate is
    the file documenting a lesson the DB holds in NO status: retired and
    promoted rows stay in the table for audit, so an absent row means this DB
    is not the store the file came from — a fresh clone, another machine, or a
    lost file — and rendering from it would delete that lesson from the only
    durable copy.
    """
    known = {
        (r["category"], r["mistake"])
        for r in conn.execute("SELECT category, mistake FROM lessons")
    }
    missing = []
    for path in (LESSONS_MD_PATH, LESSONS_ARCHIVE_PATH):
        if not path.exists():
            continue
        for m in _lesson_blocks(path.read_text()):
            if (m["category"], m["mistake"]) not in known:
                missing.append(f"[{m['id']}] {m['category']} ({path.name})")
    return missing


def cmd_render_lessons(args: argparse.Namespace) -> None:
    conn = connect()
    # Rule 9 / lesson 11, enforced rather than remembered. This command
    # regenerates a COMMITTED file from a gitignored, machine-local DB, so on
    # any machine where the DB is behind the file it is a delete, not a
    # refresh — and it reports success either way. It has destroyed the
    # tracked lesson set once and come within one command of doing so again.
    missing = _lessons_missing_from_db(conn)
    if missing and not getattr(args, "force", False):
        print(
            f"REFUSING to render: {len(missing)} committed lesson(s) are absent from"
            " this DB, so rendering would DELETE them from the only durable copy."
            "\n  "
            + "\n  ".join(missing)
            + "\n\nThe file is the source of truth on a fresh clone; the DB is a local"
            " cache.\nRun `devlog.py lessons import` first, then render."
            "\n(--force overrides, and is the wrong answer unless you have just"
            " checked git diff.)"
        )
        raise SystemExit(1)
    # `id DESC` rather than `created_at DESC` as the tiebreak. Both mean
    # "newest first" — ids are AUTOINCREMENT — but only `id` survives a
    # rehydrate: `lessons import` cannot recover a created_at the file never
    # stored, so it stamps `now()` and every rehydrated lesson ties. Sorting on
    # that reshuffled the whole occurrence-1 group on any fresh clone, which
    # made a rehydrate look like a content change in review. Ordering on data
    # the file itself carries makes import → render an identity.
    rows = conn.execute(
        "SELECT * FROM lessons WHERE status = 'active'"
        " ORDER BY occurrences DESC, id DESC"
    ).fetchall()

    lines = [
        "# Active Lessons",
        "",
        "Rendered from `dev/devlog.sqlite` (`lessons` table, `status='active'`) by",
        "`.claude/scripts/devlog.py render-lessons`. Do not hand-edit — edits made here",
        "are overwritten on the next render and are not reflected in the DB. To rehydrate",
        "the DB from this file on a fresh clone, run `devlog.py lessons import`.",
        "",
        "Injected verbatim into context at the start of every session",
        "(`.claude/hooks/session_start.py`). Subordinate to `CLAUDE.md` rules 1-6 — a",
        "lesson may never weaken those. See `architecture-docs/code-indexer-expanded-architecture.md`",
        "§5.6 for the full lifecycle (capture → reinforce → inject → promote → retire,",
        "15-lesson cap).",
        "",
    ]
    if not rows:
        lines.append("_(no active lessons yet)_")
    for r in rows:
        lines.append(
            f"## [{r['id']}] {r['category']} (occurrences: {r['occurrences']})"
        )
        lines.append(f"**Mistake:** {r['mistake']}")
        lines.append(f"**Lesson:** {r['lesson']}")
        lines.append(f"**Rationale:** {r['rationale']}")
        lines.append("")

    LESSONS_MD_PATH.write_text("\n".join(lines).rstrip() + "\n")
    print(f"rendered {len(rows)} active lesson(s) to {LESSONS_MD_PATH}")
    if len(rows) > LESSON_CAP:
        print(
            f"WARNING: {len(rows)} active lessons exceeds the {LESSON_CAP}-lesson cap —"
            " promote or retire one before adding more (do not silently exceed it)."
        )

    # The audit half. Same render, different file, and NOT injected — so
    # retiring a lesson stops it influencing sessions (the point of retiring)
    # without destroying the record that it existed and why.
    archived = conn.execute(
        "SELECT * FROM lessons WHERE status != 'active'"
        " ORDER BY status, occurrences DESC, id DESC"
    ).fetchall()
    arch = [
        "# Retired & Promoted Lessons (archive)",
        "",
        "Rendered from `dev/devlog.sqlite` by `.claude/scripts/devlog.py render-lessons`,",
        "alongside `LESSONS.md`. Do not hand-edit.",
        "",
        "**This file is NOT injected into context** — that is the whole point of it.",
        "A *retired* lesson stopped being guidance; a *promoted* one became a CLAUDE.md",
        "hard rule or a mechanical guard and no longer needs repeating. Both are kept",
        "here because the `lessons` table is machine-local and gitignored, so without a",
        "committed copy a fresh clone loses every lesson the project ever decided",
        "against or graduated — including the reason it did. `devlog.py lessons import`",
        "reads this file too, and restores each row with the status recorded below.",
        "",
    ]
    if not archived:
        arch.append("_(nothing retired or promoted yet)_")
    for r in archived:
        arch.append(
            f"## [{r['id']}] {r['category']} (occurrences: {r['occurrences']},"
            f" status: {r['status']})"
        )
        arch.append(f"**Mistake:** {r['mistake']}")
        arch.append(f"**Lesson:** {r['lesson']}")
        arch.append(f"**Rationale:** {r['rationale']}")
        arch.append("")
    LESSONS_ARCHIVE_PATH.write_text("\n".join(arch).rstrip() + "\n")
    print(f"rendered {len(archived)} archived lesson(s) to {LESSONS_ARCHIVE_PATH}")


LESSON_BLOCK_RE = (
    None  # compiled lazily to keep import cost near zero for other subcommands
)


def _lesson_blocks(text: str) -> list[dict]:
    """Parse rendered lesson blocks out of LESSONS.md.

    ``id`` is captured, not discarded. It used to be matched as a bare ``\\d+``
    and thrown away, so every import re-inserted under AUTOINCREMENT and the
    next render renumbered all fifteen headings — silently invalidating the
    cross-references the lesson bodies make to each other by number ("this is
    lesson 14's failure mode"). Round-tripping the file has to be identity, or
    the hydration guard above would itself churn what it is protecting.
    """
    import re

    global LESSON_BLOCK_RE
    if LESSON_BLOCK_RE is None:
        LESSON_BLOCK_RE = re.compile(
            r"^## \[(?P<id>\d+)\] (?P<category>.+?) \(occurrences: (?P<occurrences>\d+)"
            r"(?:, status: (?P<status>\w+))?\)\n"
            r"\*\*Mistake:\*\* (?P<mistake>.+)\n"
            r"\*\*Lesson:\*\* (?P<lesson>.+)\n"
            r"\*\*Rationale:\*\* (?P<rationale>.+)$",
            re.MULTILINE,
        )
    out = []
    for m in LESSON_BLOCK_RE.finditer(text):
        d = m.groupdict()
        # The status suffix appears only in the archive, so its absence means
        # active — which keeps LESSONS.md's heading byte-identical to what it
        # has always been, and lets one parser read both files.
        d["status"] = d["status"] or "active"
        out.append(d)
    return out


def cmd_lessons_import(args: argparse.Namespace) -> None:
    if not LESSONS_MD_PATH.exists() and not LESSONS_ARCHIVE_PATH.exists():
        print(f"{LESSONS_MD_PATH} does not exist — nothing to import")
        return

    conn = connect()
    imported = 0
    # Both files, so a fresh clone rehydrates the FULL ledger — active lessons
    # and the retired/promoted record alike. Reading only LESSONS.md is what
    # made a rehydrate silently drop every non-active row.
    blocks = []
    for path in (LESSONS_MD_PATH, LESSONS_ARCHIVE_PATH):
        if path.exists():
            blocks.extend(_lesson_blocks(path.read_text()))
    for m in blocks:
        # Resolved explicitly rather than with an upsert: the table carries TWO
        # uniqueness rules — the `id` primary key and a UNIQUE(category,
        # mistake) index — and SQLite takes only one ON CONFLICT target, so an
        # upsert on either one raises IntegrityError the moment the other is
        # the one that collides.
        existing = conn.execute(
            "SELECT id, occurrences, status FROM lessons"
            " WHERE category = ? AND mistake = ?",
            (m["category"], m["mistake"]),
        ).fetchone()
        if existing is not None:
            # Same lesson already here: refresh it in place and KEEP its id.
            # Renumbering it to the file's id would break any row that already
            # points at the old one, and the file is about to be re-rendered
            # from this table anyway.
            if int(m["occurrences"]) > existing["occurrences"]:
                conn.execute(
                    "UPDATE lessons SET occurrences = ?, lesson = ?, rationale = ?"
                    " WHERE id = ?",
                    (
                        int(m["occurrences"]),
                        m["lesson"],
                        m["rationale"],
                        existing["id"],
                    ),
                )
            # A local row that is still `active` while the committed archive
            # says retired/promoted means this machine never saw the lifecycle
            # change. The files are the shared record, so they win.
            if m["status"] != "active" and existing["status"] != m["status"]:
                conn.execute(
                    "UPDATE lessons SET status = ? WHERE id = ?",
                    (m["status"], existing["id"]),
                )
        else:
            # New to this DB — the rehydrate path. Take the file's id when it
            # is free, so a fresh clone round-trips to a byte-identical file;
            # fall back to AUTOINCREMENT when some other lesson already holds
            # it, since a wrong id is worse than a new one.
            taken = conn.execute(
                "SELECT 1 FROM lessons WHERE id = ?", (int(m["id"]),)
            ).fetchone()
            conn.execute(
                "INSERT INTO lessons (id, created_at, category, mistake, lesson,"
                " rationale, occurrences, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    None if taken else int(m["id"]),
                    now(),
                    m["category"],
                    m["mistake"],
                    m["lesson"],
                    m["rationale"],
                    int(m["occurrences"]),
                    m["status"],
                ),
            )
        imported += 1
    # Keep AUTOINCREMENT ahead of the ids just written, or the next `lesson
    # add` collides with a rehydrated row.
    conn.execute(
        "UPDATE sqlite_sequence SET seq = (SELECT MAX(id) FROM lessons)"
        " WHERE name = 'lessons' AND seq < (SELECT MAX(id) FROM lessons)"
    )
    conn.commit()
    print(
        f"imported/updated {imported} lesson(s) from {LESSONS_MD_PATH.name}"
        f" + {LESSONS_ARCHIVE_PATH.name}"
    )


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def cmd_checkpoint_add(args: argparse.Namespace) -> None:
    if args.trigger not in CHECKPOINT_TRIGGERS:
        print(
            f"invalid trigger '{args.trigger}', must be one of {sorted(CHECKPOINT_TRIGGERS)}"
        )
        sys.exit(1)

    raw_state = sys.stdin.read() if args.state == "-" else args.state
    try:
        json.loads(raw_state)
    except json.JSONDecodeError as e:
        print(f"--state is not valid JSON: {e}")
        sys.exit(1)

    conn = connect()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
        (args.session, now()),
    )
    cur = conn.execute(
        """INSERT INTO checkpoints (session_id, created_at, trigger, transcript_path, cwd, state_snapshot, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            args.session,
            now(),
            args.trigger,
            args.transcript_path,
            args.cwd,
            raw_state,
            args.notes,
        ),
    )
    conn.commit()
    print(f"checkpoint #{cur.lastrowid} written (trigger={args.trigger})")


def cmd_checkpoint_latest(args: argparse.Namespace) -> None:
    conn = connect()
    if args.session:
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (args.session,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM checkpoints ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
    _print_checkpoint(row, label="Latest checkpoint")


def cmd_checkpoint_list(args: argparse.Namespace) -> None:
    conn = connect()
    query = "SELECT * FROM checkpoints"
    params: list = []
    if args.session:
        query += " WHERE session_id = ?"
        params.append(args.session)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(args.limit)
    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("no checkpoints found")
        return
    for r in rows:
        print(
            f"#{r['id']} session={r['session_id']} trigger={r['trigger']} at={r['created_at']}"
        )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="devlog.py")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create/verify the DB schema").set_defaults(
        func=cmd_init
    )

    sp = sub.add_parser("session-start", help="record a session start")
    sp.add_argument("session_id")
    sp.set_defaults(func=cmd_session_start)

    sp = sub.add_parser("session-end", help="record a session end")
    sp.add_argument("session_id")
    sp.add_argument("--summary", required=True)
    sp.add_argument("--next", required=True, help="explicit next-steps handoff")
    sp.add_argument("--blockers", default=None)
    sp.set_defaults(func=cmd_session_end)

    sp = sub.add_parser("latest", help="print last closed session + milestone board")
    sp.add_argument("--include-dangling", action="store_true")
    sp.add_argument(
        "--exclude-session",
        default=None,
        help="omit this session id from the dangling-session list (e.g. the "
        "just-started current session, which is always open at this point)",
    )
    sp.set_defaults(func=cmd_latest)

    sp = sub.add_parser("session-get", help="print a session's stored fields as JSON")
    sp.add_argument("session_id")
    sp.set_defaults(func=cmd_session_get)

    sp = sub.add_parser("decision")
    dsub = sp.add_subparsers(dest="decision_command", required=True)
    dadd = dsub.add_parser("add")
    dadd.add_argument("--title", required=True)
    dadd.add_argument("--decision", required=True)
    dadd.add_argument("--rationale", required=True)
    dadd.add_argument("--session", default=None)
    dadd.set_defaults(func=cmd_decision_add)

    sp = sub.add_parser("milestone")
    msub = sp.add_subparsers(dest="milestone_command", required=True)
    mset = msub.add_parser("set")
    mset.add_argument("milestone_id")
    mset.add_argument("--status", required=True, choices=sorted(MILESTONE_STATUSES))
    mset.add_argument("--exit-criterion", default=None)
    mset.add_argument("--evidence", default=None)
    mset.set_defaults(func=cmd_milestone_set)

    sp = sub.add_parser("lesson")
    lsub = sp.add_subparsers(dest="lesson_command", required=True)

    ladd = lsub.add_parser("add")
    ladd.add_argument("--category", required=True)
    ladd.add_argument("--mistake", required=True)
    ladd.add_argument("--lesson", required=True)
    ladd.add_argument("--rationale", required=True)
    ladd.add_argument("--session", default=None)
    ladd.set_defaults(func=cmd_lesson_add)

    lbump = lsub.add_parser("bump")
    lbump.add_argument("lesson_id", type=int)
    lbump.set_defaults(func=cmd_lesson_bump)

    lretire = lsub.add_parser("retire")
    lretire.add_argument("lesson_id", type=int)
    lretire.set_defaults(func=cmd_lesson_retire)

    lpromote = lsub.add_parser("promote")
    lpromote.add_argument("lesson_id", type=int)
    lpromote.set_defaults(func=cmd_lesson_promote)

    rl = sub.add_parser("render-lessons")
    rl.add_argument(
        "--force",
        action="store_true",
        help="render even when the DB is missing lessons the committed file"
        " documents (this DELETES them — check `git diff dev/LESSONS.md` after)",
    )
    rl.set_defaults(func=cmd_render_lessons)

    sp = sub.add_parser("lessons")
    lssub = sp.add_subparsers(dest="lessons_command", required=True)
    limport = lssub.add_parser("import")
    limport.set_defaults(func=cmd_lessons_import)

    sp = sub.add_parser("checkpoint")
    csub = sp.add_subparsers(dest="checkpoint_command", required=True)

    cadd = csub.add_parser("add")
    cadd.add_argument("--session", required=True)
    cadd.add_argument("--trigger", required=True, choices=sorted(CHECKPOINT_TRIGGERS))
    cadd.add_argument("--transcript-path", default=None)
    cadd.add_argument("--cwd", default=None)
    cadd.add_argument(
        "--state", required=True, help="JSON string, or '-' to read JSON from stdin"
    )
    cadd.add_argument("--notes", default=None)
    cadd.set_defaults(func=cmd_checkpoint_add)

    clatest = csub.add_parser("latest")
    clatest.add_argument("--session", default=None)
    clatest.set_defaults(func=cmd_checkpoint_latest)

    clist = csub.add_parser("list")
    clist.add_argument("--session", default=None)
    clist.add_argument("--limit", type=int, default=10)
    clist.set_defaults(func=cmd_checkpoint_list)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
