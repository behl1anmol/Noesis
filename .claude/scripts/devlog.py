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


def cmd_render_lessons(args: argparse.Namespace) -> None:
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM lessons WHERE status = 'active' ORDER BY occurrences DESC, created_at DESC"
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


LESSON_BLOCK_RE = (
    None  # compiled lazily to keep import cost near zero for other subcommands
)


def cmd_lessons_import(args: argparse.Namespace) -> None:
    import re

    global LESSON_BLOCK_RE
    if LESSON_BLOCK_RE is None:
        LESSON_BLOCK_RE = re.compile(
            r"^## \[\d+\] (?P<category>.+?) \(occurrences: (?P<occurrences>\d+)\)\n"
            r"\*\*Mistake:\*\* (?P<mistake>.+)\n"
            r"\*\*Lesson:\*\* (?P<lesson>.+)\n"
            r"\*\*Rationale:\*\* (?P<rationale>.+)$",
            re.MULTILINE,
        )

    if not LESSONS_MD_PATH.exists():
        print(f"{LESSONS_MD_PATH} does not exist — nothing to import")
        return

    text = LESSONS_MD_PATH.read_text()
    conn = connect()
    imported = 0
    for m in LESSON_BLOCK_RE.finditer(text):
        conn.execute(
            """INSERT INTO lessons (created_at, category, mistake, lesson, rationale, occurrences, status)
               VALUES (?, ?, ?, ?, ?, ?, 'active')
               ON CONFLICT(category, mistake) DO UPDATE SET
                 occurrences=excluded.occurrences, lesson=excluded.lesson, rationale=excluded.rationale
               WHERE excluded.occurrences > lessons.occurrences""",
            (
                now(),
                m["category"],
                m["mistake"],
                m["lesson"],
                m["rationale"],
                int(m["occurrences"]),
            ),
        )
        imported += 1
    conn.commit()
    print(f"imported/updated {imported} lesson(s) from {LESSONS_MD_PATH}")


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

    sub.add_parser("render-lessons").set_defaults(func=cmd_render_lessons)

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
