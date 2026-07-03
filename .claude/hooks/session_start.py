#!/usr/bin/env python3
"""SessionStart hook.

Injects, as plain stdout (auto-added to context for this event per the
Claude Code hooks docs — no JSON envelope needed for a context-only hook):
  1. the last closed session's summary/next-steps/blockers + milestone board
  2. a warning + best-effort checkpoint if a prior session ended uncleanly
     (ended_at IS NULL) — explicitly says so if no checkpoint exists for it,
     rather than silently omitting the gap
  3. on source == "compact": the checkpoint PreCompact just wrote, so the
     compaction round-trip (write on PreCompact, read back here) is closed
  4. the active lessons from dev/LESSONS.md
  5. a pointer to the architecture docs

Every step is independently wrapped — one failing piece (e.g. a corrupt DB)
degrades that section's text, it never prevents the rest of the context or
crashes the session start.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, read_hook_input, run_devlog  # noqa: E402

LESSONS_MD = REPO_ROOT / "dev" / "LESSONS.md"
ARCH_DOCS = [
    "architecture-docs/code-indexer-expanded-architecture.md",
    "architecture-docs/code-indexer-initial-idea.md",
]


def main() -> None:
    data = read_hook_input()
    session_id = data.get("session_id", "")
    source = data.get("source", "")

    sections: list[str] = []

    if session_id:
        try:
            run_devlog("session-start", session_id)
        except Exception:
            sections.append("(failed to record session-start in devlog — dev/devlog.sqlite"
                             " may be missing or unwritable)")

    try:
        latest_args = ["latest", "--include-dangling"]
        if session_id:
            latest_args += ["--exclude-session", session_id]
        sections.append(run_devlog(*latest_args).strip())
    except Exception:
        sections.append("(devlog latest failed — dev/devlog.sqlite may be missing or corrupt;"
                         " run `python .claude/scripts/devlog.py init`)")

    if source == "compact" and session_id:
        try:
            checkpoint_text = run_devlog("checkpoint", "latest", "--session", session_id).strip()
            sections.append("Compaction just occurred; state immediately prior to it:\n" + checkpoint_text)
        except Exception:
            sections.append("Compaction just occurred, but no checkpoint could be retrieved for this session.")

    try:
        if not LESSONS_MD.exists():
            run_devlog("render-lessons")
        lessons_text = LESSONS_MD.read_text().strip() if LESSONS_MD.exists() else "(no lessons file found)"
        sections.append(lessons_text)
    except Exception:
        sections.append("(failed to load dev/LESSONS.md)")

    sections.append("Architecture docs: " + ", ".join(ARCH_DOCS))

    print("\n\n---\n\n".join(s for s in sections if s))


if __name__ == "__main__":
    main()
