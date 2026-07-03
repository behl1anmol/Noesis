#!/usr/bin/env python3
"""SessionEnd hook.

Docs confirm output/exit code are ignored for this event — it's side-effects
only. That also means it CANNOT literally interact with the user the way the
architecture doc's prose describes ("prompts... plus one explicit question:
any mistake this session worth a /lesson") — by the time SessionEnd fires,
no further turns happen, so there is no channel to ask anything. That
responsibility instead lives in CLAUDE.md rule 7 (capture a lesson via
/lesson before moving on) and the /session-log command (run deliberately,
mid-conversation, before the session ends) — documented deviation from the
doc's literal wording, for the reason above.

What this hook actually does, deterministically:
  1. Always stamps ended_at.
  2. If /session-log (or an earlier explicit session-end) already recorded a
     summary/next_steps, passes those through unchanged — never clobbers a
     human-authored summary with a fallback.
  3. Otherwise, falls back to a bounded tail-read of the transcript, tagged
     clearly as auto-extracted so `devlog.py latest` is honest about its
     provenance rather than presenting it as a deliberate summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_hook_input, run_devlog, tail_read_transcript  # noqa: E402

AUTO_TAG = "[auto-extracted — no /session-log run this session]"


def main() -> None:
    data = read_hook_input()
    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path")
    if not session_id:
        return

    existing_summary = None
    existing_next = None
    try:
        raw = run_devlog("session-get", session_id).strip()
        info = json.loads(raw) if raw else {}
        if info.get("found"):
            existing_summary = info.get("summary")
            existing_next = info.get("next_steps")
    except Exception:
        pass

    summary = existing_summary
    next_steps = existing_next

    if not summary or not next_steps:
        snap = tail_read_transcript(transcript_path)
        last_assistant = snap.get("last_assistant_message") or "(no transcript content available)"
        if not summary:
            summary = f"{AUTO_TAG} last assistant message: {last_assistant}"
        if not next_steps:
            next_steps = f"{AUTO_TAG} review the transcript tail; no explicit next-steps were recorded."

    try:
        run_devlog("session-end", session_id, "--summary", summary, "--next", next_steps)
    except Exception:
        try:
            run_devlog("session-end", session_id,
                       "--summary", f"{AUTO_TAG} (devlog session-end also failed once; retried)",
                       "--next", f"{AUTO_TAG} (see transcript directly)")
        except Exception:
            pass


if __name__ == "__main__":
    main()
