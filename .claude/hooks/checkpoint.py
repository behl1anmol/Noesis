#!/usr/bin/env python3
"""PreCompact + StopFailure hook — writes a checkpoint before/around an event
that could lose session continuity: context compaction (PreCompact, both
`manual` and `auto`) or a turn dying from an API/quota error (StopFailure:
`rate_limit`, `billing_error`, `overloaded`).

One combined script, dispatched via a --trigger-hint CLI argument baked into
each settings.json hook entry (one entry per matcher value), rather than by
parsing a matcher-specific stdin field. The exact stdin field name
PreCompact/StopFailure use to carry their matcher value (e.g. whether it's
literally called "trigger" or "error_type") is not shown with a verified
example in the current docs as of this build — baking the trigger label into
the command avoids depending on an unverified field name. session_id /
transcript_path / cwd / hook_event_name ARE verified common fields and are
read normally from stdin.

Explicit scope decision: no plan-limit / usage-quota proactive checkpoint.
That signal does not exist in Claude Code (verified — no hook exposes
quota-remaining telemetry). StopFailure is the closest available proxy and
is reactive (fires only after a turn already died from an API error) — this
script's StopFailure branch is that best-effort last resort, not a
proactive warning. A cutoff that never produces a StopFailure event has no
checkpoint coverage under this design; see CLAUDE.md rule 8.

MUST fail open: PreCompact supports blocking the compaction it's guarding
(exit 2, or JSON {"decision": "block", ...}); a bug in this script must
never be able to accidentally block compaction. StopFailure's output/exit
code are documented as ignored, so it's inherently safe there regardless.
This script therefore always exits 0 and never emits a "decision" key —
every error is swallowed, not surfaced as control flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_hook_input, run_devlog, tail_read_transcript  # noqa: E402

TRIGGER_CHOICES = [
    "precompact_auto",
    "precompact_manual",
    "stopfailure_rate_limit",
    "stopfailure_billing_error",
    "stopfailure_overloaded",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-hint", required=True, choices=TRIGGER_CHOICES)
    args, _ = parser.parse_known_args()

    try:
        data = read_hook_input()
        session_id = data.get("session_id") or "unknown"
        transcript_path = data.get("transcript_path")
        cwd = data.get("cwd")

        snap = tail_read_transcript(transcript_path)
        state_snapshot = {
            "trigger": args.trigger_hint,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cwd": cwd,
            **snap,
        }

        checkpoint_args = [
            "checkpoint",
            "add",
            "--session",
            session_id,
            "--trigger",
            args.trigger_hint,
            "--state",
            "-",
        ]
        if transcript_path:
            checkpoint_args += ["--transcript-path", transcript_path]
        if cwd:
            checkpoint_args += ["--cwd", cwd]

        run_devlog(*checkpoint_args, input_text=json.dumps(state_snapshot))
    except Exception:
        pass  # fail open, always — see module docstring

    sys.exit(0)


if __name__ == "__main__":
    main()
