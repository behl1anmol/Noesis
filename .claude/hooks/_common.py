"""Shared helpers for .claude/hooks/*.py.

One source of truth for reading hook stdin, calling devlog.py, and doing a
bounded tail-read of a transcript JSONL file. Individual hook scripts should
import this rather than re-implement any of it — duplicating the transcript
parser across files is exactly how the four hooks would quietly drift apart.

Every function here is best-effort and must never raise in a way that could
take down a hook: hook scripts sit on a real control path (PreToolUse,
PreCompact can block; SessionStart's stdout becomes model context), so a bug
in a "nice to have" helper must degrade to an empty/None result, not a crash.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVLOG = REPO_ROOT / ".claude" / "scripts" / "devlog.py"


def read_hook_input() -> dict[str, Any]:
    """Read and parse the hook's stdin JSON. Never raises — returns {} on any
    parse failure so a malformed/empty stdin degrades gracefully instead of
    crashing the hook."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}


def run_devlog(*args: str, input_text: str | None = None, check: bool = True) -> str:
    """Run devlog.py with the given args, return its stdout.

    Raises subprocess.CalledProcessError if check=True and devlog.py exits
    non-zero — callers on a control path (e.g. checkpoint.py) must wrap this
    in their own try/except so a devlog.py bug can't block PreCompact.
    """
    result = subprocess.run(
        [sys.executable, str(DEVLOG), *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout


def tail_read_transcript(
    transcript_path: str | None, max_lines: int = 50
) -> dict[str, Any]:
    """Bounded tail-read of a transcript JSONL file for a mechanical, cheap
    state snapshot — never a full-file parse, never an LLM call (hooks can't
    invoke the model synchronously).

    Best-effort against the transcript's actual shape: if a line isn't valid
    JSON, or the expected message/content structure isn't there, the
    corresponding field is just left None rather than raising. This function
    is a nice-to-have context aid, not a correctness-critical path.
    """
    summary: dict[str, Any] = {
        "last_user_message": None,
        "last_assistant_message": None,
        "recent_tool_calls": [],
    }
    if not transcript_path:
        return summary
    path = Path(transcript_path)
    if not path.exists():
        return summary

    try:
        with path.open() as f:
            tail_lines = list(deque(f, maxlen=max_lines))
    except OSError:
        return summary

    tool_calls: list[dict[str, Any]] = []
    for line in reversed(tail_lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        role = entry.get("type") or entry.get("role")
        message = entry.get("message", entry)
        if not isinstance(message, dict):
            continue

        if role == "user" and summary["last_user_message"] is None:
            text = _extract_text(message)
            if text:
                summary["last_user_message"] = text[:500]
        elif role == "assistant" and summary["last_assistant_message"] is None:
            text = _extract_text(message)
            if text:
                summary["last_assistant_message"] = text[:500]

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and len(tool_calls) < 5
                ):
                    tool_calls.append(
                        {
                            "tool": block.get("name"),
                            "target": _tool_target(block.get("input")),
                        }
                    )

        if (
            len(tool_calls) >= 5
            and summary["last_user_message"]
            and summary["last_assistant_message"]
        ):
            break

    summary["recent_tool_calls"] = tool_calls[:5]
    return summary


def _extract_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text")
        ]
        if parts:
            return " ".join(parts)
    return None


def _tool_target(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path", "command", "pattern"):
        if key in tool_input:
            return str(tool_input[key])[:200]
    return None
