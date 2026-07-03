#!/usr/bin/env python3
"""PreToolUse hook, matcher "Bash".

Threat model, stated plainly (see plan's risk register): this is a denylist
that guards against MISTAKES — an agent about to run something obviously
destructive — not a security boundary against adversarial bypass. Chained
commands, alternate shells, variable expansion, or a disguised destructive
command steered by prompt injection in file content the agent reads are all
realistic ways to route around a regex denylist. Do not treat this as a
substitute for the permission system; it's a second, complementary layer
that catches patterns anywhere in a compound command (e.g. "echo hi &&
rm -rf /"), which a prefix-style `Bash(rm -rf *:*)` permission-list glob
entry cannot catch since that only matches at the start of the command.

Denies via the documented JSON form (verified current over the doc's literal
"exit code 2" phrasing):
  {"hookSpecificOutput": {"hookEventName": "PreToolUse",
    "permissionDecision": "deny", "permissionDecisionReason": "..."}}
No output at all (empty stdout, exit 0) means "no opinion" — falls through
to the normal permission flow for everything not matched below.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_hook_input  # noqa: E402

# (predicate(command) -> bool, human-readable reason). Deliberately short
# and specific — patterns that a prefix-glob permission entry can't already
# catch, not an attempt at an exhaustive denylist.

def _is_force_push(command: str) -> bool:
    # --force-with-lease is the safe, reviewed variant; strip it out first so
    # the bare --force/-f check below doesn't false-positive on the sub-match
    # "--force" that's a substring of "--force-with-lease".
    sanitized = re.sub(r"--force-with-lease(=\S+)?", "", command)
    return bool(re.search(r"\bgit\s+push\b[^\n]*(--force\b|\s-f\b)", sanitized))


DANGEROUS_PATTERNS: list[tuple[object, str]] = [
    (re.compile(r"\brm\s+-rf\b").search, "rm -rf anywhere in the command, including inside a chain (e.g. `a && rm -rf ...`)"),
    (_is_force_push, "force push (git push --force / -f; --force-with-lease is allowed)"),
    (re.compile(r"\bgit\s+reset\s+--hard\b").search, "git reset --hard (discards uncommitted work)"),
    (re.compile(r"\bsudo\b").search, "sudo — this project should never need elevated privileges"),
    (re.compile(r"\bdd\s+if=").search, "dd if=... (raw disk/device write)"),
    (re.compile(r"\bmkfs\b").search, "mkfs (filesystem format)"),
    (re.compile(r"\bchmod\s+-R\s+777\b").search, "recursive chmod 777"),
    (re.compile(r">\s*/dev/sd[a-z]").search, "direct write to a raw block device"),
]


def main() -> None:
    data = read_hook_input()
    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command") or ""

    for predicate, reason in DANGEROUS_PATTERNS:
        if predicate(command):
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"guard_bash.py: blocked — {reason}. "
                                                 f"Command: {command[:200]}",
                }
            }))
            return

    # No match: emit nothing, let the normal permission flow decide.


if __name__ == "__main__":
    main()
