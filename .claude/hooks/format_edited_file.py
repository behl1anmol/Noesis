#!/usr/bin/env python3
"""PostToolUse hook, matcher "Edit|Write".

Formats exactly the file that was just edited. Reads the path from the
hook's stdin JSON (``tool_input.file_path`` — the documented shape,
https://code.claude.com/docs/en/hooks-guide#auto-format-code-after-edits),
not an env var: ``$CLAUDE_FILE_PATH`` is not a documented PostToolUse
variable and evaluated empty on at least one harness, silently turning
"format after edit" into a permanent no-op (PR #14 review).

Best-effort: a non-Python file, a path outside the repo, or ruff failing
must never fail the tool call that triggered this hook.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_hook_input  # noqa: E402


def main() -> None:
    data = read_hook_input()
    tool_input = data.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path or not str(file_path).endswith(".py"):
        return
    path = Path(file_path)
    if not path.is_file():
        return
    subprocess.run(
        ["uv", "run", "ruff", "format", "--quiet", str(path)],
        check=False,
    )


if __name__ == "__main__":
    main()
