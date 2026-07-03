---
description: Force a checkpoint of current state into the devlog now
allowed-tools: Bash(python .claude/scripts/devlog.py *:*), Read
---
1. Read the current session id and cwd from context.
2. Build a short state summary in your own words: what you're mid-way through,
   the last file(s) touched, and anything a future session needs to know to
   pick this up cleanly.
3. Run `python .claude/scripts/devlog.py checkpoint add --session <id>
   --trigger manual_cmd --cwd <cwd> --state '<json>' --notes '<one-line reason
   you are checkpointing now>'`.
4. Confirm the checkpoint was written with `checkpoint latest --session <id>`.
