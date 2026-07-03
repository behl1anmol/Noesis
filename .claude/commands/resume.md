---
description: Recall the most recent checkpoint and session state mid-conversation
allowed-tools: Bash(python .claude/scripts/devlog.py *:*)
---
1. Run `python .claude/scripts/devlog.py latest`.
2. Run `python .claude/scripts/devlog.py checkpoint latest`.
3. Summarize both for me in a few lines: where the last closed session left
   off, and the most recent checkpoint (trigger, when, what it captured).
