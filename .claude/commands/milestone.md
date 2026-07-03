---
description: Load the spec, exit criterion, and current status for milestone $1
allowed-tools: Read, Bash(python .claude/scripts/devlog.py *:*)
---
1. Run `python .claude/scripts/devlog.py latest` for current state.
2. Read the "$1" row of the Milestones table in
   architecture-docs/code-indexer-expanded-architecture.md §4 and the relevant §3 design.
3. Restate the exit criterion, list remaining work as a checklist, and confirm
   the plan with me before writing code.
