---
description: Capture a mistake as a corrective lesson in the devlog
allowed-tools: Bash(python .claude/scripts/devlog.py *:*), Read
---
1. Check existing lessons: `python .claude/scripts/devlog.py render-lessons`.
   If this mistake matches an existing lesson, run `lesson bump <id>`
   instead of adding a duplicate — recurrence is the promotion signal.
2. Otherwise run `lesson add` with: category, the concrete mistake, the
   corrective lesson in imperative voice ("Query the live filesystem for
   structural search, never the index"), and the rationale. No rationale,
   no lesson.
3. Run `render-lessons` to refresh dev/LESSONS.md, and state the new
   active-lesson count. If it exceeds 15, propose which lesson to retire
   or promote — do not silently exceed the cap.
