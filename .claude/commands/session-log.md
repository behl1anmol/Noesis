---
description: Write a mid-session or end-of-session checkpoint (summary, next steps, blockers) to the devlog
allowed-tools: Bash(python .claude/scripts/devlog.py *:*)
---
1. Summarize, in your own words: what was accomplished this session, the
   explicit next steps for whoever (human or agent) picks this up next, and
   any open blockers.
2. Run `python .claude/scripts/devlog.py session-end <session_id> --summary
   "<summary>" --next "<next steps>" --blockers "<blockers or 'none'>"`.
   Use the current session's id.
3. Confirm with `python .claude/scripts/devlog.py latest`.

Run this before ending any session — it's the deliberate, human-quality
alternative to session_end.py's hook, which can only fall back to a
mechanical transcript-tail extract if this was never run.
