---
description: Record an architecture decision to both the devlog and the doc's decision log
allowed-tools: Bash(python .claude/scripts/devlog.py *:*), Edit, Read
argument-hint: <title>
---
1. Title is `$ARGUMENTS`. Ask me for the decision and its rationale if not
   already clear from context — no rationale, no ADR (same house rule as
   lessons).
2. Run `python .claude/scripts/devlog.py decision add --title "$ARGUMENTS"
   --decision "<decision>" --rationale "<rationale>" --session <session_id>`.
3. Append a corresponding row to the Decision Log (Appendix A) in
   architecture-docs/code-indexer-expanded-architecture.md, following its
   existing table format (# | Decision | Chosen | Key reason | Alternative).
4. Confirm both writes succeeded before continuing.
