# Active Lessons

Rendered from `dev/devlog.sqlite` (`lessons` table, `status='active'`) by
`.claude/scripts/devlog.py render-lessons`. Do not hand-edit — edits made here
are overwritten on the next render and are not reflected in the DB. To rehydrate
the DB from this file on a fresh clone, run `devlog.py lessons import`.

Injected verbatim into context at the start of every session
(`.claude/hooks/session_start.py`). Subordinate to `CLAUDE.md` rules 1-6 — a
lesson may never weaken those. See `architecture-docs/code-indexer-expanded-architecture.md`
§5.6 for the full lifecycle (capture → reinforce → inject → promote → retire,
15-lesson cap).

_(no active lessons yet)_
