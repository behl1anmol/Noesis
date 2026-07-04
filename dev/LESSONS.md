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

## [2] process (occurrences: 1)
**Mistake:** Accepted uv's newer-minor resolution of tree-sitter-language-pack (1.12.2 vs doc-pinned 1.8.x) after checking only version numbers and licenses; the newer minor was a full pyo3 rewrite that downloads grammars over HTTP at runtime and changed the parsing API, discovered only mid-build by a subagent
**Lesson:** When a resolved dependency version diverges from the doc-pinned snapshot, check its changelog/release notes for behavior changes — especially network activity, API rewrites, and thread-safety — before building on it; a version-number diff alone is not verification
**Rationale:** Version drift can change runtime behavior, not just numbers: 1.12.2 silently introduced outbound HTTP from core/ (rule-2-adjacent) and a pyo3-unsendable parser that panicked under the indexer's worker thread. Both cost mid-milestone rework and a stakeholder decision; a 2-minute changelog read at uv-add time would have surfaced both

## [1] process (occurrences: 1)
**Mistake:** uv init --package --python 3.12 silently wrote requires-python >=3.12, contradicting the doc-pinned 3.11 floor (tech stack table)
**Lesson:** After any scaffolding-tool run (uv init/add), diff the generated config against doc-pinned constraints before building on it
**Rationale:** Generator defaults silently override documented pins; catching drift at scaffold time costs seconds, catching it after downstream code depends on it costs a migration
