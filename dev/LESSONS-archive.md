# Retired & Promoted Lessons (archive)

Rendered from `dev/devlog.sqlite` by `.claude/scripts/devlog.py render-lessons`,
alongside `LESSONS.md`. Do not hand-edit.

**This file is NOT injected into context** — that is the whole point of it.
A *retired* lesson stopped being guidance; a *promoted* one became a CLAUDE.md
hard rule or a mechanical guard and no longer needs repeating. Both are kept
here because the `lessons` table is machine-local and gitignored, so without a
committed copy a fresh clone loses every lesson the project ever decided
against or graduated — including the reason it did. `devlog.py lessons import`
reads this file too, and restores each row with the status recorded below.

## [11] process (occurrences: 4, status: promoted)
**Mistake:** Ran 'devlog.py render-lessons' (step 3 of the /lesson command) without checking that devlog.sqlite actually held the lessons. dev/devlog.sqlite is gitignored and this machine's copy was near-empty -- 'lesson add' returned ids #1 and #2, and 'decision add' returned #3/#4 while the architecture doc was already at ADR-46. render-lessons rebuilds dev/LESSONS.md from the DB, so it overwrote the 8 checked-in lessons with just the 2 I had added. The tracked, durable artifact was destroyed by a command documented as a refresh. Recovered via the sanctioned path (git checkout dev/LESSONS.md, then 'devlog.py lessons import' to rehydrate the DB, then re-render), and verified by diffing lesson bodies against HEAD rather than trusting the count.
**Lesson:** Before running any command that regenerates a tracked file from a local, gitignored database (render-lessons and friends), confirm the DB is hydrated -- the file is the durable source of truth on a fresh clone, the DB is a local cache that can silently be empty or stale. If the DB is behind, import first, render second. Treat a suspiciously low autoincrement id in any devlog output (lesson #1, decision #3 when the doc is at 46) as the tell that the DB is not the source of truth.
**Rationale:** The header of dev/LESSONS.md says it plainly -- 'To rehydrate the DB from this file on a fresh clone, run devlog.py lessons import' -- which means the file, not the DB, survives a clone, and render-lessons is destructive whenever the DB is behind the file. The /lesson command's step order (add, then render) silently assumes a hydrated DB and gives no warning when it is not. The id numbers were the free early signal and I read past them. Same failure class as ADR-44 (a cwd-relative db_path silently opening a fresh empty DB) and lesson 'verify provenance of auto-written artifacts': a tool that regenerates from an empty store does not error, it just produces an authoritative-looking empty answer.

## [10] process (occurrences: 1, status: retired)
**Mistake:** uv init --package --python 3.12 silently wrote requires-python >=3.12, contradicting the doc-pinned 3.11 floor (tech stack table)
**Lesson:** After any scaffolding-tool run (uv init/add), diff the generated config against doc-pinned constraints before building on it
**Rationale:** Generator defaults silently override documented pins; catching drift at scaffold time costs seconds, catching it after downstream code depends on it costs a migration
