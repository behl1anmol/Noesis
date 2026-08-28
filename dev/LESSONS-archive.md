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

## [16] verification (occurrences: 4, status: promoted)
**Mistake:** While measuring the ADR-59 fix I wrote a smoke script that printed 'fds after telemetry close: 2 (runtime conn only)'. The label was a claim -- that the writer's handle had been released -- and the number actually showed the opposite. The real cause was SQLite's deferred close: unixClose parks an fd on a pending-close list while another connection in the same process still holds POSIX locks on that inode, so the writer's fd only disappeared when the runtime conn closed. Nothing was wrong with the code; the annotation on the evidence was wrong, and I nearly quoted it into a PR as proof.
**Lesson:** A measurement's label is a claim, and it needs verifying separately from the number. Before quoting an instrument (fd counts, row counts, timings, queue sizes) as evidence, prove what the number counts -- perturb it and check it moves the way the label says. Here, one probe settled it: a leaked fd owned by a dead thread cannot be released by closing a DIFFERENT connection, so the fact that conn.close() cleared it disproved 'leak' outright.
**Rationale:** This is lesson 14's failure mode reached through the evidence rather than the test: an assertion that passes for a reason you did not check proves nothing, and a measurement that reads plausibly for a reason you did not check is worse -- it ships as a claim in a PR body or a decision record, where future readers trust it. PR #24's round 8 found exactly two such claims already shipped (SRV-2's mechanism in configuration.md, and ADR-51's 'exactly one carve-out'), which is the standing evidence that this class of error survives review in this repo.

## [11] process (occurrences: 4, status: promoted)
**Mistake:** Ran 'devlog.py render-lessons' (step 3 of the /lesson command) without checking that devlog.sqlite actually held the lessons. dev/devlog.sqlite is gitignored and this machine's copy was near-empty -- 'lesson add' returned ids #1 and #2, and 'decision add' returned #3/#4 while the architecture doc was already at ADR-46. render-lessons rebuilds dev/LESSONS.md from the DB, so it overwrote the 8 checked-in lessons with just the 2 I had added. The tracked, durable artifact was destroyed by a command documented as a refresh. Recovered via the sanctioned path (git checkout dev/LESSONS.md, then 'devlog.py lessons import' to rehydrate the DB, then re-render), and verified by diffing lesson bodies against HEAD rather than trusting the count.
**Lesson:** Before running any command that regenerates a tracked file from a local, gitignored database (render-lessons and friends), confirm the DB is hydrated -- the file is the durable source of truth on a fresh clone, the DB is a local cache that can silently be empty or stale. If the DB is behind, import first, render second. Treat a suspiciously low autoincrement id in any devlog output (lesson #1, decision #3 when the doc is at 46) as the tell that the DB is not the source of truth.
**Rationale:** The header of dev/LESSONS.md says it plainly -- 'To rehydrate the DB from this file on a fresh clone, run devlog.py lessons import' -- which means the file, not the DB, survives a clone, and render-lessons is destructive whenever the DB is behind the file. The /lesson command's step order (add, then render) silently assumes a hydrated DB and gives no warning when it is not. The id numbers were the free early signal and I read past them. Same failure class as ADR-44 (a cwd-relative db_path silently opening a fresh empty DB) and lesson 'verify provenance of auto-written artifacts': a tool that regenerates from an empty store does not error, it just produces an authoritative-looking empty answer.

## [14] testing (occurrences: 3, status: promoted)
**Mistake:** Wrote a regression test for a log-message defect by asserting the NEW wording was absent ('accepting the empty scan'). Against the pre-fix source it passed vacuously — the old code emitted 'accepting as deletion', a string the assertion never looked for. Only the stash-and-rerun step exposed it; without that the test would have shipped proving nothing.
**Lesson:** When a fix changes a user-visible string (log line, error message, status text), assert on the invariant the string expresses, not on the new phrasing. Match the bare verb or the semantic fact ('accepting'), or assert against both old and new wording. Every new regression test must be run against the pre-fix source and observed to FAIL for the reason it was written; a test that merely passes both ways is only valid when it was deliberately written as an over-correction guard.
**Rationale:** A test asserting the absence of a string that only the fixed code can produce is a tautology against the buggy code. It reports green, is counted as coverage, and pins nothing — which is worse than no test, because it stops anyone from writing the real one.

## [13] correctness (occurrences: 1, status: retired)
**Mistake:** ADR-52 moved telemetry writes onto a dedicated writer thread with its OWN sqlite connection, but state.log_query never commits — it had always relied on a later commit() by some other operation on the SHARED connection. On a private connection nothing else ever commits, so every telemetry row was rolled back at conn.close(). Zero rows reached disk; found in round 5 only because a new test read the count back through a fresh connection.
**Lesson:** When moving a write onto a new connection, check who commits it. A write that was durable only as a side effect of another caller's commit on the same connection becomes a silent data-loss bug the moment it gets its own handle. Assert durability the way a reader outside the process would: read it back through a SECOND connection, never through the one that wrote it.
**Rationale:** sqlite3's default isolation_level opens an implicit transaction on DML and discards it on close without commit. Same-connection reads see uncommitted rows, so every test that read through the writing connection passed while the data was being thrown away. The bug survived two review rounds over this exact module because no test ever crossed a connection boundary.

## [10] process (occurrences: 1, status: retired)
**Mistake:** uv init --package --python 3.12 silently wrote requires-python >=3.12, contradicting the doc-pinned 3.11 floor (tech stack table)
**Lesson:** After any scaffolding-tool run (uv init/add), diff the generated config against doc-pinned constraints before building on it
**Rationale:** Generator defaults silently override documented pins; catching drift at scaffold time costs seconds, catching it after downstream code depends on it costs a migration
