# Proposed lesson to add — cap decision needed

Recorded in `dev/devlog.sqlite` as lesson **#22** while fixing issue #43. It
pushed active lessons from 15 to **16**, over the ADR-27 cap. `render-lessons`
already reflects it in the committed `dev/LESSONS.md` (it renders regardless
of the cap and just warns), but per the `/lesson` skill's own instruction —
*"If it exceeds 15, propose which lesson to retire or promote — do not
silently exceed the cap"* — the retire/promote call is yours to make. This
file is not committed; delete it once you've decided.

## Lesson #22 (full content, as stored)

**Category:** design
**Status:** active
**Occurrences:** 1

**Mistake:**
While fixing issue #43 (structural_search's discovery-time-charged-to-scan-budget
bug), the first implementation pass added a new `discovery_s: float` field
directly onto `StructuralResult`, exposing per-call wall-clock timing in the
REST/MCP response body. This broke
`test_mcp.py::test_structural_search_identical_to_rest` — its REST call and
MCP call are two separate live scans, so their `discovery_s` readings differ,
and the byte-identity assertion the M6 milestone's design goal rests on
failed. The field also silently diverged from an existing, unchecked
convention: no other core surface returns wall-clock timing in its response
body (search latency is telemetry-only via `ctx.telemetry.record_query`;
`indexer.py` already solves the identical "report discovery timing" problem
with `logger.info("... discovery took=%.1fs discovered=%d", ...)` rather than
a return value).

**Lesson:**
Before adding a field to a response shape asserted byte-identical across
adapters (REST vs MCP, or any two callers of the same core function), grep
for how the codebase already reports similar values elsewhere (the sibling
case — here, existing latency/duration handling) instead of inventing the
field's placement from the issue text alone. Wall-clock/duration values
belong in logs or telemetry, never in a response body compared for
cross-call identity.

**Rationale:**
`test_mcp.py`'s own module docstring states the M6 design goal: "for the same
query the two surfaces must return byte-identical bodies." `indexer.py` had
already solved the identical problem (reporting discovery timing) via
`logger.info` instead of a return value — a short grep before writing the new
field would have surfaced the precedent. Caught immediately by the mandatory
full-suite run (rule 9's mechanism working as intended) rather than by a
reviewer, but the same mistake reaching PR review would have cost a round,
the way the PR #42 review history repeatedly shows for less avoidable
versions of the same class: reinventing a shape the codebase already has a
convention for.

## Suggested candidate to retire: lesson #12

**Category:** implementation · **Occurrences:** 1

> **Mistake:** While fixing the telemetry write path, a fix batch opened its
> dedicated best-effort connection with a raw `sqlite3.connect()` call
> instead of `state.connect()`. The codebase enforces a single sqlite
> connection constructor, and the structural-search golden test `sp-02` pins
> the AST pattern `sqlite3.connect($$$ARGS)` to exactly `{state.py: 1}`. The
> second call site broke that golden test; it was caught only by the
> mandatory full-suite run after all four fix batches had landed.
>
> **Lesson:** Open new resources through the codebase's existing sanctioned
> constructor, never by calling the underlying library primitive a second
> time...

**Why this one:** it's the narrowest-scoped and most mechanically-idiosyncratic
of the 15 — and, unlike the others, its own corrective mechanism is already
load-bearing in the codebase independent of the lesson staying injected:
`tests/eval/golden.yaml`'s `sp-02` structural pattern pins
`sqlite3.connect($$$ARGS)` to exactly one occurrence (`state.py`), so a second
raw `sqlite3.connect()` call site fails the default suite on its own, with or
without an agent having read this lesson first. Retiring it trades a
context-window line for zero loss of enforcement. (Lessons 3 and 15 have a
similar shape — each gained a structural guard after the fact — but their
underlying principles, "test the layer the user perceives" and "persist a
verdict rather than re-deriving it," are broadly reusable across the
codebase in a way #12's single-constructor rule isn't; #12 is the cleanest
cut.)

**If you agree**, retire it and add #22:

```
python .claude/scripts/devlog.py lesson retire 12
python .claude/scripts/devlog.py render-lessons
```

(#22 is already in the DB from this session — `render-lessons` will pick it
up once #12 is out and the count is back to 15. No `lesson add` needed.)

**If you'd rather promote #12** instead of retiring it (its 1-occurrence
count doesn't meet the usual ≥3 promotion bar rule 7/9 used for lessons 14/16,
so this would be a discretionary early promotion given the golden test
already backs it) — note `lesson promote` only flips its DB status out of
`active` the same way `retire` does; it does **not** itself add anything to
`CLAUDE.md`. A real promotion (as lessons 14/16 → rule 9) also means you
hand-write the new hard rule into `CLAUDE.md` yourself:

```
python .claude/scripts/devlog.py lesson promote 12
```

Either way, re-run `render-lessons` afterward to refresh the committed
`dev/LESSONS.md` / `dev/LESSONS-archive.md`, then `git add`/commit those two
files (they're the durable, tracked record — `devlog.sqlite` itself is
gitignored).
