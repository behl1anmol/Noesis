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

## [7] process (occurrences: 1)
**Mistake:** M8: ran 'uv add jinja2 watchdog' before recording their rule-3 decision rows — the rows landed minutes later in the same session, but the hard rule's order (decision row, then dep) was inverted. Caught immediately after the uv output and corrected before any code used the deps.
**Lesson:** Before running any dependency-adding command (uv add / pip install into the project), record the decision row first — treat the /adr write as the gate that unlocks the install, not as paperwork to backfill, even when the dependency is pre-decided in the architecture doc.
**Rationale:** Rule 3's value is the forcing function: writing the rationale BEFORE the install is what catches a wrong version, a license problem, or a rejected alternative while reversal is still free (lesson 2's tree-sitter incident was exactly a consequence of install-first thinking). Backfilling inverts the check into a rubber stamp; the same-session correction was cheap this time only because the deps were genuinely pre-decided in Overview §4.9/§4.12.

## [6] correctness (occurrences: 1)
**Mistake:** M7 gitfast's first implementation took 'git status --porcelain -z --untracked-files=all' at its documented word — that it lists every untracked file individually — and built candidate matching on exact path equality. A smoke run against the real noesis repo showed a nested git worktree surfaced as a single '?? dir/' entry (-uall never expands nested repos; submodule changes likewise collapse to one gitlink path), so a changed file inside such a directory would have been silently carried forward as unchanged, breaking the fast path's identical-partitioning guarantee before any test existed for it.
**Lesson:** Before building logic on the output shape of an external tool (git plumbing, porcelain formats, CLI parsers), run the exact command against a real, messy instance of the data — not just fixtures or the documentation — and design the consumer to fail safe (here: directory entries match as prefixes, which can only widen the hash set, never shrink it).
**Rationale:** Documentation describes the common case; real repos contain the edge cases (nested repos, submodules, worktrees) that change output shape. The 2-minute smoke run on the live repo caught an invariant-breaking bug that the planned fixture tests (clean repos, no nesting) would have missed entirely. Fail-safe design direction matters too: when ambiguity remains, resolving it toward more hashing preserves correctness by construction (§3.2 rule 1).

## [5] testing (occurrences: 1)
**Mistake:** An M4 rerank latency of ~12s/query looked anomalously high, so I attributed it to a device fallback — first a CPU-only torch build, then sentence-transformers placing the model on CPU despite a T4. Both were wrong: device logging confirmed cuda, a tokenizer micro-benchmark ruled out my code (215ms), and a FLOP estimate (568M params x 512 tokens x 50 pairs at fp32 on a T4 = 7-14s) showed 12s was simply the honest model cost. I nearly wrote each wrong root cause into the docs.
**Lesson:** Before attributing an anomalous latency to a bug or misconfig, verify the cause by measurement: log the actual device, micro-benchmark the suspected hot path, and sanity-check against a first-principles FLOP/throughput estimate. A large number is often the genuine cost of a large model over many items, not a defect — confirm which before acting or documenting.
**Rationale:** Guessing a root cause from a number's magnitude alone produced two successive wrong hypotheses and almost put fabricated causes in the architecture doc (violates constraint A). A 30-minute measurement pass (device log + tokenizer timing + FLOP math) settled it definitively. Recording the compute device remains good practice — it is what let us rule out a fallback — but the binding lesson is measure-before-attributing.

## [3] process (occurrences: 1)
**Mistake:** The eval harness's write-baseline-if-missing convenience let a subagent's early background run (contaminated corpus: golden.yaml indexed, pre-fix labels) silently become the stored M2 baseline; the clean gate run then loaded it instead of writing its own, and the mismatch only surfaced because live-dense and stored-dense NDCG disagreed in the third decimal
**Lesson:** Before trusting any auto-written-on-absence artifact (baselines, caches, snapshots), verify its provenance matches the current methodology - after changing corpus, labels, or scoring, delete and regenerate such artifacts rather than letting stale ones be silently consumed
**Rationale:** Write-if-missing is provenance-blind by construction: it records whichever run got there first, not the run that matches the current method. A baseline is a measurement standard - a stale one corrupts every future gate comparison while looking perfectly healthy

## [2] process (occurrences: 1)
**Mistake:** Accepted uv's newer-minor resolution of tree-sitter-language-pack (1.12.2 vs doc-pinned 1.8.x) after checking only version numbers and licenses; the newer minor was a full pyo3 rewrite that downloads grammars over HTTP at runtime and changed the parsing API, discovered only mid-build by a subagent
**Lesson:** When a resolved dependency version diverges from the doc-pinned snapshot, check its changelog/release notes for behavior changes — especially network activity, API rewrites, and thread-safety — before building on it; a version-number diff alone is not verification
**Rationale:** Version drift can change runtime behavior, not just numbers: 1.12.2 silently introduced outbound HTTP from core/ (rule-2-adjacent) and a pyo3-unsendable parser that panicked under the indexer's worker thread. Both cost mid-milestone rework and a stakeholder decision; a 2-minute changelog read at uv-add time would have surfaced both

## [1] process (occurrences: 1)
**Mistake:** uv init --package --python 3.12 silently wrote requires-python >=3.12, contradicting the doc-pinned 3.11 floor (tech stack table)
**Lesson:** After any scaffolding-tool run (uv init/add), diff the generated config against doc-pinned constraints before building on it
**Rationale:** Generator defaults silently override documented pins; catching drift at scaffold time costs seconds, catching it after downstream code depends on it costs a migration
