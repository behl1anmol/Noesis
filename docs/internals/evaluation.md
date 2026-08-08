# Evaluation harness

Quality claims in Noesis are measured, not assumed: `tests/eval/` holds a human-labeled golden set, a metrics harness, and stored baselines. Every retrieval-quality decision — hybrid vs dense, reranker on/off — was gated on these numbers.

## The golden set

`tests/eval/golden.yaml` — 40 queries over this repository itself (the harness self-indexes Noesis), in three categories:

| Category | Count | Example intent |
|---|---|---|
| `nl` | 14 | natural-language ("where is the run crash recovery?") |
| `symbol` | 14 | identifier lookups (exact names, casings) |
| `structural` | 12 | AST-pattern queries |

Each retrieval query lists its relevant items as a `path` plus an **`anchor`** — a substring occurring exactly once in that file — and optionally an `anchor_end` widening the label to a region. `load_golden` resolves anchors to line numbers against the tree being measured. Loading is fail-loud: a missing id, bad category, empty relevant list, or an anchor that matches zero or several lines raises rather than silently skewing gate numbers. A separate `structural_patterns` section carries ast-grep patterns with **exact expected per-file match counts**, evaluated pass/fail outside the retrieval metrics — pattern matching is exact, so partial credit would only hide regressions.

### Why labels address content, not line numbers ([ADR-64](../project/decisions.md))

Labels used to store `lines: [start, end]`. Because a result must overlap that range to count, a label whose code moved silently forced its query to zero — and by 2026-08, **22 of 46 labels had drifted off their own ranges**, with two that never matched anything from the day they were written. Nothing detected it, while `structural_patterns` in the very same file stayed correct across three commits because a default-suite test pinned it.

So the labels are content-addressed and `tests/eval/test_golden_labels.py` runs **in the default suite**, on every pull request: it re-resolves every anchor against the working tree, checks each labeled file is one the corpus actually indexes, and requires a `symbol` query's identifier to appear inside its own labeled span. Rot now fails on the commit that causes it. `tests/eval/migrate_labels.py` records how the original 46 labels were mechanically re-derived.

Width follows the question: a `symbol` query is an identifier lookup, so its label is the definition site. `nl` and `structural` queries ask about a region, and there the width matters — deduplication keeps only the best-ranked chunk per file, so a one-line label would score zero whenever the right region was retrieved but ranked behind another chunk of the same file.

## Metrics

Scoring rules are stated in `tests/eval/harness.py` so numbers are reproducible:

- A result matches a relevant item iff `file_path` is equal and, when the item carries a `lines` range, the result span overlaps it.
- Results are deduplicated by `file_path` keeping the best rank — several chunks of one file count as one retrieval.
- **Recall@5 / Recall@10**: fraction of a query's relevant items matched in the top k, averaged over queries.
- **NDCG@10**: binary gains with greedy credit — walking the deduped ranking, a result gains 1 only the first time it matches a not-yet-credited relevant item; IDCG assumes all relevant items ranked first (log2 discount).
- **Latency p50/p95 (ms)**: wall time of the full search call per query, nearest-rank percentiles. Latency is *reported next to* quality but never mixed into the quality gate — they are separate stakeholder decisions.

## The gate ([ADR-65](../project/decisions.md))

The golden run **asserts**. Until issue #38 it checked only that categories were present and values sat in `[0, 1]`, so a run whose every metric had halved reported `1 passed`; the real gate was a human reading a printed table that the documented `-q` invocation suppresses. Three layers now:

| Layer | Compares | Fires on |
|---|---|---|
| **1 — relational** | channels within the *same run* | hybrid losing to dense on Recall@10 overall or on the symbol subset (M3's exit criterion), or rerank losing to hybrid on NDCG@10 (ADR-35's shipped claim) |
| **2 — regression** | this run vs `baselines/reference.json` | any metric dropping by more than *both* a relative band (10 % overall, 20 % per category) *and* one query's worth in absolute terms |
| **3 — re-baseline** | — | never automatic; `NOESIS_EVAL_REBASELINE=1` only, and it refuses a dirty tree |

Layer 1 is corpus-independent by construction — every channel sees the same corpus, labels and models — which is why it always asserts, including during a re-baseline. Layer 2 is the one that catches *uniform* degradation, and it only runs when the stored reference is comparable: same models, same store implementation, same set of questions, and a corpus within a 20 % chunk-count band. Otherwise the run **fails** (never skips) with the tables marked **NOT A GATE** and the exact re-baseline command. That guard is the fix for a stale baseline having gone unnoticed for a month: recall against a fixed label set falls mechanically as the corpus grows, so an older reference is not a weaker reference, it is a different experiment.

The absolute floor beside the relative band is deliberate arithmetic. The gate measures a relative drop, but one query moving changes the mean by `1/n` absolutely — which at a low score is a huge *relative* figure and at a high score a small one. Requiring both means "one query is noise" stays true at every score level.

### Baselines

- `tests/eval/baselines/reference.json` — the single **living** reference. All channels plus a provenance block: corpus (files, chunks, commit, path-manifest hash), models, store kind and server version, a digest of the golden questions, and device.
- `tests/eval/baselines/m2_dense.json` — a **frozen** historical record, read by nothing.

No run writes a baseline implicitly. Both provenance-blind writers are gone: the write-if-missing that made whichever run got there first the standard (the mechanism lesson 8 was recorded for), and two unconditional writes that dirtied the tree on every run.

Every run also writes `tests/eval/report_latest.md` — verdicts first, then provenance, then the tables — plus `report_latest.json`. Both are gitignored. The report lists any query scoring zero on *every* channel, because four channels failing the same query is far likelier to be a broken label than a retrieval failure; printing that is what would have surfaced the label rot at the first run rather than two milestones later.

## Running

```bash
uv run pytest tests/eval/ -m golden      # golden harness (loads the real model, self-indexes this repo)
uv run pytest                            # default suite: label integrity + gate logic, fully offline
uv run pytest -m integration             # opt-in: real embedding model

# Record a new reference — deliberate, needs a clean tree and a decision row
NOESIS_EVAL_REBASELINE=1 uv run pytest tests/eval/ -m golden

# Measure against a real Qdrant server instead of the embedded client (ADR-66)
docker compose up -d
NOESIS_EVAL_QDRANT_URL=http://127.0.0.1:6333 uv run pytest tests/eval/ -m golden
```

The default suite runs against `FakeEmbedder` and an in-memory Qdrant — no model download, no Docker. The `integration` and `golden` marks are excluded by default.

**No layer of the golden gate runs in CI**, and that is deliberate: the tier takes roughly two hours and multi-GB model weights, and a CPU run would produce latency numbers comparable to no recorded device. The part that *is* cheap — label integrity — is unmarked and therefore runs in the `tests` job on every pull request. Anything that depends on measured retrieval quality is run by a human, on purpose, and recorded.

## The M4 reranker gate — measured decision

The flagship use of the harness (full data: `architecture-docs/m4-reranker-benchmarks.md`; Colab T4 16 GB, fp32, both models confirmed on `cuda` in-run).

!!! note "Frozen historical record"
    These numbers were measured on 2026-07-04, against a corpus of ~94 files and the pre-[ADR-64](../project/decisions.md) positional label set — two of whose labels could never match anything. They are kept because they are the evidence behind ADR-35, and the *relations* they establish are still asserted by the gate's layer 1 on every run. They are **not** the live comparison; that is `baselines/reference.json`.

**Quality — decisive, uniform win** (same-run hybrid vs hybrid+rerank, zero regressions):

| Metric (overall) | hybrid | hybrid+rerank | delta |
|---|---|---|---|
| Recall@5 | 0.775 | 0.863 | +0.088 |
| Recall@10 | 0.787 | 0.875 | +0.088 |
| NDCG@10 | 0.620 | 0.726 | **+0.106** |

**Latency — disqualifying as a default** (full search call per query):

| Channel | p50 | p95 |
|---|---|---|
| sparse | 7.2 ms | 8.2 ms |
| dense | 19.4 ms | 25.6 ms |
| hybrid | 60.2 ms | 67.4 ms |
| hybrid+rerank | **12 180 ms** | **13 400 ms** |

The ~12 s is intrinsic, not a defect — three independent checks agree: device confirmed CUDA, tokenization ruled out (~215 ms), and the FLOP lower bound for a 568 M-parameter cross-encoder scoring 50 pairs at fp32 on a T4 lands at 7.2–14.4 s, bracketing the observed 12.2 s.

**Decision ([ADR-35](../project/decisions.md), sequencing per [ADR-19](../project/decisions.md)):** quality gate passed, latency ~27× over the 500 ms p95 budget → the reranker ships **default-off, per-request opt-in** (`rerank: true`). The measured win justifies keeping the feature; the latency forbids making it the default.

## Key invariants

- No quality feature ships default-on without beating the stored baseline on this harness.
- Golden labels fail loudly on any malformation — corrupted gate numbers are worse than no numbers.
- **Labels address content, never positions.** A line number is a claim that rots silently; an anchor that stops resolving raises.
- **The check lives in the tier that actually runs.** Maintenance follows enforcement, not intent — the half of `golden.yaml` with a default-suite test stayed correct while the half without one rotted.
- **A baseline is only a reference if its provenance says it is comparable**, and no run may write one implicitly.
- Latency and quality are always reported together and gated separately.
- Baselines carry device provenance in their metadata: a latency number without a recorded device is not a measurement.
