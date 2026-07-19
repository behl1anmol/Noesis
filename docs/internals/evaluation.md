# Evaluation harness

Quality claims in Noesis are measured, not assumed: `tests/eval/` holds a human-labeled golden set, a metrics harness, and stored baselines. Every retrieval-quality decision — hybrid vs dense, reranker on/off — was gated on these numbers.

## The golden set

`tests/eval/golden.yaml` — 40 queries over this repository itself (the harness self-indexes Noesis), in three categories:

| Category | Count | Example intent |
|---|---|---|
| `nl` | 14 | natural-language ("where is the run crash recovery?") |
| `symbol` | 14 | identifier lookups (exact names, casings) |
| `structural` | 12 | AST-pattern queries |

Each retrieval query lists its relevant items (`path`, optional `lines` range); loading is fail-loud — a missing id, bad category, or empty relevant list raises rather than silently skewing gate numbers. A separate `structural_patterns` section carries ast-grep patterns with **exact expected per-file match counts**, evaluated pass/fail outside the retrieval metrics — pattern matching is exact, so partial credit would only hide regressions.

## Metrics

Scoring rules are stated in `tests/eval/harness.py` so numbers are reproducible:

- A result matches a relevant item iff `file_path` is equal and, when the item carries a `lines` range, the result span overlaps it.
- Results are deduplicated by `file_path` keeping the best rank — several chunks of one file count as one retrieval.
- **Recall@5 / Recall@10**: fraction of a query's relevant items matched in the top k, averaged over queries.
- **NDCG@10**: binary gains with greedy credit — walking the deduped ranking, a result gains 1 only the first time it matches a not-yet-credited relevant item; IDCG assumes all relevant items ranked first (log2 discount).
- **Latency p50/p95 (ms)**: wall time of the full search call per query, nearest-rank percentiles. Latency is *reported next to* quality but never mixed into the quality gate — they are separate stakeholder decisions.

Baselines are stored as JSON (`tests/eval/baselines/`, e.g. `m2_dense.json`) with `save_baseline`/`load_baseline`; `format_delta` renders a challenger-vs-baseline table for gate reviews.

## Running

```bash
uv run pytest tests/eval/ -m golden      # golden harness (loads the real model, self-indexes this repo)
uv run pytest                            # default suite: ~300 tests, fully offline
uv run pytest -m integration             # opt-in: real embedding model
```

The default suite runs against `FakeEmbedder` and an in-memory Qdrant — no model download, no Docker. The `integration` and `golden` marks are excluded by default.

## The M4 reranker gate — measured decision

The flagship use of the harness (full data: `architecture-docs/m4-reranker-benchmarks.md`; Colab T4 16 GB, fp32, both models confirmed on `cuda` in-run):

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
- Latency and quality are always reported together and gated separately.
- Baselines carry device provenance in their metadata: a latency number without a recorded device is not a measurement.
