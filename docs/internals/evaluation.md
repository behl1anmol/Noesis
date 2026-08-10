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
- Results are **grouped** by `file_path` in first-appearance order — several chunks of one file count as one retrieval and take one rank slot, but every retrieved chunk of that file stays available for matching ([ADR-67](../project/decisions.md)). Keeping only the file's best-ranked chunk discarded correct answers: the query `chunk_point_id` retrieved the chunk holding its definition at rank 2 and scored zero, because a *usage* chunk of the same file ranked first. Three of forty queries were pinned at zero by that alone, unable to register either a regression or an improvement.
- Inside a group, **one chunk credits at most one relevant item**. That is what lets a single file answer two labels through two different chunks, and it is also the limit: two labels of the same file that land inside the *same* chunk credit only one, capping that query at `1/len(relevant)`. Two labels are exposed to it, not one: `structural-03` (two async methods in `core/embedder.py`) and `structural-08` (two endpoints in `api/routes.py`). Neither pair currently shares a chunk — both reach 1.0 on at least one channel of the stored reference, which is only possible if their two items were credited through different chunks — so this is a documented limit rather than a live cap. The cap is deliberate — without it one wide chunk would sweep every label in its file and score 1.0 for retrieving a single thing — and it depends on chunk boundaries rather than on retrieval quality, so read a `0.5` on a two-label query as "check the chunking", not "retrieval halved".
- **Recall@5 / Recall@10**: fraction of a query's relevant items matched in the top k, averaged over queries.
- **NDCG@10**: binary gains with greedy credit — walking the deduped ranking, a result gains 1 only the first time it matches a not-yet-credited relevant item; IDCG assumes all relevant items ranked first (log2 discount).
- **Latency p50/p95 (ms)**: wall time of the full search call per query, nearest-rank percentiles. Latency is *reported next to* quality but never mixed into the quality gate — they are separate stakeholder decisions.

## The gate ([ADR-65](../project/decisions.md))

The golden run **asserts**. Until issue #38 it checked only that categories were present and values sat in `[0, 1]`, so a run whose every metric had halved reported `1 passed`; the real gate was a human reading a printed table that pytest's stdout capture swallows under every invocation without `-s`. Three layers now:

| Layer | Compares | Fires on |
|---|---|---|
| **1 — relational** | channels within the *same run* | hybrid losing to dense on Recall@10 overall or on the symbol subset (M3's exit criterion), or rerank losing to hybrid on NDCG@10 (ADR-35's shipped claim) |
| **2 — regression** | this run vs `baselines/reference.json` | any metric dropping by more than *both* a relative band (10 % overall, 20 % per category) *and* one query's worth in absolute terms |
| **3 — re-baseline** | — | never automatic; `NOESIS_EVAL_REBASELINE=1` only, and it refuses a dirty tree |

Layer 1 is corpus-independent by construction — every channel sees the same corpus, labels and models — which is why it always asserts, including during a re-baseline. Layer 2 is the one that catches *uniform* degradation, and it only runs when the stored reference is comparable: same models, same store implementation, same set of questions, and a corpus within a 20 % chunk-count band. Otherwise the run **fails** (never skips) with the tables marked **NOT A GATE** and the exact re-baseline command. That guard is the fix for a stale baseline having gone unnoticed for a month: recall against a fixed label set falls mechanically as the corpus grows, so an older reference is not a weaker reference, it is a different experiment.

The absolute floor beside the relative band is deliberate arithmetic. The gate measures a relative drop, but one query moving changes the mean by `1/n` absolutely — which at a low score is a huge *relative* figure and at a high score a small one. Requiring both means "one query is noise" stays true at every score level.

### The current reference (2026-08-09)

Measured under content-anchored labels ([ADR-64](../project/decisions.md)) and grouped scoring ([ADR-67](../project/decisions.md)) — 184 files, 559 chunks, embedded Qdrant, CodeRankEmbed and bge-reranker-v2-m3 both on CUDA at their default batch sizes:

| channel | Recall@5 | Recall@10 | NDCG@10 | p50 latency |
|---|---|---|---|---|
| dense | 0.575 | 0.637 | 0.458 | 45 ms |
| dense (python-only) | 0.700 | 0.713 | 0.570 | 49 ms |
| sparse | 0.613 | 0.662 | 0.461 | 29 ms |
| **hybrid** | 0.662 | **0.775** | 0.502 | 80 ms |
| **hybrid+rerank** | 0.775 | **0.825** | 0.583 | 8 551 ms |

!!! note "What re-anchoring three labels did to the numbers: nothing measurable"
    This reference replaces the 2026-08-08 one because the PR #42 review moved `nl-07`, `nl-13` and `structural-04` off docstring prose onto signature lines, which changes `golden_sha256` and so makes the older reference incomparable by construction. **Every Recall figure is identical across all five channels and all three categories**, and hybrid and hybrid+rerank are identical on NDCG too. Only dense and sparse NDCG@10 moved, by at most **0.010** — the largest single move is dense/`symbol` at −0.0094, which reads as 0.009 only at this page's three decimals — rank-order noise, not a change in what matched. That is the expected result: widening a label inside a file it already labelled cannot change that file's group rank, and group rank is what NDCG credits.

    Latency rose on every channel, but **not** by a common factor, and the exception is the informative one: dense ×2.96, `dense (python-only)` ×2.83, hybrid ×2.42, sparse ×2.20 — and `hybrid+rerank` only **×1.30** (6581 → 8551 ms). The channel dominated by model forward time barely moved, while `sparse`, which runs no model at all, more than doubled (13 ms → 29 ms). That rules retrieval out as the cause and points at machine state: the two runs used different batch sizes (8/4 then, the defaults 32/16 now) on an 8 GB GPU where the warning box below applies, but nothing here isolates which factor dominates, and this report cannot. Latency is reported and never gated, precisely so a difference nobody has explained cannot fail a run.

**M3's exit criterion, asserted rather than assumed for the first time:** hybrid beats dense on Recall@10 overall (0.775 vs 0.637) and on the symbol subset (0.857 vs 0.786). With rerank, symbol Recall@10 reaches **1.000**.

**What the python-only diagnostic says about corpus growth.** Filtering to Python removes 99 of 184 files — every non-`.py` distractor — and lifts Recall@10 by **+0.075** overall (`0.7125 − 0.6375`; the previous text read +0.076, which was the displayed three-decimal figures subtracted rather than the stored ones). The effect is concentrated where you would expect: `structural` **+0.167**, `nl` +0.071, `symbol` **0.000**. Identifier lookups do not compete with prose; structure-phrased questions very much do. That is the measured form of the issue's Finding B, and it is only visible because the channel exists.

These numbers are **not** comparable to `m2_dense.json`: the labels changed identity and the scorer changed semantics. That is why the M2 file is frozen rather than refreshed.

### Embedded Qdrant does not rank like a real server

Measured 2026-08-08 at commit `13fcd9d` against `qdrant/qdrant:v1.18.3` — the question [ADR-62](../project/decisions.md) left open and assigned to this work. Both sides of the comparison were measured on the same corpus, labels and models **as they stood that day**, i.e. under the pre-re-anchoring label set; the pairing is therefore still sound and the delta still holds, but the absolute figures are the 2026-08-08 ones, not the table above. It has not been re-measured, because doing so needs a second container run and the delta, not the absolutes, is what the section is about:

| channel | embedded | real server | delta |
|---|---|---|---|
| dense | 0.637 | 0.637 | 0.000 |
| dense (python-only) | 0.713 | 0.713 | 0.000 |
| sparse | 0.662 | 0.650 | −0.012 |
| **hybrid** | **0.775** | **0.713** | **−0.062** |
| hybrid+rerank | 0.825 | 0.825 | 0.000 |

Dense is bit-identical — it is a plain vector query, so the two implementations agree. Sparse and hybrid are not, through exactly the mechanism ADR-62 predicted: BM25 IDF is computed **server-side** (`Modifier.IDF`) and RRF fusion uses a constant the server fixes and `FusionQuery` does not expose. **The embedded harness over-reports hybrid Recall@10 by 0.062** — about 2.5 queries of 40 — relative to what production actually serves. `hybrid+rerank` converges again at 0.825, because the cross-encoder reorders the union and absorbs the fusion difference.

This is why `store` is a hard comparability term: an embedded-measured reference must never gate a server-measured run. Observed doing so — L1 passed, L2 refused with `store kind: reference 'embedded' != run 'server'`, and the run failed with the tables marked NOT A GATE.

**Open, deliberately:** the committed reference is embedded-measured, because that is what the harness does without Docker. Whether the canonical reference should instead be server-measured — matching production at the cost of requiring a container — is a decision worth taking on its own evidence now that the gap is quantified.

### Baselines

- `tests/eval/baselines/reference.json` — the single **living** reference. All channels plus a provenance block: corpus (files, chunks, commit, path-manifest hash), models, store kind and server version, a digest of the golden questions, device, and the run's **duration** (`measure_s`, the five channels' evaluation loop; `total_s`, adding collection, the corpus build and both model loads). Duration is recorded as evidence and is never a comparability term — a reference must not stop gating because the machine was busier that day. It exists so a claim about how long the tier takes can be checked against a committed artifact instead of quoted from memory.
- `tests/eval/baselines/m2_dense.json` — a **frozen** historical record, read by nothing.

No run writes a baseline implicitly. Both provenance-blind writers are gone: the write-if-missing that made whichever run got there first the standard (the mechanism lesson 8 was recorded for), and two unconditional writes that dirtied the tree on every run.

**Pass `-s`.** The tier prints its compute device and each channel's Recall@10 as it completes, and every print flushes — but pytest captures stdout by default and releases it only on failure, so without `-s` a thirteen-minute run shows a bare `tests/eval/test_golden.py` and nothing else. A run with no progress signal is indistinguishable from a hung one; that has already cost wrong diagnoses. `-q` is *not* the culprit and never was: measured on a scratch test, bare `pytest` hides the print exactly as `-q` does, while both `-s` and `-s -q` show it. The suppressor is the capture, which is why the landed fix adds `-s` everywhere rather than dropping `-q`. This is the same shape as the swallowed-table bug ADR-65 was written for: the fix that made the run legible was defeated by the invocation the docs recommended.

Every run also writes `tests/eval/report_latest.md` — verdicts first, then provenance, then the tables — plus `report_latest.json`. Both are gitignored. The report lists any query scoring zero on *every* channel, because four channels failing the same query is far likelier to be a broken label than a retrieval failure; printing that is what would have surfaced the label rot at the first run rather than two milestones later.

## Running

```bash
uv run pytest tests/eval/ -m golden -s   # golden harness (loads the real model, self-indexes this repo)
uv run pytest                            # default suite: label integrity + gate logic, fully offline
uv run pytest -m integration             # opt-in: real embedding model

# Record a new reference — deliberate, needs a clean tree and a decision row
NOESIS_EVAL_REBASELINE=1 uv run pytest tests/eval/ -m golden -s

# Measure against a real Qdrant server instead of the embedded client (ADR-66)
docker compose up -d
NOESIS_EVAL_QDRANT_URL=http://127.0.0.1:6333 uv run pytest tests/eval/ -m golden -s

# Place the models — needed on a GPU with less headroom than the 16 GB T4 the
# M4 numbers were measured on. Unset means today's behaviour (auto-detect,
# default batch sizes). Placement is recorded in the run's provenance.
NOESIS_EVAL_RERANKER_DEVICE=cpu NOESIS_EVAL_EMBED_BATCH_SIZE=8 \
  uv run pytest tests/eval/ -m golden -s

# The embedder has its own pair of knobs — the reranker is the bigger model, so
# it is the usual one to move, but the OOM below is embedder-side.
NOESIS_EVAL_EMBEDDER_DEVICE=cpu NOESIS_EVAL_RERANK_BATCH_SIZE=2 \
  uv run pytest tests/eval/ -m golden -s
```

### Model placement variables

All four are read only by the golden harness (`tests/eval/test_golden.py`), never by the service. An empty value counts as unset; a batch size that is not an integer ≥ 1 is rejected rather than ignored.

| variable | default when unset | what it moves |
|---|---|---|
| `NOESIS_EVAL_EMBEDDER_DEVICE` | auto-detect (`cuda` when available) | where the embedder runs — `cpu`, `cuda`, `cuda:1` |
| `NOESIS_EVAL_RERANKER_DEVICE` | auto-detect | where the cross-encoder runs |
| `NOESIS_EVAL_EMBED_BATCH_SIZE` | 32 | embedder batch — the lever for indexing-time device memory |
| `NOESIS_EVAL_RERANK_BATCH_SIZE` | 16 | cross-encoder batch — the lever for query-time device memory |

!!! warning "The harness needs real GPU headroom"
    Both models are resident at once — a 137M embedder and a 568M cross-encoder — and the harness constructs them directly rather than through `config.toml`, so until the four variables above existed there was no way to place them. On an 8 GB laptop GPU, indexing this corpus drove device memory to **7600 MiB of 8151** and WSL2's paravirtualization layer failed an allocation with `dxgkio_make_resident: Ioctl failed: -12` (ENOMEM). CUDA reported that as `cudaErrorIllegalInstruction`, not as an out-of-memory error, and sustained pressure took the host GPU driver down with it. If you see an illegal-instruction fault from a model forward pass, check `nvidia-smi` before suspecting the code. The pressure is highest during indexing, so reach for `NOESIS_EVAL_EMBED_BATCH_SIZE` first; if it persists, move the reranker off the device with `NOESIS_EVAL_RERANKER_DEVICE=cpu`, and as a last resort the embedder too with `NOESIS_EVAL_EMBEDDER_DEVICE=cpu` (which makes the run very slow, but finishes).

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
