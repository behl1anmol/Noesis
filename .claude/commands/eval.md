---
description: Run the golden evaluation harness and report its gate verdicts
allowed-tools: Bash(uv run pytest *:*), Bash(python .claude/scripts/devlog.py *:*), Read
---
1. Run `uv run pytest tests/eval/ -m golden -s` (the golden-set harness). It loads
   the real embedder and reranker; the last recorded run took 11 m 31 s on a
   CUDA GPU (`provenance.duration` in `tests/eval/baselines/reference.json`),
   and far longer on CPU.
2. Read `tests/eval/report_latest.md` — verdicts first, then provenance, then
   the tables. Do NOT report from the terminal output alone: pytest captures
   stdout, so the printed tables reach a terminal only when `-s` is passed, and
   tables being swallowed is the bug ADR-65 was written for. (`-q` is innocent
   — measured: bare `pytest` hides the print too, and `-s -q` shows it.)
   The verdict block also states whether the run **re-baselined**; a run that
   wrote the reference gated nothing, so never read its tables as a pass.
3. Report the three layers as the harness decided them, not as you judge them:
   - L1 relational (same-run): hybrid vs dense on Recall@10 overall and on the
     symbol subset, and hybrid+rerank vs hybrid on NDCG@10.
   - L2 regression vs `tests/eval/baselines/reference.json` — and whether it
     ran at all. If the reference was incomparable the run fails and the
     tables are marked **NOT A GATE**; say that, and say why it was
     incomparable. An ungated table is not a passing one.
   - Any query with zero recall on every channel. Four channels missing the
     same query is a label to inspect before it is a retrieval regression.
4. Report Recall@5, Recall@10 and NDCG@10 per category, plus p50/p95 latency
   and the compute device. A latency number without a recorded device is not a
   measurement.

**This command cannot re-baseline, by design.** Recording a new reference is
`NOESIS_EVAL_REBASELINE=1 uv run pytest tests/eval/ -m golden -s`, which this
command's `allowed-tools` cannot express. If the numbers should become the new
reference, say so and let the human run it — then it needs a decision row:
`python .claude/scripts/devlog.py decision add --title "golden reference <date>"
--decision "<numbers + corpus provenance>" --rationale "<why re-baseline now>"`.

Numbers or it didn't happen — do not report an eval as passing without
running the harness this turn, and do not read a passing test as a quality
verdict without reading the verdict block.
