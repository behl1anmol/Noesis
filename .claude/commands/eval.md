---
description: Run the M3 evaluation harness and report metrics against the stored baseline
allowed-tools: Bash(uv run pytest *:*), Bash(python .claude/scripts/devlog.py *:*), Read
---
1. Run `uv run pytest tests/eval/ -m golden` (the golden-set harness).
2. Report Recall@10, Recall@5, and NDCG@10 per query category (NL / symbol /
   structural), plus p50/p95 latency with and without rerank if the reranker
   is wired.
3. Compare against the stored baseline for the current milestone (see the
   Milestones table in architecture-docs/code-indexer-expanded-architecture.md
   §4) — state whether this run beats, matches, or regresses it.
4. If this run's numbers should become the new recorded baseline, log it:
   `python .claude/scripts/devlog.py decision add --title "M<n> eval baseline"
   --decision "<numbers>" --rationale "<why this is now the baseline>"`.

Numbers or it didn't happen — do not report an eval as passing without
running the harness this turn.
