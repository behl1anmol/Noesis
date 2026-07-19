# Reranker

The second and last model-loading boundary: `src/noesis/core/reranker.py` defines the `Reranker` Protocol and the cross-encoder implementation behind the optional `rerank` search flag.

## Role

```python
@runtime_checkable
class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...
    async def rerank(self, query: str, texts: list[str]) -> list[float]: ...
```

`rerank` is a **pure scoring function**: one relevance score per candidate text, same order as the input. Reordering is the caller's job (the retriever sorts by score). That keeps implementations trivial to fake — `FakeReranker` scores by lexical overlap of query tokens (splitting on underscores so `validate_token` overlaps "validate token") and records calls for assertions.

| Implementation | Model | Batch size |
|---|---|---|
| `LocalCrossEncoderReranker` | `BAAI/bge-reranker-v2-m3` (~568 M params, ~2.3 GB) | 16 |
| `FakeReranker` | lexical overlap, deterministic | — |

## Design decisions

- **Dedicated worker thread, never the embedder's ([ADR-20](../project/decisions.md)).** A rerank of ~50 pairs takes materially longer than a query embed; sharing one worker would let a rerank head-of-line-block the next query's embedding. Two single-thread workers keep both models safe from concurrent forward passes and both latencies bounded. Every rerank job is interactive — there is no indexing-path job class on this worker — so a plain FIFO queue implements the HIGH/LOW discipline vacuously.
- **Lazy load + kill switch.** The worker starts on the first `rerank` call and the model loads inside the worker on its first job. `reranker.preload = true` calls `preload()` at startup instead (a no-op job forces the load). `reranker.enabled = false` removes the reranker entirely — the system is fully functional without it, and ~2.3 GB of weights is a real cost on developer laptops.
- **Default-off.** Reranking measurably improves quality but costs ~12 s per reranked query on a T4 — an unacceptable interactive latency ([ADR-35](../project/decisions.md)). Full numbers and the gate decision are in [Evaluation](evaluation.md).
- **Truncation is counted, not silent.** Pairs longer than the model's max sequence length are truncated by the model. Chunks target 300–800 tokens so this should be rare; `_count_truncated` tallies affected pairs per call and logs a warning when any occur.
- **`rerank_score` is response-only.** Nothing in the rerank path touches the Qdrant schema or SQLite state; the score rides along in the search response and disappears.
- **Same guardrails as the embedder.** Lazy `sentence_transformers` import (one of the two allowed sites, [ADR-33](../project/decisions.md)); explicit device resolution instead of ST auto-detect; device hot-swap via the same generation-bump mechanism ([ADR-40](../project/decisions.md)); log lines carry `model_id` and device only, never query or chunk text ([ADR-25](../project/decisions.md)).

## Key invariants

- Forward passes are never concurrent — the single worker thread owns the model.
- Worker exceptions propagate to the awaiting caller; the worker thread never dies.
- Scores return in input order; `float(s)` coercion guarantees plain Python floats.
- The reranker can be absent, disabled, or unloaded at any time without affecting indexing, hybrid search, or the data model.
