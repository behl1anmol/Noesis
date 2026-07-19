# GPU and device control

Both models (embedder and reranker) can run on CUDA, Apple Silicon (MPS), or CPU; Noesis resolves the device per model with a strict precedence and can hot-swap it from the dashboard.

## Resolution precedence

1. **A `device` pin in `config.toml`** (`[embedder]` / `[reranker]`) — the operator's explicit choice, never overridden. When a pin is present the dashboard control is locked and shows the pin.
2. **The dashboard device setting** (`auto` / `cuda` / `mps` / `cpu`), persisted in the state DB across restarts.
3. **Auto-detect** — `cuda` → `mps` → `cpu` (`resolve_device` in `src/noesis/core/compute.py`; the chosen device is logged).

## Hot reload from the dashboard

Changing the device on the dashboard's compute panel reloads the models on their worker threads via a generation bump — the running process never restarts. The first job after the switch pays the reload cost; subsequent jobs run on the new device. A "GPU available" badge appears when CUDA or MPS is detected.

```toml
[embedder]
# device = "cuda"   # pin — wins over the dashboard setting

[reranker]
# device = "cuda"
```

## What a GPU buys you

- **Indexing** — document embedding is the dominant cost of an index run; a GPU shortens it substantially.
- **Reranking** — the 568M-parameter cross-encoder is compute-bound. Even on a T4 GPU it measured **~12.2 s p50 per reranked query** over 50 candidates, which is why the reranker ships default-off as a per-request opt-in ([ADR-35](../project/decisions.md); full numbers in [Evaluation](../internals/evaluation.md)).
- **Search itself stays fast on CPU** — a query embed is one short text; hybrid search without rerank measured ~60 ms p50.

!!! note "Verifying the toggle moves real models"
    The test suite proves the device-setting plumbing on fakes (offline, CPU). Verifying that the toggle actually relocates the real models requires GPU hardware; a ready-to-run Colab T4 notebook with pass criteria (auto resolves to `cuda`; `set_compute_device("cpu")` makes the next run reload and report `cpu`; a scoped run embeds exactly one file) is in [`architecture-docs/m8-colab-gpu-verification.md`](https://github.com/behl1anmol/Noesis/blob/main/architecture-docs/m8-colab-gpu-verification.md).

See also: [Configuration reference](../reference/configuration.md) · [Embedder internals](../internals/embedder.md) · [Reranker internals](../internals/reranker.md)
