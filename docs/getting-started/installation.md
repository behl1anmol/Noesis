# Installation

Noesis runs entirely on your machine: one Python process, one Qdrant container, and locally cached model weights. After a one-time asset download, the service makes zero outbound network calls at runtime.

## Prerequisites

| Requirement | Why |
|---|---|
| Python ≥ 3.11 | Runtime floor (`requires-python` in `pyproject.toml`; 3.12 is the development target) |
| [uv](https://docs.astral.sh/uv/) | Dependency and environment management — the project is uv-native (`uv.lock` is the source of truth) |
| Docker | Runs Qdrant, the vector store — the only container Noesis uses |
| ~4–5 GB disk | Model weights and tree-sitter grammars on first run |
| Optional: NVIDIA GPU (CUDA) or Apple Silicon (MPS) | Indexing and reranking are much faster on a GPU; CPU works fine for retrieval |

## 1. Clone and install

```bash
git clone https://github.com/behl1anmol/Noesis.git
cd Noesis
uv sync --all-groups
```

`uv sync` creates a local virtual environment and installs the locked dependency set. `--all-groups` includes the dev and docs groups; plain `uv sync` installs runtime dependencies only.

## 2. Start Qdrant

```bash
docker compose up -d
```

This starts a single `qdrant/qdrant:v1.18.3` container bound to localhost only — REST on `127.0.0.1:6333`, gRPC on `127.0.0.1:6334` — with a named volume (`qdrant_storage`) for persistence. Qdrant ≥ 1.15.2 is required for native BM25 sparse vectors (see [Retrieval](../concepts/retrieval.md)), and the container runs with `QDRANT__TELEMETRY_DISABLED=true` so the datastore does not report usage statistics outbound ([ADR-63](../project/decisions.md)).

The server tag and the `qdrant-client` pin in `pyproject.toml` are a **matched pair** and move together ([ADR-62](../project/decisions.md)). The client only vouches for a server within one minor version of itself and warns on every connection otherwise, so `tests/test_qdrant_pin_compat.py` fails if the two pins drift apart.

### Upgrading Qdrant

Changing the image tag on an existing `qdrant_storage` volume is a **migration, not a restart**. Qdrant guarantees storage compatibility across [one minor version only](https://qdrant.tech/documentation/upgrades/), and a Noesis collection written by an older server can stop a newer one from starting *at all* — the failure is a startup panic on the whole process, not a single unreadable collection:

```
Failed to load local shard "./storage/collections/noesis_chunks/0":
unknown variant `rocks_db`, expected `gridstore` or `mmap`
```

That specific one bites anyone moving across 1.17, which removed the RocksDB-backed payload index. Noesis creates payload indexes on `project_id` and `language` in `ensure_collection`, so every Noesis collection carries the affected structure.

The supported paths are a stepwise upgrade (one minor at a time, latest patch of each) or a wipe. **Wiping is the recommended path here**, because the Qdrant collection is a *derived* artifact: SQLite plus your working tree is the source of truth, and Noesis detects a missing collection against a non-empty file ledger at startup, warns, and self-heals by re-embedding.

```bash
docker compose down -v     # -v drops the volume; without it the panic persists
docker compose up -d
```

Then re-index your projects. Nothing but embeddings is lost, and they are reproducible from the files.

## 3. Prefetch assets

```bash
uv run python -m noesis.prefetch
```

`prefetch` is what makes runtime fully offline. It downloads:

| Asset | Approx. size | Skip flag |
|---|---|---|
| Tree-sitter grammars (all 23 mapped languages) | small | — |
| BM25 sparse-model assets (`Qdrant/bm25`) | ~100 KB | — |
| Embedding model (`nomic-ai/CodeRankEmbed`) | ~2 GB | `--skip-model` |
| Reranker (`BAAI/bge-reranker-v2-m3`) | ~2.3 GB | `--skip-reranker` |

The reranker ships **disabled by default** (see [GPU and devices](gpu.md) and the [evaluation results](../internals/evaluation.md) for why). If you never enable it, skip its weights:

```bash
uv run python -m noesis.prefetch --skip-reranker
```

Alternative models can be prefetched with `--model` / `--reranker-model` (they must then match your [configuration](../reference/configuration.md)).

!!! note "Where assets land"
    Model weights go to the Hugging Face cache. BM25 assets are cached under `$XDG_CACHE_HOME/noesis/fastembed` (the `FASTEMBED_CACHE_PATH` environment variable, set automatically by both `prefetch` and the service so runtime never re-downloads). A grammar that fails to download is non-fatal — files in that language fall back to line-based chunking.

## Verify

```bash
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/healthz
# {"status":"ok"}
```

Continue to the [Quickstart](quickstart.md) to index your first project.
