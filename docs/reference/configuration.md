# Configuration reference

Noesis runs with zero config — every setting in `src/noesis/core/config.py` has a working default. A `config.toml` overrides per key; unknown keys are ignored, invalid values (wrong type, non-positive numbers) fail startup with a clear error.

## File resolution order

1. Explicit path passed by the embedding process (internal).
2. `$NOESIS_CONFIG` — environment variable, for hosts that can't control their cwd (typical for MCP host entries). `~` is expanded.
3. `./config.toml` — deliberate dev override when running from a checkout.
4. `$XDG_CONFIG_HOME/noesis/config.toml` (default `~/.config/noesis/config.toml`).

If no file is found, all defaults apply.

## Top level

| Key | Type | Default | Effect |
|---|---|---|---|
| `db_path` | path | `~/.local/share/noesis/noesis.sqlite` (respects `$XDG_DATA_HOME`) | SQLite state DB location. Anchored, never cwd-relative, so the HTTP server and a stdio MCP server always share one DB regardless of where each was launched ([ADR-44](../project/decisions.md)). A relative path resolves against the config file's own directory, not the process cwd. |

## `[embedder]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `model` | str | `nomic-ai/CodeRankEmbed` | Dense embedding model id. Changing it triggers the full-re-embed rule — the system refuses to serve mixed-model results. |
| `dim` | int > 0 | `768` | Vector dimension; the Qdrant collection's dense size is read from here at creation time. |
| `batch_size` | int > 0 | `32` | Documents per embed batch during indexing. |
| `device` | str | unset | Unset → auto-detect (`cuda` → `mps` → `cpu`). A pin here wins over the dashboard device setting. |

## `[qdrant]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `url` | str | `http://127.0.0.1:6333` | Qdrant server URL (localhost only by design). |
| `collection` | str | `noesis_chunks` | Collection name — one shared collection, filtered per project. |

## `[reranker]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `model` | str | `BAAI/bge-reranker-v2-m3` | Cross-encoder model id. |
| `enabled` | bool | `false` | Kill switch **and** per-request default: `false` never loads the model and requests cannot opt in; `true` makes `rerank` default on with per-request opt-out ([ADR-34](../project/decisions.md)). Default-off is the measured M4 gate decision ([ADR-35](../project/decisions.md)). |
| `preload` | bool | `false` | `true` loads the ~568M model at startup instead of on first reranked request. |
| `candidates` | int ≥ 0 | `50` | Fused candidates passed to the reranker per request. |
| `batch_size` | int > 0 | `16` | Pairs scored per cross-encoder batch. |
| `device` | str | unset | Same semantics as `embedder.device`. |

## `[structural]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `max_results` | int > 0 | `100` | Cap on matches per structural query. A request may lower it, never raise it. |
| `timeout_s` | float > 0 | `10.0` | Wall-clock scan budget. On expiry the scan stops and returns partial results with `timed_out: true` — partial matches are still actionable to an iterating agent. |

## `[git]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `fast_path` | bool | `true` | `false` disables git candidate narrowing entirely — every run does a full hash-walk (the correctness baseline the fast path must match). |

## `[watcher]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `poll_interval_s` | float > 0 | `1.0` | Snapshot cadence of the polling observer, used only for watched roots on inotify-blind filesystems (9p/cifs/nfs/fuse — e.g. WSL2's `/mnt/c`). Natively watched roots never poll. |

## Environment variables

| Variable | Effect |
|---|---|
| `NOESIS_CONFIG` | Explicit config-file path (resolution step 2). |
| `FASTEMBED_CACHE_PATH` | Where fastembed caches the BM25 assets. Set automatically by prefetch and the service to `$XDG_CACHE_HOME/noesis/fastembed` so runtime stays offline — override only if you know why. |

## Full example

```toml
db_path = "~/.local/share/noesis/noesis.sqlite"

[embedder]
model = "nomic-ai/CodeRankEmbed"
dim = 768
batch_size = 32
# device = "cuda"

[qdrant]
url = "http://127.0.0.1:6333"
collection = "noesis_chunks"

[reranker]
model = "BAAI/bge-reranker-v2-m3"
enabled = false
preload = false
candidates = 50
batch_size = 16

[structural]
max_results = 100
timeout_s = 10.0

[git]
fast_path = true

[watcher]
poll_interval_s = 1.0
```
