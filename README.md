<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="assets/noesis-banner-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="assets/noesis-banner-dark.svg">
    <img src="assets/noesis-banner.png" alt="Noesis" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://behl1anmol.github.io/Noesis/"><strong>📚 Full documentation</strong></a>
  — every detail of the models, indexing, MCP tools, dashboard, and design decisions.
</p>

## Beyond search. Toward understanding

Noesis is an AI-native code-understanding engine. It gives AI agents deep, current
knowledge of your codebase through **hybrid retrieval** (dense embeddings + lexical
BM25 + optional reranking), **structural AST search**, and **local-first incremental
indexing** — with a human dashboard and a file watcher that keeps the index fresh.

**MCP is the primary interface** (agents consume retrieval over MCP), REST is the
secondary interface (dashboard + scripting). Everything runs on `127.0.0.1` only:
after the one-time asset download, no code, query, or metadata ever leaves the
machine ([ADR-25](architecture-docs/code-indexer-expanded-architecture.md)).

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the service](#running-the-service)
- [The dashboard](#the-dashboard)
- [The file watcher](#the-file-watcher)
- [The indexer](#the-indexer)
- [The models](#the-models)
- [Configuration](#configuration)
- [Connecting an agent (MCP)](#connecting-an-agent-mcp)
- [REST API reference](#rest-api-reference)
- [GPU / device control](#gpu--device-control)
- [Development](#development)
- [Architecture docs](#architecture-docs)

---

## How it works

```
                        ┌─────────────────────────────────────────────┐
   agents ── MCP ──────▶│                                             │
   (stdio / HTTP)       │   FastAPI app  (127.0.0.1 only)             │
                        │   ┌──────────┐  ┌──────────┐  ┌───────────┐ │
   humans ── HTTP ─────▶│   │ MCP tools│  │ REST API │  │ Dashboard │ │
   (browser)            │   └────┬─────┘  └────┬─────┘  └─────┬─────┘ │
                        │        └── thin adapters over ──┐    │       │
                        │                                 ▼    ▼       │
                        │   ┌───────────────── core/ ─────────────┐   │
                        │   │ indexer · retriever · structural ·   │   │
                        │   │ embedder · reranker · watcher · jobs │   │
                        │   └──────┬───────────────────┬───────────┘   │
                        └──────────┼───────────────────┼───────────────┘
                                   ▼                   ▼
                           SQLite (state)        Qdrant (vectors)
                           projects, files,      dense + sparse
                           runs, pending,        chunk vectors
                           query_log
```

1. You **register a project** (a repo or folder). The indexer discovers indexable
   files, chunks them along AST boundaries, embeds each chunk, and writes vectors to
   Qdrant plus per-file state to SQLite.
2. Agents **search** — a natural-language query fuses dense semantic search with BM25
   lexical search (and optional reranking), or an AST **structural** pattern matches
   the live filesystem.
3. The **file watcher** notices edits and (if you enable auto-reindex) re-embeds only
   the changed files within seconds.
4. The **dashboard** shows index health, freshness, pending changes, and usage
   analytics so a human can see when the index is behind reality.

---

## Requirements

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** for dependency + environment management
- **Docker** (for Qdrant — the only container this project runs)
- **~4–5 GB disk** for model weights and grammars on first run
- **Optional: an NVIDIA GPU (CUDA) or Apple Silicon (MPS)** — indexing and reranking
  are much faster on a GPU, but CPU works

---

## Setup

```bash
# 1. Start Qdrant (vector store) — bound to localhost only
docker compose up -d

# 2. Install dependencies into a local venv
uv sync --all-groups

# 3. One-time asset download: tree-sitter grammars + BM25 + embedding model
uv run python -m noesis.prefetch
```

`prefetch` is what makes runtime fully offline. It downloads:

| Asset | Size (approx) | Flag to skip |
|---|---|---|
| Tree-sitter grammars (all supported languages) | small | — |
| BM25 sparse model (`Qdrant/bm25`) | ~100 KB | — |
| Embedding model (`nomic-ai/CodeRankEmbed`) | ~2 GB | `--skip-model` |
| Reranker (`BAAI/bge-reranker-v2-m3`) | ~2.3 GB | `--skip-reranker` |

The reranker ships **disabled by default** (see [The models](#the-models)); if you
never enable it you can skip its weights:

```bash
uv run python -m noesis.prefetch --skip-reranker
```

After prefetch the service makes **zero outbound network calls at runtime**.

---

## Running the service

```bash
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
```

> ⚠️ **Always bind `127.0.0.1`.** Never `0.0.0.0`. The whole security model is
> local-only; a wildcard bind exposes your source code and index to the network.

Once running:

- **Dashboard** → open <http://127.0.0.1:8000/> in a browser
- **MCP endpoint** → `http://127.0.0.1:8000/mcp/` (note the trailing slash)
- **REST + OpenAPI docs** → <http://127.0.0.1:8000/docs>
- **Health** → `GET /healthz`

---

## The dashboard

The dashboard is the **human monitoring surface** — server-rendered, no build
tooling, no CDN assets (it renders with the network cable pulled). Three pages:

### Overview (`/`)

- **Add project** — a modal to register a repo without leaving the browser: pick the
  folder (type the path or use the built-in folder browser), optionally scope the index
  (choose languages, max file size, follow-symlinks, extra ignore globs), see a
  **pre-flight preview** of how many files each language contributes before committing,
  then **Add only** (register without indexing) or **Add + index now**. Watch /
  auto-reindex can be enabled at registration.
- **Delete project** — each card has a Delete action (with a confirm dialog) that
  removes the project's index entirely: chunks, run history, pending changes.
  Source files are never touched; re-add and re-index any time.
- **Totals row** — projects, files indexed, chunks, pending changes, runs in flight.
- **Compute-device panel** — the active device and pills to switch between
  `auto` / `cuda` / `mps` / `cpu` (see [GPU / device control](#gpu--device-control)).
  A "GPU available" badge appears when CUDA or MPS is present.
- **Project cards**, one per indexed folder, each showing:
  - file / chunk counts and freshness ("indexed 3m ago" / "never indexed")
  - a **pending-changes badge** (amber when the watcher has seen edits)
  - the latest **run status** chip — green *done*, red *failed*, animated blue *running*
  - a **live progress bar** with percent complete and estimated time remaining while a
    run is in flight
  - **Watch** and **Auto-reindex** toggles (per project — see below)
  - **Reindex** (full incremental pass) and **Index pending** (only the watched
    changes) buttons

### Project detail (`/projects/{id}/view`)

Drill into one project: the **pending files** awaiting reindex (with event type and
detection time), the **failed files** of the most recent run (with the error), and a
**recent-runs** table (status, what triggered it, files changed / failed, chunk
counts, duration).

### Usage (`/usage`)

Graph-based analytics over the last 30 days, rendered as hand-drawn inline SVG charts:

- **Index activity** — runs per day (watcher- vs manual-triggered, failures in red),
  total runs, fast-path hit rate, average run duration
- **Search usage** — queries per day (MCP vs REST), latency p50 / p95, and the mix of
  search channels (hybrid / dense / sparse / structural)
- **Watcher activity** — filesystem events seen vs coalesced per day, auto-reindex
  triggers
- **Index health** — a per-project table of files, chunks, pending backlog, freshness
  age, and failed-file counts

> **Privacy note:** search usage is **metadata only**. Noesis records *that* a query
> ran and how it performed (interface, channel, latency, result count) — it never
> stores the query text.

The pages poll a small JSON API (`/api/state`, `/api/projects/{id}/state`,
`/api/usage`) to update progress bars and badges live without a full reload.

---

## The file watcher

The watcher keeps the index fresh automatically. It is built to be **lightweight** —
its filesystem-event thread does string checks only (no hashing, no file reads, no
database access), so watching never contends with your editor writing a file.

**Two per-project flags, both OFF by default:**

| Flag | Off (default) | On |
|---|---|---|
| **Watch** | project is not observed | filesystem events recorded as *pending changes*, visible on the dashboard |
| **Auto-reindex** | pending changes wait for a manual "Reindex" / "Index pending" | after a brief quiet period, changed files are re-embedded automatically within seconds |

The default-off design is deliberate: unsolicited background embedding would burn
GPU/CPU without your consent. With Watch on but Auto-reindex off, you still *see*
staleness — the index just doesn't rebuild until you ask. Turning **Auto-reindex on
also catches up** any changes that accumulated while it was off.

**What the watcher ignores** (so it doesn't create noise): excluded directories
(`.git`, `node_modules`, `.venv`, …), secret files (`.env`, `*.pem`, keys),
generated lockfiles, editor scratch files (`*.swp`, `#…#`, vim's write-probe), and
anything your root `.gitignore` excludes.

**Correctness:** a watcher-triggered run is *scoped* to exactly the changed files, but
it still re-runs full discovery and SHA-256 hashing on them — the hash remains the
source of truth. Scoped runs deliberately **never advance the git fast-path anchor**,
so the next full pass can never skip a file the watcher didn't see.

---

## The indexer

The indexing pipeline (`core/indexer.py`) is: **discover → hash-diff → chunk → embed →
upsert**.

- **Discovery** walks the tree and filters via `.gitignore` (with nested-file and
  negation semantics), a secret skip-list, a generated-lockfile skip-list, a size cap,
  and a binary sniff.
- **Hash-diff** SHA-256s each file and partitions into *new / changed / unchanged /
  deleted* against stored state. Only new and changed files are re-embedded; deleted
  files have their chunks pruned. This is what makes re-indexing cheap.
- **Git fast-path** (when the project is a git repo) narrows the candidate set to
  what changed since the last indexed commit, then falls back safely to a full
  hash-walk on any ambiguity (nested repos, submodules, history rewrites, detached
  HEAD, mid-merge — every fallback is tested).
- **Chunking** splits along AST boundaries (via tree-sitter) into a 300–800 token
  budget so a chunk is a coherent unit (a function, a class), then concatenation
  reproduces the file exactly.
- **Per-file error containment:** if one file fails to read/chunk/embed, it's recorded
  (visible on the dashboard) and skipped — the rest of the run still completes, and the
  failed file is retried next run. A run where *every* file fails is marked failed.

Indexing runs in a **background job** and reports progress live. Register a project or
kick a reindex and you get a `run_id` back immediately; poll `GET /runs/{run_id}` (or
watch the dashboard) for status, percent complete, and ETA.

### Concurrent search + indexing

`search_code` and an index run can genuinely overlap — the watcher launches index
runs on its own whenever a file changes, which is the normal working state of this
tool, not an edge case. So can two agents searching at the same moment, and that
turns out to be the harder case.

qdrant-client keeps its client-side BM25 inference bookkeeping on the `QdrantClient`
**object**, unsynchronized. Two threads that build a query on one client can receive
each other's embeddings — and the damage latches: the drain step recomputes only
when its FIFO is empty and otherwise pops from the front, so one race leaves that
client permanently offset and every later query silently returns another query's
results until the process restarts. Measured against a live server, one shared
client under 16 concurrent readers returned the wrong document for up to 15,864 of
16,000 queries, raising only 4–7 exceptions. No indexing has to be in flight for
that; readers alone are enough.

Connections are therefore owned by role, and the read path gets a *pool* rather than
a single connection (ADR-83). The production wiring is **K query connections +
1 index + 1 admin = K + 2**:

```
  search_code (agents)         index run (jobs.py)          everything else
              │                            │                            │
              ▼                            ▼                            ▼
 ┌──────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐
 │ SearchGate (ADR-84)      │ │ index_client             │ │ admin client             │
 │ K threads run, K+Q may   │ │ upserts only, so an      │ │ deletes, counts, scroll, │
 │ be admitted; the rest    │ │ index run never takes a  │ │ retrieve, setup — it     │
 │ get a 429 / ToolError    │ │ query connection         │ │ builds no Document, so   │
 └──────────────────────────┘ └──────────────────────────┘ │ it needs no slot         │
              │                            │               └──────────────────────────┘
              │ 1 thread ⇄                 │                            │
              │ 1 connection               │                            │
              ▼                            │                            │
 ┌──────────────────────────┐              │                            │
 │ query_clients: K         │              │                            │
 │ connections, one held    │              │                            │
 │ per sparse or hybrid     │              │                            │
 │ query (a dense-only      │              │                            │
 │ query takes none)        │              │                            │
 └──────────────────────────┘              │                            │
              │                            │                            │
              └────────────────────────────┴────────────────────────────┘
                                           │
                                           ▼
                           one Qdrant server, one collection
```

- **K is sized to the machine.** `clamp(available CPUs, 2, 8)`, read through
  `sched_getaffinity` so a cgroup-limited container sizes for its own allowance
  rather than for the host's cores. On a 4-CPU box (16 concurrent searches, 5,000
  points, hybrid, prefetch 50, against a live Qdrant 1.18.3) throughput saturates
  *exactly* at the CPU count — 384 q/s at K=4 — then stays flat out to K=12 while
  p95 climbs from 16.0 ms to 48.4 ms, because Qdrant is co-resident and competing
  for the same cores. The floor of 2 is measured (K=1 gives up about a third of
  throughput, 258 q/s). The ceiling of 8 is a **judgment call, not a measurement** —
  no larger box was available to test on — which is exactly why
  `[qdrant] query_connections` can override it.
- **Search runs on its own bounded executor**, sized 1:1 with the pool (ADR-84), so
  a thread is free exactly when a connection is. It admits `K + query_queue_depth`
  calls and refuses the rest: REST `/search` answers **429 with `Retry-After`**, MCP
  `search_code` raises a `ToolError` naming the live counts and the knob. 429 and
  not 503, because 503 is also what a dead server returns, and a client has to be
  able to tell "busy" from "down". A fast, legible refusal beats a request that sits
  for an unbounded time.
- **Index runs are capped machine-wide** by `[indexing] max_concurrent_index_runs`
  (default 4, ADR-85), enforced inside the same `BEGIN IMMEDIATE` transaction that
  claims a run — so the cap holds *across processes*, which an in-process semaphore
  could not: the documented deployment runs an HTTP server and a stdio MCP process
  against one state DB. At the cap, REST answers 429, MCP `reindex` raises a
  `ToolError` saying nothing was started, and the watcher defers and re-arms.
- **Dense-only search takes no slot at all.** It builds no `models.Document`, so
  there is no inference state on that path to protect and no reason to make it
  contend for one.

Underneath the pool there is still one lock per distinct client *object* — the
correctness floor for embedded callers, where a single in-memory client necessarily
fills all three roles at once. In the production wiring the objects are distinct, so
those locks are uncontended.

---

## The models

Noesis retrieves over three complementary channels, fused with Reciprocal Rank Fusion:

| Channel | Model | Role | Default |
|---|---|---|---|
| **Dense** | `nomic-ai/CodeRankEmbed` (768-dim) | semantic "what does this *mean*" search | always on |
| **Sparse** | `Qdrant/bm25` (native IDF) | lexical "find this exact token/symbol" | always on |
| **Rerank** | `BAAI/bge-reranker-v2-m3` cross-encoder | reorders the top candidates for precision | **off** (opt-in) |

**Why the reranker is off by default:** measured on a T4 GPU it added ~+0.106 NDCG@10
but cost ~12.2 s p50 / ~13.4 s p95 per reranked query — an unacceptable interactive latency
([ADR-35](architecture-docs/code-indexer-expanded-architecture.md), full numbers in
[`m4-reranker-benchmarks.md`](architecture-docs/m4-reranker-benchmarks.md)). It ships
as a per-request opt-in: set `reranker.enabled = true` in config to make it available,
or pass `"rerank": true` on a search request when enabled.

All model calls go through two boundaries — `core/embedder.py` (the `Embedder`
Protocol) and `core/reranker.py` (the `Reranker` Protocol). **Local implementations
only**: remote/hosted embedding was rejected, and nothing in `core/` may make outbound
HTTP (CI enforces both).

---

## Configuration

Noesis runs with **zero config** — every setting has a working default. To override,
create `~/.config/noesis/config.toml` (or a `config.toml` in the working directory
when running from a checkout, or point `NOESIS_CONFIG` at a file):

```toml
# State DB. Default: ~/.local/share/noesis/noesis.sqlite — anchored, never
# cwd-relative, so the HTTP server and the stdio MCP server always share one
# DB no matter where each was launched from. A relative path here resolves
# against this file's directory.
db_path = "~/.local/share/noesis/noesis.sqlite"

[embedder]
model = "nomic-ai/CodeRankEmbed"
dim = 768
batch_size = 32
# device: omit for auto-detect (cuda → mps → cpu); or pin "cuda" / "cpu".
# A pin here WINS over the dashboard device setting.
# device = "cuda"

[qdrant]
url = "http://127.0.0.1:6333"
collection = "noesis_chunks"
# Search concurrency. Omit both for auto: clamp(available CPUs, 2, 8)
# connections, and 4x that many searches allowed to wait for one.
# Past connections + queue_depth in flight, /search answers 429.
# query_connections = 4
# query_queue_depth = 16

[reranker]
model = "BAAI/bge-reranker-v2-m3"
enabled = false        # true → reranker loads and `rerank` defaults on
preload = false        # true → load the model at startup instead of first use
candidates = 50        # fused candidates reranked per request
batch_size = 16
# device = "cuda"

[structural]
max_results = 100      # cap on matches per structural query
timeout_s = 10.0       # wall-clock scan budget; partial results returned on expiry

[git]
fast_path = true       # false → every run does a full hash-walk

[indexing]
max_concurrent_index_runs = 4   # machine-wide cap on index runs, across
                                # processes; at the cap, REST answers 429
                                # and nothing is silently queued

[server]
port = 8000            # HTTP port, and where `python -m noesis.mcp --shared`
                       # looks for (or starts) the shared server. There is no
                       # host key: the bind is fixed at 127.0.0.1.
```

**Environment:** `FASTEMBED_CACHE_PATH` controls where BM25 assets are cached
(default `$XDG_CACHE_HOME/noesis/fastembed`); it's set automatically to keep runtime offline.

---

## Connecting an agent (MCP)

Noesis serves the same six tools over both transports:
`search_code`, `structural_search`, `list_projects`, `get_index_status`,
`get_chunk`, `reindex`.

**Option A — HTTP** (service already running on port 8000):

```bash
claude mcp add --transport http noesis http://127.0.0.1:8000/mcp/
```

**Option B — stdio** (the agent host spawns the server itself):

```bash
claude mcp add noesis -- uv run --project /absolute/path/to/noesis python -m noesis.mcp
```

The stdio server builds its own core resources from `config.toml` in the working
directory. Full walkthrough (including a Python client example) is in
[`architecture-docs/m6-agent-connection-guide.md`](architecture-docs/m6-agent-connection-guide.md).

**Option C — stdio shim onto a shared server** (what you want with more than one
agent):

```bash
claude mcp add noesis -- uv run --project /absolute/path/to/noesis python -m noesis.mcp --shared
```

Each agent gets a thin stdio-to-HTTP proxy that holds no model, no Qdrant pool and
no state handle; the first one to start elects a single shared server and the rest
attach to it. The reason is memory, and it is not the one people expect: a Noesis
process that has loaded the embedding model measures **1342 MB RSS, and 1140 MB PSS
with two of them running — `Shared_File` is 0 MB**, because the weights land in
private tensors, so agents do *not* share them. The on-disk model cache is already
shared (523 MB, one copy; a warm second process loads in 11.7 s and downloads
nothing), so re-downloading was never the problem. The shim measures 165 MB RSS /
133 MB PSS against a 1427 MB server, which makes ten agents **~3.1 GB instead of
~13.4 GB**. Default stays standalone, so existing setups are unaffected
(ADR-86).

**Typical agent loop:** `list_projects` → `search_code` (get ranked spans with
`chunk_id`s) → read the live file, or `get_chunk(chunk_id)` for the exact indexed
span → `structural_search` for precise AST matches. Search hits are *candidates* —
agents should read the live file before acting.

---

## REST API reference

Secondary interface, for the dashboard and scripting. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness check |
| `POST` | `/projects` | register a folder and start indexing → `202` + `run_id` |
| `GET` | `/projects` | list registered projects |
| `GET` | `/projects/{id}/status` | latest run status for a project |
| `POST` | `/projects/{id}/reindex` | incremental reindex → `202` + `run_id` |
| `GET` | `/runs/{run_id}` | run row (+ live `progress` while running) |
| `POST` | `/search` | hybrid / dense / sparse search |
| `POST` | `/structural-search` | AST-pattern search over live files |

`/search`, `/projects` and `/projects/{id}/reindex` also answer **429 with
`Retry-After`** when the machine is at search or index-run capacity — busy, not
down, and nothing was queued. See
[Concurrent search + indexing](#concurrent-search--indexing).

Register and search (or use the dashboard's **Add project** modal for the same thing
with a folder picker, per-language scope, and a pre-flight preview):

```bash
# Register + index a project
curl -X POST http://127.0.0.1:8000/projects \
  -H 'content-type: application/json' \
  -d '{"root_path": "/absolute/path/to/your/repo"}'
# → {"project_id": "…", "run_id": "…", "status": "accepted"}

# Search (channel: hybrid | dense | sparse; rerank optional)
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "validate jwt expiry", "project_id": "…", "top_k": 10}'

# Structural (AST) search
curl -X POST http://127.0.0.1:8000/structural-search \
  -H 'content-type: application/json' \
  -d '{"pattern": "def $NAME($$$ARGS): $$$BODY", "language": "python", "project_id": "…"}'
```

---

## GPU / device control

Model placement resolves in this order of precedence:

1. **A `device` pin in `config.toml`** (`[embedder]` / `[reranker]`) — the operator's
   explicit choice, never overridden.
2. **The dashboard device setting** (`auto` / `cuda` / `mps` / `cpu`), persisted across
   restarts.
3. **Auto-detect** — `cuda` → `mps` → `cpu`.

Changing the device from the dashboard **hot-reloads** the models on their worker
threads (the first job after the switch pays the reload). If a config pin is present,
the dashboard control is locked and shows the pin.

> Verifying the GPU toggle needs real hardware. A ready-to-run Colab notebook is in
> [`architecture-docs/m8-colab-gpu-verification.md`](architecture-docs/m8-colab-gpu-verification.md).

---

## Development

```bash
uv run pytest                          # full suite (offline: fakes, in-memory Qdrant)
uv run pytest -m integration           # opt-in: loads the real embedding model
uv run pytest tests/eval/ -m golden -s    # M3 evaluation harness (self-indexes this repo)
bash .claude/scripts/ci_greps.sh       # guardrail greps (local-only invariants)
```

The suite runs fully offline against a `FakeEmbedder` and an in-memory Qdrant — no
model download, no Docker required. The `integration`, `golden` and `server` marks exercise the
real model and are excluded from the default run.

**Guardrails enforced in CI:** no `sentence_transformers` import outside the two model
boundaries; no HTTP client anywhere in `core/`; `127.0.0.1`-only binds; `mcp` pinned
`<2`. See [`CLAUDE.md`](CLAUDE.md) for the full house rules.

**Brand assets** (icon/logo/banner, dark + light) are code-generated, not hand-drawn:

```bash
python3 assets/scripts/generate_assets.py   # regenerates assets/noesis-*-{dark,light}.svg
```

Edit `assets/scripts/generate_assets.py` and rerun to change the mark; then re-rasterize
the PNGs used by the dashboard (`src/noesis/api/static/favicon.png`,
`src/noesis/api/static/noesis-logo.png`) and the top-level `assets/noesis*.png` via
`cairosvg` (or any SVG rasterizer) from the `-dark` SVG variants.

---

## Architecture docs

The authoritative design lives in
[`architecture-docs/`](architecture-docs/):

- [`code-indexer-expanded-architecture.md`](architecture-docs/code-indexer-expanded-architecture.md)
  — the authoritative plan: design decisions (ADRs), risk register, milestone roadmap
- [`code-indexer-initial-idea.md`](architecture-docs/code-indexer-initial-idea.md)
  — the approved baseline and rationale
- [`m4-reranker-benchmarks.md`](architecture-docs/m4-reranker-benchmarks.md)
  — reranker latency/quality measurements behind the default-off decision
- [`m6-agent-connection-guide.md`](architecture-docs/m6-agent-connection-guide.md)
  — connecting agents over MCP (both transports)
- [`m8-colab-gpu-verification.md`](architecture-docs/m8-colab-gpu-verification.md)
  — GPU device-toggle verification on Colab

Every design decision carries a recorded rationale — no rationale, no merge.
