# REST API reference

REST is the secondary interface — the dashboard and scripting surface over the same core engine as MCP (`src/noesis/api/routes.py`). Interactive OpenAPI docs are served at `/docs` on the running service.

Two middleware layers guard every request: `TrustedHostMiddleware` (accepts only `127.0.0.1` / `localhost` hosts — a DNS-rebinding guard) and, on every mutating route, `verify_local_origin` (`src/noesis/api/security.py`).

## Core routes

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/healthz` | liveness + model readiness | 200 |
| `POST` | `/projects` | register a folder and start indexing | 202 |
| `GET` | `/projects` | list registered projects | 200 |
| `GET` | `/projects/{id}/status` | latest run status (+ drift and coverage fields) | 200 / 404 |
| `POST` | `/projects/{id}/reindex` | incremental reindex (`?force=true` accepts an empty root as deletion, [ADR-55](../project/decisions.md)) | 202 / 404 / 409 / 429 |
| `GET` | `/runs/{run_id}` | run row, + live `progress` while running | 200 / 404 |
| `POST` | `/search` | hybrid / dense / sparse search | 200 / 404 / 429 |
| `POST` | `/structural-search` | AST-pattern search over live files | 200 / 400 / 404 |

### `GET /healthz`

```json
{"status": "ok", "assets": "ready", "embedder_ready": true,
 "reranker_assets": "disabled", "reranker_ready": "disabled"}
```

`status` is unconditional — it says the process is answering, nothing more. The other four are the cold-start signal ([ADR-77/78](../project/decisions.md), extended to the reranker by [ADR-87](../project/decisions.md)) — before them a caller saw green here and then paid a multi-minute silent model download inside its first search.

`assets` and `reranker_assets` are `"ready"` or `"missing"` — are that model's weights cached locally, asked of the HF cache without touching the network. `embedder_ready` and `reranker_ready` are `true` only once the model is actually loaded (each boundary records its resolved device *after* the constructor returns, so this cannot go true mid-load), `false` while it is not, and the string `"n/a"` for an implementation that reports no device (a test double). Cached weights with `ready: false` is the interesting state: the assets are there, but the next call still pays the in-memory load — seconds for the embedder, a ~2.3 GB construction for the reranker.

The reranker pair reads `"disabled"` when `reranker.enabled = false` (the shipped default). A value rather than absent keys, so a client can tell "reranking is off" from "this server doesn't report it", and no operator is sent to fetch 2.3 GB for a feature nobody turned on. All four read `"unknown"` when the app has no runtime context wired (a bare app without the lifespan), and the reranker pair alone reads `"unknown"` when the wired context does not model a reranker at all ([ADR-89](../project/decisions.md)) — a healthcheck reports what it cannot tell rather than guessing or raising.

### `POST /projects`

```bash
curl -X POST http://127.0.0.1:8000/projects \
  -H 'content-type: application/json' \
  -d '{"root_path": "/absolute/path/to/repo"}'
```

```json
{"project_id": "…", "run_id": "…", "status": "accepted"}
```

Errors are typed: a missing or non-directory `root_path` → **400**; the mixed-model guard (existing index built with a different embedding model) → **409 Conflict** with a "re-index required" detail. If a run is already in flight, the response carries `"status": "already_running"` with the live run's id.

At the machine-wide index-run cap ([ADR-85](../project/decisions.md)) this route answers **202** with `"status": "capacity_reached"` and a `project_id`, *not* 429 — registration commits before the run is launched, so a bare 429 would report that nothing happened while leaving a project whose id the caller never learned. `POST /projects/{id}/reindex` has no such side effect and does answer **429**.

### `GET /runs/{run_id}`

Returns the `index_runs` row (status, files_total/changed/failed, chunks_written, timestamps, trigger, error). While `status` is `"running"` it adds a REST-only live block:

```json
{"progress": {"files_done": 40, "files_to_index": 115, "chunks_written": 120,
              "percent": 34.8, "elapsed_s": 41.0, "eta_s": 77.0}}
```

### `POST /search`

Body: `query` (non-blank), `project_id`, `top_k` (1–100, default 10), `language?`, `channel` (`hybrid|dense|sparse`, default hybrid), `rerank?` (null → server default per [ADR-34](../project/decisions.md)).

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "where is RRF fusion applied", "project_id": "…", "top_k": 5}'
```

Response shape is identical to the MCP tool — see [`search_code`](mcp-tools.md#search_code).

### `POST /structural-search`

Body: `pattern`, `language`, `project_id`, `paths?`, `max_results?` (≥1, may only lower the config cap). Response shape identical to the MCP tool — see [`structural_search`](mcp-tools.md#structural_search). Typed errors: `unknown_project` → **404**, everything else (`unsupported_language`, `pattern_error`, `invalid_path`) → **400** with `{"type": …, "message": …}`.

## Dashboard endpoints

Server-rendered pages (excluded from the OpenAPI schema) and the JSON they poll (`src/noesis/api/dashboard.py`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Overview page |
| `GET` | `/projects/{id}/view` | Project detail page |
| `GET` | `/usage?days=30` | Usage analytics page (days clamped 1–365) |
| `GET` | `/api/state` | overview JSON (polled for live progress/badges) |
| `GET` | `/api/projects/{id}/state` | project-detail JSON |
| `GET` | `/api/usage?days=30` | usage JSON |
| `GET` | `/api/prefetch` | model-prefetch job status (polled; folded into `/api/state`'s `prefetch` field too) |

### Actions (all require local origin)

| Method | Path | Purpose | Status |
|---|---|---|---|
| `POST` | `/api/projects/{id}/flags` | toggle `watch_enabled` / `auto_reindex` | 200 / 404 |
| `POST` | `/api/projects/{id}/reindex-pending` | index only the watcher's pending changes | 202 / 400 / 404 / 409 / 429 |
| `POST` | `/api/settings/device` | set compute device (`auto`/`cuda`/`mps`/`cpu`), hot-reloads models | 200 / 400 |
| `DELETE` | `/api/projects/{id}` | delete a project's index entirely (chunks, runs, pending) — source files untouched | 200 / 404 |
| `POST` | `/api/prefetch` | start the "Download models" background job ([ADR row 95](../project/decisions.md)) | 202 |

### `GET` / `POST` `/api/prefetch` ([ADR row 95](../project/decisions.md))

The dashboard's "Download models" button, backed by the same functions `uv run python -m noesis.prefetch` runs from a terminal (grammars, BM25 tokenizer assets, the configured embedding model, the configured reranker model) — no separate download path, just a UI trigger over the existing one. One job at a time, process-wide (not per-project): a second `POST` while a job is running answers `{"status": "already_running"}` instead of launching a concurrent download.

```json
{"status": "running", "started_at": "2026-09-11T12:00:00+00:00", "finished_at": null,
 "error": null, "percent": null,
 "steps": {
   "grammars": {"status": "done", "detail": null},
   "bm25": {"status": "done", "detail": null},
   "model": {"status": "running", "detail": null},
   "reranker": {"status": "pending", "detail": null}
 }}
```

`status` is `idle` (never triggered), `running`, `done`, or `failed`. `percent` is `null` while running — the underlying functions report no sub-step progress, so this is an honest "step N of M is in flight" rather than a fabricated fraction (same "no smoothing pretence" as `GET /runs/{run_id}`'s `progress.eta_s`) — and becomes the percentage of applicable steps that finished `done` once the job stops running (100 on success; partial if it failed part-way; a `skipped` step, e.g. the model/reranker pair when `config.toml` cannot be read, counts toward neither the numerator nor the denominator). A hard failure stops the remaining steps rather than attempting them (a broken network fails every later step the same way).

### Registration flow ([ADR-42](../project/decisions.md))

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/api/languages` | supported language list for the register modal | 200 |
| `GET` | `/api/browse?path=…` | server-side folder browser (directories only) | 200 / 400 |
| `POST` | `/api/register/preview` | pre-flight scan: per-language file counts for a candidate root + scope | 200 / 400 |
| `POST` | `/api/register` | register with scope (`index_languages`, `max_file_bytes`, `follow_symlinks`, `extra_ignores`), `watch`, `auto_reindex`, `index_now` | 201 / 400 / 409 |

## Error-code summary

| Code | Meaning |
|---|---|
| 400 | invalid input: bad path, unsupported language, bad pattern, bad device |
| 404 | unknown `project_id` / `run_id` / `chunk_id` |
| 409 | mixed-model conflict — index was built with a different embedding model; full re-index required |
| 422 | request-body validation failure (FastAPI/pydantic), e.g. blank query, `top_k` out of range |
| 429 | at capacity, with `Retry-After`: the search gate is saturated ([ADR-84](../project/decisions.md)) or the machine-wide index-run cap is reached ([ADR-85](../project/decisions.md)). Nothing failed and nothing was queued — retry. **Not 503**, deliberately: 503 is also what a dead or unreachable server returns, and a client that cannot tell "busy" from "down" retries the wrong way. Connection-refused and a failing `/healthz` already mean down |
