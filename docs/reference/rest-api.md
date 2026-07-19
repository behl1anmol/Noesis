# REST API reference

REST is the secondary interface — the dashboard and scripting surface over the same core engine as MCP (`src/noesis/api/routes.py`). Interactive OpenAPI docs are served at `/docs` on the running service.

Two middleware layers guard every request: `TrustedHostMiddleware` (accepts only `127.0.0.1` / `localhost` hosts — a DNS-rebinding guard) and, on every mutating route, `verify_local_origin` (`src/noesis/api/security.py`).

## Core routes

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/healthz` | liveness | 200 |
| `POST` | `/projects` | register a folder and start indexing | 202 |
| `GET` | `/projects` | list registered projects | 200 |
| `GET` | `/projects/{id}/status` | latest run status (+ drift fields) | 200 / 404 |
| `POST` | `/projects/{id}/reindex` | incremental reindex | 202 / 404 / 409 |
| `GET` | `/runs/{run_id}` | run row, + live `progress` while running | 200 / 404 |
| `POST` | `/search` | hybrid / dense / sparse search | 200 / 404 |
| `POST` | `/structural-search` | AST-pattern search over live files | 200 / 400 / 404 |

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

### Actions (all require local origin)

| Method | Path | Purpose | Status |
|---|---|---|---|
| `POST` | `/api/projects/{id}/flags` | toggle `watch_enabled` / `auto_reindex` | 200 / 404 |
| `POST` | `/api/projects/{id}/reindex-pending` | index only the watcher's pending changes | 202 / 400 / 404 / 409 |
| `POST` | `/api/settings/device` | set compute device (`auto`/`cuda`/`mps`/`cpu`), hot-reloads models | 200 / 400 |
| `DELETE` | `/api/projects/{id}` | delete a project's index entirely (chunks, runs, pending) — source files untouched | 200 / 404 |

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
