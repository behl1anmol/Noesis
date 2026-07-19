# MCP tools reference

Noesis serves six tools over both MCP transports (streamable HTTP at `/mcp/`, stdio via `python -m noesis.mcp`). Every tool body is the same core call as its REST twin, and success payloads are identical dicts — tests assert byte-equality so the two surfaces cannot drift (`src/noesis/mcp/server.py`). Failures raise `ToolError` with the same detail REST puts in its HTTP error body.

## `search_code`

Semantic + lexical (hybrid) search over an indexed project.

| Param | Type | Default | Constraints |
|---|---|---|---|
| `query` | str | required | min length 1; blank/whitespace-only rejected (`ToolError: query must not be blank`) |
| `project_id` | str | required | must exist (`ToolError: unknown project_id`) |
| `top_k` | int | `10` | 1–100 |
| `language` | str \| null | `null` | canonical language filter (payload-indexed) |
| `channel` | `"hybrid" \| "dense" \| "sparse"` | `"hybrid"` | — |
| `rerank` | bool \| null | `null` | `null` → server default (`reranker.enabled`); ignored unless the reranker is enabled |

Returns:

```json
{
  "query": "validate jwt expiry",
  "channel": "hybrid",
  "reranked": false,
  "hits": [
    {
      "chunk_id": "6f0c…",
      "file_path": "src/auth/jwt.py",
      "start_line": 42,
      "end_line": 78,
      "language": "python",
      "symbol_name": "validate_token",
      "score": 0.5,
      "snippet": "def validate_token(…"
    }
  ]
}
```

When reranking was applied, `reranked` is `true` and each hit carries a `rerank_score`. Hits are candidates, not ground truth — read the live file before acting on a span. Telemetry records metadata only (interface, channel, latency, result count) — never the query text.

## `structural_search`

AST-pattern search (ast-grep) over the project's **live files** — never the index, so results are current even when the index is stale.

| Param | Type | Default | Constraints |
|---|---|---|---|
| `pattern` | str | required | ast-grep pattern, e.g. `def $NAME($$$ARGS): $$$BODY` |
| `language` | str | required | patterns are per-language; unsupported → error |
| `project_id` | str | required | — |
| `paths` | list[str] \| null | `null` | relative path-prefix restriction |
| `max_results` | int \| null | `null` | ≥ 1; `null` → config `structural.max_results` (may lower the cap, never raise it) |

Returns:

```json
{
  "pattern": "models.FusionQuery($$$A)",
  "language": "python",
  "matches": [
    {
      "file_path": "src/noesis/core/vectorstore.py",
      "start_line": 449,
      "end_line": 452,
      "matched_text": "models.FusionQuery(fusion=models.Fusion.RRF)",
      "meta_vars": {"A": "fusion=models.Fusion.RRF"}
    }
  ],
  "scanned_files": 57,
  "truncated": false,
  "timed_out": false
}
```

`truncated: true` means the scan stopped at `max_results`; `timed_out: true` means the wall-clock budget expired and matches are partial. Errors are typed — `unknown_project`, `unsupported_language`, `pattern_error` (with ast-grep's diagnostic, so agents can iterate on patterns cheaply), `invalid_path` — surfaced as `ToolError("type: message")`.

## `list_projects`

No parameters. Returns the registered projects (root path, embedding model, timestamps, per-project flags). Project ids from here feed every other tool.

## `get_index_status`

| Param | Type |
|---|---|
| `project_id` | str (required) |

Status of the most recent index run, shaped identically for REST and MCP:

```json
{
  "project_id": "…",
  "run_id": "…",
  "status": "done",
  "files_total": 115,
  "files_changed": 14,
  "chunks_written": 81,
  "started_at": "…",
  "finished_at": "…",
  "error": null,
  "expected_chunks": 333,
  "vector_count": 333,
  "drift": false
}
```

`status` is one of `never_indexed | queued | running | done | failed` (a registered project with no runs reports `never_indexed` with run fields nulled — a stable shape for agent consumers). The drift fields compare what the state DB expects against what Qdrant actually holds; a mismatch (`drift: true`) means the vector store lost data externally, and the next index run self-heals it ([ADR-49](../project/decisions.md)).

## `get_chunk`

| Param | Type |
|---|---|
| `chunk_id` | str (required, from `search_code` hits) |

Returns the exact stored span with the full chunk content — the indexed snapshot, which may lag the live file. Unknown id → `ToolError: unknown chunk_id`. Exposing `chunk_id` in every search hit is what makes this tool discoverable ([ADR-36](../project/decisions.md)).

## `reindex`

| Param | Type |
|---|---|
| `project_id` | str (required) |

Starts an incremental re-index (only changed files are re-embedded) and returns immediately with a `run_id` — poll `get_index_status` until `done`. If a run is already in flight the existing run's id is returned (`status: "already_running"`). The mixed-model guard (index built with a different embedding model) raises a `ToolError` explaining that a full re-index is required.

!!! note "Registration is not an MCP tool"
    Registering a project is an operator step over REST or the dashboard — agents get `reindex`, not register. See [Connecting agents](../getting-started/connecting-agents.md).
