# Noesis MCP tool reference

Ground truth for the six tools. Shapes match the Noesis MCP server exactly (the
MCP and REST surfaces return byte-identical success payloads). All tools are
async; call them as normal MCP tools under the `noesis:` prefix.

---

## search_code

Semantic + lexical (hybrid) search over an **indexed** project.

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | — | Natural language or code. Describe *intent*; hybrid handles keywords too. |
| `project_id` | string | — | From `list_projects` (`id`). Unknown → `unknown project_id`. |
| `top_k` | int | `10` | `1`–`100`. |
| `language` | string \| null | `null` | Filter by language, e.g. `"python"`. |
| `channel` | `"hybrid"` \| `"dense"` \| `"sparse"` | `"hybrid"` | `dense` = embeddings only, `sparse` = BM25 only. Prefer `hybrid`. |
| `rerank` | bool \| null | `null` | `null` → server default (on iff the reranker is enabled). `true` forces rerank; `false` disables. |

**Returns**
```json
{
  "query": "…",
  "channel": "hybrid",
  "reranked": true,
  "hits": [
    {
      "chunk_id": "…",
      "file_path": "src/pkg/mod.py",
      "start_line": 42,
      "end_line": 88,
      "language": "python",
      "symbol_name": "fuse_channels",
      "score": 0.83,
      "snippet": "def fuse_channels(...):\n    ..."
    }
  ]
}
```
`hits` is ranked best-first. `chunk_id` feeds `get_chunk`. `file_path` is
project-relative — read the **live** file at `start_line`–`end_line` before acting.

---

## structural_search

AST-pattern search (ast-grep) over the project's **live files** — results are
current even if the index is stale.

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `pattern` | string | — | An ast-grep pattern. Metavars: `$NAME` (one node), `$$$ARGS` (zero+ nodes). |
| `language` | string | — | Required. e.g. `"python"`, `"typescript"`. Unknown → `unsupported_language` with the supported list. |
| `project_id` | string | — | Unknown → `unknown_project`. |
| `paths` | list[str] \| null | `null` | Restrict to these relative path prefixes. |
| `max_results` | int \| null | `null` | `null` → server cap. Requests may lower the cap, not raise it. |

**Pattern examples**
- Python def: `def $NAME($$$ARGS): $$$BODY`
- Python call site: `models.FusionQuery($$$A)`
- TS log: `console.log($$$A)`

**Returns**
```json
{
  "pattern": "…",
  "language": "python",
  "matches": [
    {
      "file_path": "src/pkg/mod.py",
      "start_line": 51,
      "end_line": 51,
      "matched_text": "models.FusionQuery(prefetch=..., query=...)",
      "meta_vars": {"A": ["prefetch=...", "query=..."]}
    }
  ],
  "scanned_files": 214,
  "truncated": false,
  "timed_out": false
}
```
`truncated: true` → the `max_results` cap was hit; more matches may exist.
`timed_out: true` → the scan hit its wall-clock budget; `matches` are partial.
Treat either flag as "this list may be incomplete."

---

## list_projects

List registered projects. No arguments.

**Returns** a list of rows:
```json
[
  {
    "id": "…",
    "root_path": "/abs/path/to/repo",
    "embedding_model": "nomic-ai/CodeRankEmbed",
    "created_at": "2026-07-06T…",
    "updated_at": "2026-07-06T…"
  }
]
```
Use `id` as the `project_id` everywhere else. Empty list → nothing is registered
yet (see registration in [transports.md](transports.md)).

---

## get_index_status

Status of the most recent index run for a project.

| Param | Type | Notes |
|-------|------|-------|
| `project_id` | string | Unknown → `unknown project_id`. |

**Returns** (stable shape; fields null when `never_indexed`)
```json
{
  "project_id": "…",
  "run_id": "…",
  "status": "done",
  "files_total": 214,
  "files_changed": 3,
  "chunks_written": 41,
  "started_at": "…",
  "finished_at": "…",
  "error": null
}
```
`status` ∈ `never_indexed | running | done | failed`. On `failed`, read `error`.
Poll after `reindex` until `done`.

---

## get_chunk

Fetch one indexed chunk by id (ids come from `search_code` hits).

| Param | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | Unknown → `unknown chunk_id`. |

**Returns** the exact stored span with full chunk content (file path, line range,
language, symbol, and the complete indexed text). This is the **indexed
snapshot** — it may lag the live file. Use it to compare stored vs. live, or to
get a hit's full text when the search `snippet` was truncated.

---

## reindex

Re-index a registered project. Incremental — only changed files are re-embedded.

| Param | Type | Notes |
|-------|------|-------|
| `project_id` | string | Unknown → `unknown project_id`. |

**Returns** immediately (async):
```json
{ "project_id": "…", "run_id": "…", "status": "accepted" }
```
Then poll `get_index_status(project_id)` until `done`. If a run is already
`running` for the project, this returns that run's id rather than starting a
second one. A mixed embedding-model conflict raises a `ToolError`.

---

## Error strings you may see

| Error | Meaning | Fix |
|-------|---------|-----|
| `unknown project_id` | The id isn't registered. | `list_projects`; register if absent. |
| `unknown chunk_id` | Stale/typo'd chunk id. | Re-run `search_code` for fresh ids. |
| `unknown_project: …` | Same, from `structural_search`. | As above. |
| `unsupported_language: …` | Language not supported for AST search. | Use one from the listed set. |
| mixed-model `ToolError` on `reindex` | Project embedded with a different model. | Re-register + full re-index (operator step). |
