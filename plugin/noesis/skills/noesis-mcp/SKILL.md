---
name: noesis-mcp
description: >-
  Use Noesis to understand and navigate a codebase through its MCP tools —
  semantic + lexical hybrid code search, AST structural search, and reading
  indexed chunks. Trigger when the user asks to find where something is
  implemented, locate call sites or definitions, search a repo by meaning
  rather than exact text, run an ast-grep pattern, check index status, or
  reindex a project, and the Noesis MCP server (tools prefixed `noesis:`) is
  connected. Also the guide for connecting/registering Noesis when its tools
  are expected but missing.
---

# Using Noesis over MCP

Noesis is a **local** hybrid code-retrieval service. It gives you deep, current
knowledge of an indexed codebase through three retrieval modes: dense embeddings,
lexical BM25, and AST structural matching. Everything runs on `127.0.0.1` — no
code, query, or metadata leaves the machine.

This plugin connects you to Noesis over MCP. The tools appear under the `noesis:`
prefix: `search_code`, `structural_search`, `list_projects`, `get_index_status`,
`get_chunk`, `reindex`.

## The one rule that matters

**Search hits are candidates, not ground truth.** `search_code` returns ranked
spans; `get_chunk` returns the *indexed snapshot*, which can lag the live file.
Always read the live file at the returned `file_path:start_line-end_line` before
you edit, quote, or reason about code. `structural_search` is the exception — it
scans the live filesystem, so its matches are current.

## Golden path

```
list_projects                      → get the project_id (the `id` field)
  │                                   (empty? see "No project yet" below)
search_code(query, project_id)     → ranked spans, each with a chunk_id
  │
read the live file at file_path:start_line-end_line   (Read tool)
  └─ or get_chunk(chunk_id)         → the exact indexed span, to compare
structural_search(pattern, lang, project_id)          → precise AST call sites
```

Typical task — *"find where X is implemented and change it"*:
1. `list_projects` → pick the `id` for the repo in question.
2. `search_code("what X does, in plain words", project_id)` — describe intent,
   not keywords. Hybrid retrieval handles both.
3. Open the top hits' live files at their line ranges to confirm.
4. Use `structural_search` to enumerate every call site precisely before editing
   (e.g. `pattern="foo($$$ARGS)"`), so you don't miss one.

## Tool quick reference

| Tool | Use it to | Key args |
|------|-----------|----------|
| `search_code` | Find code by meaning or keyword | `query`, `project_id`, `top_k=10`, `channel="hybrid"`, `language?`, `rerank?` |
| `structural_search` | Match exact AST shapes on live files | `pattern`, `language`, `project_id`, `paths?`, `max_results?` |
| `list_projects` | Discover project ids + roots | *(none)* |
| `get_index_status` | Check a project's last index run | `project_id` |
| `get_chunk` | Fetch one indexed span in full | `chunk_id` (from a search hit) |
| `reindex` | Re-index after files changed | `project_id` |

Every tool that takes a `project_id` raises `unknown project_id` if it isn't
registered. Full parameter, return-shape, and error tables: **[references/tools.md](references/tools.md)**.
Task recipes (find-and-read, safe refactor, structural sweeps): **[references/workflows.md](references/workflows.md)**.

## No project yet (registration)

If `list_projects` is empty, or the repo you need isn't listed, it must be
**registered and indexed first**. Registration is *not* an MCP tool — it is a
one-time operator step over the REST API (or the dashboard at
`http://127.0.0.1:8000/`). This plugin ships a helper:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/noesis-mcp/scripts/register_project.py" \
  /absolute/path/to/repo --wait
```

It POSTs to `/projects`, prints the new `project_id`, and (`--wait`) polls until
the first index finishes. First index downloads/loads the embedding model, so it
can take a while on CPU. Don't invent a `project_id` — register, then use the one
returned. Why registration is REST-only, and the mixed-model guard:
**[references/transports.md](references/transports.md)**.

## When the tools are missing

If `noesis:` tools aren't available at all, the service or connection is the
problem, not the query. Diagnose before retrying:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/noesis-mcp/scripts/healthcheck.py"
```

It checks `/healthz`, lists projects, and tells you exactly what's down (service
not running, wrong port, Qdrant down). Fixes: **[references/troubleshooting.md](references/troubleshooting.md)**.

## Reindex loop

After files change, the index goes stale for `search_code`/`get_chunk` (not for
`structural_search`). To refresh:

```
reindex(project_id)                → returns run_id immediately (async)
get_index_status(project_id)       → poll until status == "done"
```

`reindex` is incremental — only changed files are re-embedded. Poll
`get_index_status`; don't block. `status` moves `running → done` (or `failed`,
with an `error`).
