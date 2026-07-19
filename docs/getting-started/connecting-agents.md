# Connecting agents (MCP)

MCP is Noesis's primary interface: agents consume the same six tools over either of two transports, backed by the same core engine as REST.

The six tools — `search_code`, `structural_search`, `list_projects`, `get_index_status`, `get_chunk`, `reindex` — are documented in the [MCP tools reference](../reference/mcp-tools.md).

## Option A — streamable HTTP

Use this when the service is already running (`uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000`). The MCP endpoint is:

```
http://127.0.0.1:8000/mcp/
```

!!! warning "The trailing slash matters"
    `/mcp` (no slash) returns a redirect that some MCP clients don't follow. Always configure `/mcp/`.

Connect Claude Code:

```bash
claude mcp add --transport http noesis http://127.0.0.1:8000/mcp/
```

## Option B — stdio

The agent host spawns the server itself; no separately running service is needed (Qdrant must still be up):

```bash
claude mcp add noesis -- uv run --project /absolute/path/to/noesis python -m noesis.mcp
```

The stdio entry point (`src/noesis/mcp/__main__.py`) builds its own core resources through the same shared build path as the HTTP app (`build_runtime_context` in `src/noesis/runtime.py`), reading `config.toml` per the normal [resolution order](../reference/configuration.md). Logging goes to stderr only, so JSON-RPC on stdout stays clean.

!!! tip "One state DB for both transports"
    The default DB path is anchored (`~/.local/share/noesis/noesis.sqlite`), never cwd-relative — so the HTTP server and a stdio MCP server spawned from any directory see the same projects. If an MCP host can't control its working directory, point `NOESIS_CONFIG` at your config file.

## Registering projects

Registration is deliberately an operator step: the MCP surface exposes `reindex`, not register. Register once over REST or the dashboard (see [Quickstart](quickstart.md)), then agents discover projects with `list_projects`.

## The typical agent loop

1. `list_projects` — get `project_id`s.
2. `search_code(query, project_id)` — ranked spans with `chunk_id`s.
3. Read the **live file** at the returned span (hits are candidates, not ground truth), or `get_chunk(chunk_id)` to fetch the exact indexed snapshot for comparison.
4. `structural_search(pattern, language, project_id)` — precise AST matches against live files when the question is structural ("every call to X without argument Y").
5. `get_index_status` / `reindex` when freshness is in doubt.

A real end-to-end example (the M6 exit-criterion task, including a Python `fastmcp.Client` driver) is in [`architecture-docs/m6-agent-connection-guide.md`](https://github.com/behl1anmol/Noesis/blob/main/architecture-docs/m6-agent-connection-guide.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 404 or client hangs on connect | Missing trailing slash — use `http://127.0.0.1:8000/mcp/` |
| Connection refused | Service not running, or agent configured for HTTP while only stdio was set up |
| `unknown project_id` from every tool | stdio server resolving a different state DB than the one you registered in — set `NOESIS_CONFIG` or use the anchored default; see the tip above |
| First `search_code` very slow | Embedding model loading lazily on first use — expected, especially on CPU |
| `reindex` returns a ToolError about the embedding model | Mixed-model guard: the stored index was built with a different embedder — a full re-index with the current model is required |
