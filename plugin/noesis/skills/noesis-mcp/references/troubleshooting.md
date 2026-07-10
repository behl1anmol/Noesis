# Troubleshooting

Run the bundled diagnostic first — it pinpoints most of these:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/noesis-mcp/scripts/healthcheck.py"
```

It probes `/healthz`, lists projects, and reports what's reachable.

---

## `noesis:` tools don't appear at all

The MCP server didn't connect. In order of likelihood:

1. **Service not running.** Start it:
   ```bash
   uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
   ```
2. **Qdrant down.** The service needs it:
   ```bash
   docker compose up -d      # from the noesis repo
   ```
3. **Wrong URL/port.** The plugin points at `${user_config.base_url}` + `/mcp/`.
   If you run uvicorn elsewhere, set `base_url` in the plugin config (`/plugin`
   → configure) to match, e.g. `http://127.0.0.1:9000`.
4. **Reload.** After changing config or `.mcp.json`, run `/reload-plugins` or
   restart the session.

Verify the endpoint by hand:
```bash
curl -sS http://127.0.0.1:8000/healthz          # → {"status":"ok"}
```

## `list_projects` is empty

Nothing is registered. Register the repo (one-time, REST-only):
```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/noesis-mcp/scripts/register_project.py" \
  /absolute/path/to/repo --wait
```
See [transports.md](transports.md#registration-is-rest-only-there-is-no-mcp-tool-for-it).

## `unknown project_id` / `unknown_project`

The id isn't registered (or you invented one). Call `list_projects` and use an
`id` from it. If the repo genuinely isn't there, register it.

## `get_index_status` shows `never_indexed` or `failed`

- `never_indexed`: registered but the first index hasn't run/finished. Trigger
  `reindex(project_id)` (or re-register) and poll until `done`.
- `failed`: read the `error` field. Common causes: the `root_path` vanished
  (moved/deleted repo), or Qdrant went down mid-run. Fix the cause, reindex.

## Search results look stale (don't match the live file)

The index lags the filesystem. Either:
- Trust the live file (always read it before acting), and/or
- `reindex(project_id)` and poll `get_index_status` until `done`.

`structural_search` never goes stale — it reads live files. Prefer it when you
need certainty about the *current* code.

## `unsupported_language` from `structural_search`

The `language` isn't supported for AST search. The error message lists the
supported set — pick one from it. (Semantic `search_code` still works across
languages; only structural AST matching is language-gated.)

## `structural_search` returns `truncated:true` or `timed_out:true`

- `truncated`: hit the `max_results` cap — raise it, or scope with `paths=`.
- `timed_out`: the scan exceeded its wall-clock budget; matches are partial.
  Narrow `paths=` to the relevant subtree and re-run.

## Reindex conflict (mixed model)

`reindex` raises a `ToolError` when the project was indexed with a different
embedding model than the service now uses. Switching models needs a full
re-index (re-register the root) — an operator decision. Surface it to the user;
don't force it.
