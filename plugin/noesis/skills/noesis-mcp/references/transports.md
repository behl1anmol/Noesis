# Transports, connection, and registration

## What this plugin registers

The plugin ships `.mcp.json` with **one active server**: an HTTP connection to a
running Noesis service.

```json
{ "mcpServers": { "noesis": { "type": "http", "url": "${user_config.base_url}/mcp/" } } }
```

`${user_config.base_url}` defaults to `http://127.0.0.1:8000` and is prompted at
enable time (change it only if you run uvicorn on a non-default host/port). The
MCP endpoint is always `base_url` + `/mcp/` (note the trailing slash).

**Prerequisite:** the Noesis service must be running, and Qdrant must be up.

```bash
docker compose up -d                                          # Qdrant on :6333
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000    # the service
```

If the service is down when the plugin loads, the `noesis:` tools won't connect —
that's a service problem, not a query problem. Run `scripts/healthcheck.py`.

---

## HTTP vs. stdio — which and why

| | HTTP (default, active) | stdio (alternative) |
|---|---|---|
| Config | `type: http`, URL to a running service | `command: uv run … python -m noesis.mcp` |
| Who runs the server | You (`uvicorn …`) | The agent host spawns it per session |
| Needs a running service? | Yes, on `:8000` | No — self-spawns |
| Needs a path to the noesis repo? | No | **Yes**, an absolute one |
| Shares state with the dashboard | Yes (same process) | Separate process, same SQLite/Qdrant |

**Why HTTP is the default here.** A marketplace-installed plugin is copied into a
cache directory, and installed plugins cannot reference files outside their own
directory (`../` paths don't resolve post-install). stdio needs
`uv run --project <noesis-repo-root>`, and that root is *outside* the plugin — so
it cannot be expressed with `${CLAUDE_PLUGIN_ROOT}`-relative paths. HTTP has no
such dependency: it just needs a URL. Hence HTTP is active; stdio is opt-in with
a hand-supplied absolute path.

### Switching to stdio (opt-in)

Add Noesis as a stdio MCP server yourself, pointing at your local checkout:

```bash
claude mcp add noesis -- \
  uv run --project /absolute/path/to/noesis python -m noesis.mcp
```

The stdio server builds its own core resources from `config.toml` in the working
directory (same defaults as HTTP). Qdrant must still be up. Use stdio when you
don't want to run a long-lived uvicorn, and you're fine giving each session its
own server process.

---

## Registration is REST-only (there is no MCP tool for it)

The MCP surface exposes `reindex`, but **not** project registration. Registering
a new repo is an operator action, done once, over REST or the dashboard:

```bash
# via the bundled helper (recommended)
python "${CLAUDE_PLUGIN_ROOT}/skills/noesis-mcp/scripts/register_project.py" \
  /absolute/path/to/repo --wait

# or directly
curl -X POST http://127.0.0.1:8000/projects \
  -H 'content-type: application/json' \
  -d '{"root_path": "/absolute/path/to/repo"}'
# → 202 {"project_id": "…", "run_id": "…", "status": "accepted"}
```

`POST /projects` registers the repo and kicks off the first index in the
background. Poll `GET /projects/{id}/status` (or the `get_index_status` MCP tool)
until `status == "done"`. The first index downloads/loads the embedding model —
slow on CPU, fast on GPU.

### The mixed-model guard

A project is pinned to the embedding model it was first indexed with. Re-registering
the same `root_path` under a *different* model is refused (`409` over REST, a
`ToolError` from `reindex`) to avoid serving a half-migrated index. Switching
models means a full re-index, an operator step — surface this to the user rather
than trying to force it.

### Why REST-only, by design

Registration points Noesis at an arbitrary filesystem path and starts a heavy
indexing job. Keeping it off the agent-facing MCP surface (which is the retrieval
contract) means an agent can't silently index new roots; a human decides what
gets indexed. `reindex` is safe to expose because it only refreshes an
already-registered, already-trusted root.
