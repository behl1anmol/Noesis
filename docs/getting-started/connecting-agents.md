# Connecting agents (MCP)

MCP is Noesis's primary interface: agents consume the same six tools over either transport — HTTP or stdio — backed by the same core engine as REST.

The six tools — `search_code`, `structural_search`, `list_projects`, `get_index_status`, `get_chunk`, `reindex` — are documented in the [MCP tools reference](../reference/mcp-tools.md).

!!! tip "Running more than one agent? Start at Option C"
    A standalone stdio server loads its own copy of the embedding model, and that copy is **not** shared between processes — ten agents is ~13.4 GB of RAM. [Option C](#option-c-shared-server-via-the-stdio-shim) gives each agent a 165 MB proxy onto one shared server instead (~3.1 GB for the same ten), with the numbers below.

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

## Option C — shared server via the stdio shim

Recommended whenever several agents run at once ([ADR-86](../project/decisions.md)). Same spawn-a-command shape as Option B, one extra flag:

```bash
claude mcp add noesis -- uv run --project /absolute/path/to/noesis python -m noesis.mcp --shared
```

`--shared` runs a thin stdio-to-HTTP MCP proxy that holds no model, no Qdrant connection pool and no state-DB handle. It talks to one shared server that holds those once — the daemon-plus-thin-client shape a language server or `docker` uses.

| Flag | Effect |
|---|---|
| `--shared` | Proxy to a shared server instead of building core resources in this process. |
| `--port N` | Port of the shared server. Defaults to `[server] port` (default `8000`). |
| `--no-spawn` | Fail if no server is answering, instead of starting one. For setups where the server is managed by systemd, a launch agent, or an operator. |

### Why: the duplication is in RAM, not on disk

The intuition most people have — "won't they re-download the model?" — is the wrong worry, and fixing the wrong one costs 10 GB:

| Measured | Standalone (`python -m noesis.mcp`) | Shim (`--shared`) |
|---|---|---|
| Process holding the model | 1342 MB RSS; 1140 MB PSS with two running, `Shared_File` **0 MB** | — |
| Per-agent process | 1342 MB | **165 MB RSS / 133 MB PSS** |
| Shared server | — | 1427 MB, once |
| Ten agents | **~13.4 GB** | **~3.1 GB** |

`Shared_File = 0 MB` is the whole point: the weights land in private tensors, not in a shared file mapping, so two processes running the same model share none of it. The **on-disk cache, by contrast, is already shared** — 523 MB, one copy, and a warm second process loads in 11.7 s and downloads nothing. Caching was never the problem; resident memory was.

This topology is only safe because one process now serves every agent's queries, which is exactly the reader-vs-reader concurrency the [search pool and gate](../internals/vector-store.md#connection-model-adr-83) were built to make correct.

### How one server gets elected

Ten agents starting at once must produce one server, and the nine losers must wait for it rather than fail while it is still coming up:

1. **Probe the port** — the warm path, no lock at all, measured at 0.07 s.
2. **Take an exclusive `filelock`** if nothing answered. `filelock` over a PID file or a SQLite row for one property neither has: **the OS releases it when the holder dies**, so a killed starter cannot wedge every future shim.
3. **Re-check under the lock** — whoever was ahead may already have started it. This is the branch 9 of 10 shims take.
4. **Spawn detached and poll readiness.** The port bind is the final arbiter: two servers on one port cannot both exist.

Verified end to end: 10 simultaneous cold shims produced exactly 1 server, 3.4–6.7 s each.

There is deliberately **no endpoint or discovery file**. The port is configured (`[server] port`, read by both sides), not discovered, and liveness is the probe — an earlier draft that published an endpoint file silently proxied an operator who asked for port 8199 to a live server on 8123.

!!! note "Default is still standalone"
    Without `--shared`, `python -m noesis.mcp` behaves exactly as it always has. No existing agent configuration changes behaviour.

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
| First `search_code` very slow (minutes) | Embedding model assets (~550 MB) weren't fetched before startup — run `uv run python -m noesis.prefetch` once, or check `assets`/`embedder_ready` on `/healthz` (HTTP) or `embedder_assets`/`embedder_ready` on `get_index_status` (MCP, works over stdio too — `/healthz` isn't reachable there). The service also warms the model in the background at startup (ADR-77), so this should only bite if prefetch was skipped |
| First `search_code` slightly slow (seconds, on CPU) | Normal model-load cost even with assets cached — the background warm-up (above) usually absorbs this before the agent's first call arrives |
| `reindex` returns a ToolError about the embedding model | Mixed-model guard: the stored index was built with a different embedder — a full re-index with the current model is required |
| `429` from `/search`, or a `search_code` ToolError saying "at search capacity" | More concurrent searches than `query_connections + query_queue_depth` ([ADR-84](../project/decisions.md)). Nothing failed and nothing ran — retry after `Retry-After` (the message carries the live counts), or raise `[qdrant] query_connections` / `query_queue_depth`. **Busy, not down**: a dead server gives connection-refused or a failing `/healthz` |
| `429` from `/projects/{id}/reindex` or `reindex-pending`, a **202** with `"status": "capacity_reached"` from `POST /projects` (it registers before it launches, so the registration is reported rather than hidden behind an error), or a `reindex` ToolError saying "at indexing capacity" | The machine is at `[indexing] max_concurrent_index_runs` ([ADR-85](../project/decisions.md)). Nothing was started and nothing was queued — retry once a run finishes, or raise the cap. The cap is machine-wide and holds across processes, so another Noesis process may be holding the slots |
| `--shared` exits with "no Noesis server answering … and `--no-spawn` was given" | Nothing is listening on `[server] port` and the shim was told not to start one. Start the server (`uvicorn noesis.app:app --host 127.0.0.1 --port 8000`) or drop `--no-spawn` |
| `--shared` exits with "server did not become ready" | The elected starter spawned a server that never answered `/healthz` within 60 s. Its output is in `$XDG_RUNTIME_DIR/noesis/server.log`, or `~/.cache/noesis/server.log` when that variable is unset — usually Qdrant being down or a port already taken by something else |
| `--shared` starts a *second* server, or agents disagree about what is indexed | The shim probes exactly the port it was given and nothing else — the port is configured, never discovered. If `--port` (or `[server] port`) doesn't match where the server actually listens, the shim will start its own there. Check both sides |
