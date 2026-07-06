# Noesis plugin for Claude Code

Connects an agent to [Noesis](https://github.com/behl1anmol/Noesis) — a local,
hybrid code-retrieval service — over MCP, and ships the **`noesis-mcp` skill**: a
usage guide, a full tool reference, and helper scripts so the agent can use
Noesis effectively without hand-holding.

## What you get

- **MCP connection** (`.mcp.json`) to a running Noesis service over HTTP. The
  `noesis:` tools (`search_code`, `structural_search`, `list_projects`,
  `get_index_status`, `get_chunk`, `reindex`) become available to the agent.
- **`noesis-mcp` skill** (`skills/noesis-mcp/`):
  - `SKILL.md` — when and how to use each tool; the golden retrieval path.
  - `references/` — `tools.md` (params/returns/errors), `workflows.md` (task
    recipes), `transports.md` (HTTP vs. stdio, registration), `troubleshooting.md`.
  - `scripts/` — `register_project.py` (register a repo via REST) and
    `healthcheck.py` (diagnose connectivity). Standard-library only.

## Prerequisites

The plugin talks to a Noesis service you run locally:

```bash
docker compose up -d                                          # Qdrant on :6333
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000    # the service
```

## Install

```
/plugin marketplace add behl1anmol/Noesis
/plugin install noesis@noesis-tools
```

At enable time you're asked for **Noesis base URL** (default
`http://127.0.0.1:8000`). Change it only if you run uvicorn on a different
host/port. The MCP endpoint is that URL + `/mcp/`.

## First use

```
list_projects
  → empty? register a repo:
    python "$CLAUDE_PLUGIN_ROOT/skills/noesis-mcp/scripts/register_project.py" /abs/repo --wait
search_code("what you're looking for, in plain words", project_id)
  → read the live file at each hit's file_path:start_line-end_line
```

Not connecting? `python "$CLAUDE_PLUGIN_ROOT/skills/noesis-mcp/scripts/healthcheck.py"`.

## Design notes

- **HTTP transport is the default**, not stdio. A marketplace-installed plugin is
  copied into a cache and cannot reference files outside its own directory, so a
  stdio server (`uv run --project <noesis-repo>`) can't be wired with plugin-root
  paths. HTTP only needs a URL. stdio remains available as a documented opt-in —
  see `skills/noesis-mcp/references/transports.md`.
- **Registration is REST-only by design** — an agent shouldn't silently index new
  filesystem roots. The skill documents this and ships the helper script for the
  operator step.

## License

MIT.
