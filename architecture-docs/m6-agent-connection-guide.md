# M6 — Connecting a Real Agent to the Noesis MCP Server

The M6 exit criterion: an agent completes a task using `search_code` → read
file, and `structural_search`, end to end. This guide covers both transports
locally, plus the optional Colab run for real-model latency numbers.

## Prerequisites

```bash
docker compose up -d           # Qdrant on 127.0.0.1:6333
uv sync                        # installs fastmcp 3.4.x (mcp pinned <2, rule 5)
```

First index downloads/loads `nomic-ai/CodeRankEmbed` — allow time on CPU.

## Option A — streamable HTTP (service already running)

```bash
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
```

MCP endpoint: `http://127.0.0.1:8000/mcp/` (note trailing slash). Register
the project once over REST (registration is an operator step; the MCP surface
exposes `reindex`, not register):

```bash
curl -X POST http://127.0.0.1:8000/projects \
  -H 'content-type: application/json' \
  -d '{"root_path": "/absolute/path/to/repo"}'
```

Connect Claude Code:

```bash
claude mcp add --transport http noesis http://127.0.0.1:8000/mcp/
```

## Option B — stdio (agent host spawns the server)

```bash
claude mcp add noesis -- uv run --project /absolute/path/to/noesis python -m noesis.mcp
```

stdio builds its own core resources from `config.toml` in the working
directory (same defaults as the HTTP app; the two share one build path,
`noesis.app.build_runtime_context`). Qdrant must be up.

## The exit-criterion task

Ask the connected agent something like:

> Using the noesis tools: find where hybrid search fuses the dense and
> sparse channels, open that file at the returned lines to confirm, then
> use structural_search to list every `models.FusionQuery(...)` call site.

A passing run shows the agent calling `list_projects` → `search_code` →
reading the live file at the returned span → `structural_search`, with
sane spans at each step. `get_chunk(chunk_id)` (ids are in every search
hit, ADR-36) fetches the stored chunk to compare against the live file.

## Colab run (optional — real-model latency on GPU)

Local CPU indexing works but is slow; a T4 run gives comparable numbers to
the M4 benchmarks. In a fresh Colab notebook (GPU runtime):

```bash
# 1. Qdrant server (no Docker in Colab — use the static binary)
!wget -q https://github.com/qdrant/qdrant/releases/download/v1.15.5/qdrant-x86_64-unknown-linux-gnu.tar.gz
!tar xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
!nohup ./qdrant > qdrant.log 2>&1 &

# 2. Repo + deps
!git clone https://github.com/behl1anmol/Noesis && cd Noesis
!pip install uv && cd Noesis && uv sync

# 3. Service (Colab cell)
!cd Noesis && nohup uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &

# 4. Index + drive the MCP tools in-process
```

```python
import asyncio, httpx
from fastmcp import Client

async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as http:
        pid = (await http.post("/projects", json={"root_path": "/content/Noesis"})).json()["project_id"]
        while (await http.get(f"/projects/{pid}/status")).json()["status"] != "done":
            await asyncio.sleep(5)
    async with Client("http://127.0.0.1:8000/mcp/") as mcp:
        hits = (await mcp.call_tool("search_code", {
            "query": "hybrid RRF fusion of dense and sparse channels",
            "project_id": pid})).structured_content["hits"]
        print(hits[0])
        chunk = (await mcp.call_tool("get_chunk", {"chunk_id": hits[0]["chunk_id"]})).structured_content
        print(chunk["file_path"], chunk["start_line"], chunk["end_line"])
        matches = (await mcp.call_tool("structural_search", {
            "pattern": "models.FusionQuery($$$A)", "language": "python",
            "project_id": pid})).structured_content["matches"]
        print(matches)

await main()  # Colab top-level await works in ipython
```

Everything binds 127.0.0.1 inside the Colab VM — nothing is exposed
publicly (rule 2 holds there too).

## Local verification record (2026-07-04, this machine, CPU)

Performed by Claude (an MCP agent) against the live service over
streamable HTTP with a scratch collection (`noesis_chunks_m6verify`) —
results recorded in the M6 session log / PR body.
