# Quickstart

Index a repository and run your first hybrid and structural searches in about five minutes ([installation](installation.md) done, Qdrant running).

## 1. Start the service

```bash
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
```

!!! warning "Always bind 127.0.0.1"
    Never bind `0.0.0.0`. The entire security model is local-only ([ADR-25](../project/decisions.md)); a wildcard bind would expose your source code and index to the network. CI greps enforce this in the codebase.

Once running:

| Surface | URL |
|---|---|
| Dashboard | <http://127.0.0.1:8000/> |
| MCP endpoint | `http://127.0.0.1:8000/mcp/` (trailing slash required) |
| REST + OpenAPI docs | <http://127.0.0.1:8000/docs> |
| Health | `GET /healthz` |

## 2. Register a project

Either use the dashboard's **Add project** modal (folder browser, per-language scoping, pre-flight file-count preview — see [Dashboard](../reference/dashboard.md)), or one REST call:

```bash
curl -X POST http://127.0.0.1:8000/projects \
  -H 'content-type: application/json' \
  -d '{"root_path": "/absolute/path/to/your/repo"}'
# → 202 {"project_id": "…", "run_id": "…", "status": "accepted"}
```

Registration starts an index run in the background and returns immediately. Poll the run:

```bash
curl http://127.0.0.1:8000/runs/<run_id>
# while running, includes "progress": {"percent": …, "eta_s": …}
```

The first index loads the embedding model — allow extra time on CPU. Subsequent runs are incremental: only new and changed files are re-embedded (see [The indexing pipeline](../concepts/indexing-pipeline.md)).

## 3. Search

Hybrid search (dense semantic + BM25 lexical, fused with RRF):

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "validate jwt expiry", "project_id": "<project_id>", "top_k": 10}'
```

Each hit is a span — `chunk_id`, `file_path`, `start_line`, `end_line`, `language`, `symbol_name`, `score`, `snippet`. Hits are candidates, not ground truth: read the live file before acting on one.

Structural (AST) search matches syntax patterns against the **live filesystem**, never the index:

```bash
curl -X POST http://127.0.0.1:8000/structural-search \
  -H 'content-type: application/json' \
  -d '{"pattern": "def $NAME($$$ARGS): $$$BODY", "language": "python", "project_id": "<project_id>"}'
```

## 4. Optional: a config file

Noesis runs with zero config — every setting has a working default. To override, create `~/.config/noesis/config.toml` (full reference: [Configuration](../reference/configuration.md)):

```toml
[embedder]
batch_size = 32
# device = "cuda"        # pin; omit for auto-detect

[reranker]
enabled = false          # opt-in — see the evaluation page for the latency data

[git]
fast_path = true
```

## Next steps

- Connect an AI agent over MCP — [Connecting agents](connecting-agents.md)
- Keep the index fresh automatically — [Freshness: watcher and git fast-path](../concepts/freshness.md)
- Understand what happens under the hood — [Architecture](../concepts/architecture.md)
