<p align="center">
  <img src="assets/noesis-banner.png" alt="Noesis" width="900">
</p>


## Beyond search. Toward understanding

An AI-native code understanding engine that gives AI agents deep understanding of your codebase through hybrid retrieval, structural search, and local-first indexing.

## Quickstart

```bash
docker compose up -d                    # Qdrant (localhost only)
uv sync --all-groups
uv run python -m noesis.prefetch        # one-time asset download: grammars + embedding model
uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000
```

After prefetch the service makes zero outbound network calls at runtime —
no code, query, or metadata ever leaves the machine (ADR-25).
