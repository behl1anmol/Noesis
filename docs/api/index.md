# API reference

These pages are generated straight from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) using griffe's *static*
analysis — the package is never imported, so what you read here is exactly
what is in `src/noesis/` at the commit the site was built from.

| Group | Modules |
|---|---|
| [Indexing](core-indexing.md) | chunker, languages, discovery, gitfast, hashdiff, indexer, jobs |
| [Retrieval](core-retrieval.md) | retriever, vectorstore, structural |
| [Models](core-models.md) | embedder, reranker, compute |
| [Service & state](core-service.md) | config, state, watcher, telemetry, dashboard |
| [REST layer](api.md) | routes, dashboard adapter, security |
| [MCP layer](mcp.md) | server |
| [App & runtime](app.md) | app, runtime, prefetch, logging_config |

For narrative documentation of the same modules, see
[Concepts](../concepts/architecture.md) and
[Internals](../internals/embedder.md).
