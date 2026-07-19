# Security model

Noesis is local-only by design: after a one-time model download at install time, **no code, query, or metadata ever leaves the machine** ([ADR-25](../project/decisions.md)).

## The trust boundary

There is no deliberate trust-boundary crossing at all. Every model runs in-process; the only sockets are localhost (Qdrant, the bound service). The system makes **zero outbound network calls at runtime** — model weights, tree-sitter grammars, and BM25 assets are fetched once by `python -m noesis.prefetch`, before any user code is read.

## Local-only is enforced structurally, not by convention

Hosted/remote embedding was **rejected by stakeholder decision**, not merely deferred (ADR-25, 2026-07-03). A hosted embedder would have sent every chunk of source code to a third-party API — a reversal of a security constraint disguised as a feature. The rejection is enforced mechanically:

- The `Embedder` Protocol has **no remote/credentials surface**, and no HTTP client dependency exists anywhere in `core/`.
- **CI greps** assert: no `sentence_transformers` import outside the two model boundaries (`core/embedder.py`, `core/reranker.py`), no HTTP client imports in `core/`, and no `0.0.0.0` binds (`.claude/scripts/ci_greps.sh`).
- Adding a remote implementation would require new dependencies, which trips the no-new-runtime-deps-without-a-decision rule and re-opens ADR-25 explicitly.

Guardrails over goodwill.

## Localhost binding

The service binds `127.0.0.1` only — never `0.0.0.0`. A wildcard bind would expose your source code and index to the network.

Two additional guards harden the local surface (`src/noesis/api/security.py`, `app.py`):

- **`TrustedHostMiddleware`** allows only `127.0.0.1` / `localhost` Host headers — a DNS-rebinding guard (a malicious website resolving to your loopback cannot forge a valid Host).
- **`verify_local_origin`** protects mutating endpoints (reindex, flags, device, delete, register).

## Defense-in-depth: the index is a retrievable surface

Even though nothing leaves the machine, the index itself can be queried — so secrets must never enter it:

- **Secret skip-list**: discovery excludes `.env`, key files, `*.pem`, and ~two dozen secret patterns before a byte is read.
- **Redaction pass** at chunk time as a second layer.
- **Structural search reuses the same discovery filters**, so it cannot match content that indexing would have excluded — a skip-listed file never appears in either surface.
- **Generated lockfiles** are skipped too (retrieval noise, and they can embed registry tokens).

## Telemetry is metadata-only

The usage page is powered by `query_log`, which records *that* a query ran and how it performed — interface (REST/MCP), channel, latency, result count. It **never stores query text** (`core/telemetry.py`), keeping proprietary code and intent out of the local database.

## Threat model summary

| Surface | Risk | Mitigation |
|---|---|---|
| Outbound network | Code/query exfiltration | No HTTP clients in `core/` (CI-greped); models in-process; prefetch is the only download, at install time |
| Service port | LAN exposure of code + index | `127.0.0.1` bind only; documented never-`0.0.0.0` rule |
| Browser | DNS rebinding to loopback | `TrustedHostMiddleware` host allowlist; local-origin check on mutations |
| Index contents | Secrets retrievable via search | Secret skip-list + redaction; filters shared with structural search |
| Local DB | Query text accumulating | Metadata-only telemetry, by decision |
| Dashboard file browser | FS disclosure | `GET /api/browse` returns directory names only, never file contents; localhost-only |
| ast-grep | Tool mutating user code | Rewrite capability never exposed — search only ([ADR-22](../project/decisions.md)) |

## What runs where

| Component | Location | Network |
|---|---|---|
| Embedder (CodeRankEmbed) | In-process worker thread | None |
| Reranker (bge-reranker-v2-m3) | In-process worker thread, lazy | None |
| BM25 term encoding | In-process (fastembed assets, prefetched) | None |
| Qdrant | Docker container | `127.0.0.1:6333` only |
| SQLite | Local file | None |
