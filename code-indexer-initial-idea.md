# Code Index MVP — Architecture & Tech Stack Decisions
 
**Status:** Draft v1 (MVP scope)
**Author:** Software architecture review
**Constraint adherence:** Every decision below carries an explicit rationale, a named alternative, and a “why not the alternative.” Nothing is asserted without a reason. Version-sensitive claims are described by capability rather than by exact release number, because those move faster than this document will.
 
-----
 
## 0. TL;DR — the one thing to get right
 
Do **not** build a pure vector-DB tool. Build a **hybrid retrieval service**:
 
- **Two retrieval channels** that run together and get fused: a **lexical/sparse** channel (BM25-style) and a **dense/semantic** channel (code embeddings). Pure semantic search collapses on short keyword queries; pure lexical misses “I don’t know the symbol name” queries. You need both.
- **Incremental, hash-based re-indexing** so the index does not go stale on every commit. Staleness is the single biggest reason teams abandon index-first code search.
- **MCP as the primary agent interface**, HTTP/dashboard as the *human* monitoring interface. Your stated consumer is “agents and LLMs”; in 2026 agents consume retrieval over MCP, not a browser.
- Treat the index as a **candidate generator that agents verify**, not as ground truth. The agent still reads the real file before acting. This sidesteps the correctness objections to embedding search.
If you build only “index → embed → vector DB → dashboard,” you will reproduce the exact design that shipping teams (e.g., Claude Code) tried and then walked back. The hybrid + incremental + MCP design is what survives the known failure modes.
 
-----
 
## 1. Problem statement (restated precisely)
 
**Goal:** Given a pointer to a codebase, build and maintain a searchable index so that agents/LLMs can retrieve the *relevant* code spans for a query in one cheap call, instead of repeatedly scanning the whole repo.
 
**What “matching results” must actually mean.** This phrase hides a fork that determines the whole design:
 
|Query type             |Example                                                  |Best channel            |
|-----------------------|---------------------------------------------------------|------------------------|
|Natural-language → code|“where do we validate JWT expiry”                        |dense/semantic          |
|Symbol / keyword → code|`parseJwtClaims`, `RateLimiter`                          |lexical/BM25            |
|Structural / shape     |“all functions that call `db.Exec` without a context arg”|AST/structural (phase 2)|
 
A single-channel design only serves one row well. The MVP must serve the first two; structural is a deliberate phase-2 add.
 
**Non-goals for the MVP (explicitly out of scope):** multi-user auth, distributed/sharded indexing, a graph/call-hierarchy engine, fine-tuned embedding models, cross-repo “code graph” navigation, and a polished SPA dashboard. Each is a real feature; none is needed to prove the core value.
 
-----
 
## 2. The premise stress-test (why the obvious design is wrong)
 
The obvious design — “embed everything, store vectors, query by cosine similarity” — fails in four documented ways:
 
1. **Staleness.** Code changes constantly. A vector index is stale the moment a file is edited until it is re-embedded. Re-embedding the whole repo on every change is wasteful; not re-embedding returns wrong spans. → *Mitigation in this design: per-file content hashing + a file watcher; only changed files are re-chunked and re-embedded.*
1. **Short-query collapse.** Benchmarks in 2026 (CoREB) show short keyword queries — the most common developer search shape — drive most semantic models to near-zero ranking quality. → *Mitigation: a lexical/BM25 channel that excels exactly here, fused with the dense channel.*
1. **Chunking destroys logic.** Fixed-size text splitting cuts a function from its signature/return, so the retrieved chunk is misleading. The counter-evidence is real too: a 2025 scaling study found line-based chunking competitive with AST chunking and BM25 dominant on quality-per-millisecond. → *Mitigation: AST-aware chunking (cheap insurance, function/class as the natural unit) but do not over-invest; the lexical channel carries a lot of the load.*
1. **“Similar” ≠ “correct.”** Embedding similarity returns code that *looks* like the answer, not code that *is* the answer. → *Mitigation: the index returns candidates; the agent reads the real file before acting. Optionally add a reranker (phase 2) to reorder candidates before they reach the agent.*
**What survives the stress-test:** a persistent index is still worth building because (a) it saves tokens versus re-scanning (embedding search cuts retrieval tokens meaningfully versus raw grep; AST retrieval more), (b) it answers NL→code queries grep cannot, and (c) latency is sub-100ms versus multi-second agentic exploration. The index earns its place — as one channel in a hybrid, kept fresh incrementally.
 
-----
 
## 3. Architecture overview
 
```
                        ┌──────────────────────────────────────────────┐
                        │                  CONSUMERS                     │
                        │   Agents/LLMs (MCP)        Humans (browser)    │
                        └───────┬───────────────────────────┬───────────┘
                                │ MCP (stdio / HTTP+SSE)     │ HTTP (REST + dashboard)
                        ┌───────▼───────────────────────────▼───────────┐
                        │                 SERVICE LAYER                  │
                        │  FastMCP server        FastAPI app (+ static)  │
                        └───────┬───────────────────────────┬───────────┘
                                │                            │
                        ┌───────▼────────────────────────────▼──────────┐
                        │                 CORE ENGINE                    │
                        │  ┌──────────┐  ┌───────────┐  ┌─────────────┐  │
                        │  │ Indexer  │  │ Retriever │  │ Job/State   │  │
                        │  │ pipeline │  │ (hybrid)  │  │ manager     │  │
                        │  └────┬─────┘  └─────┬─────┘  └──────┬──────┘  │
                        └───────┼──────────────┼───────────────┼─────────┘
                                │              │               │
        ┌───────────────┐ ┌────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
        │ File watcher  │ │ Qdrant    │ │ Embedding  │ │ SQLite       │
        │ (watchdog) +  │ │ (dense +  │ │ runtime    │ │ (projects,   │
        │ hash store    │ │ sparse,   │ │ (sentence- │ │ files, runs, │
        │               │ │ fusion)   │ │ transformers)│ │ job status) │
        └───────────────┘ └───────────┘ └────────────┘ └──────────────┘
```
 
**Data flow — indexing:** discover files → filter (gitignore, binaries, size) → detect changed files by hash → AST-chunk changed files → embed chunks (dense) + build sparse vectors → upsert into Qdrant with metadata → record run/state in SQLite.
 
**Data flow — retrieval:** query arrives (MCP tool call or REST) → embed query (dense) + tokenize (sparse) → Qdrant hybrid search with project/path/language filters → server-side fusion (RRF) → (optional rerank) → return ranked spans with file path + line range + score.
 
-----
 
## 4. Tech stack decisions
 
Each decision: **Choice → Rationale → Alternative considered → Why not.**
 
### 4.1 Language / runtime — **Python 3.11+**
 
- **Rationale:** The entire code-RAG ecosystem (tree-sitter bindings, sentence-transformers, Qdrant client, the MCP Python SDK) is Python-first. You (the maintainer) are fluent in Python. Python 3.11+ for the meaningful async and performance improvements and `tomllib`.
- **Alternative:** Rust (LanceDB/Qdrant are Rust; native tree-sitter). **Why not:** far slower to MVP; embedding-model serving in Rust is immature; no payoff at MVP scale. Hot paths can drop to Rust later via existing libraries without rewriting the service.
- **Alternative:** TypeScript/Node. **Why not:** weaker local-embedding story; tree-sitter and embedding runtimes are less mature than Python’s.
### 4.2 Web framework — **FastAPI + Uvicorn**
 
- **Rationale:** Async (matches the I/O-bound indexer and concurrent query load), automatic OpenAPI docs (free machine-readable API surface), trivial static-file serving for the dashboard, Pydantic models for typed request/response validation.
- **Alternative:** Flask. **Why not:** sync-first; async support is bolted on; you’d hand-roll schema validation.
- **Alternative:** Django. **Why not:** heavyweight; ORM/admin/templating you don’t need for a single-purpose tool.
### 4.3 Agent interface — **MCP via the official Python SDK (FastMCP)** *(primary interface)*
 
- **Rationale:** Your consumer is agents. MCP is the 2026 standard for exposing tools/resources to LLM agents (Claude Code, Cursor, Codex all speak it). Exposing the retriever as MCP tools means any compatible agent can use it with zero glue. This is the difference between “a search box” and “a tool agents actually call.”
- **MCP tools to expose:** `search_code(query, project, top_k, filters)`, `list_projects()`, `get_index_status(project)`, `get_chunk(chunk_id)` (fetch the exact span). Optionally a `reindex(project)` action.
- **Transport:** stdio for local single-agent use; HTTP+SSE for the always-on service shared by multiple agents.
- **Alternative:** REST-only and let each agent author a custom client. **Why not:** every agent integration becomes bespoke work; defeats the “agents just use it” goal. (REST still exists — for the dashboard and scripting — but it is secondary.)
### 4.4 Code parsing & chunking — **tree-sitter (AST-aware), split-then-merge (cAST pattern)**
 
- **Rationale:** tree-sitter is battle-tested (powers editor syntax engines), has grammars for ~all mainstream languages, and exposes a real AST. The cAST algorithm — recursively merge AST nodes into size-bounded chunks, split nodes that overflow, and guarantee concatenation reproduces the file — keeps functions/classes intact and is language-agnostic. Published results show measurable RAG gains (Recall@5 on RepoEval, Pass@1 on SWE-bench) over fixed-size splitting.
- **Honest caveat (do not over-invest):** a 2025 scaling study found line-based chunking competitive with AST chunking once you have a strong lexical channel. So: implement AST chunking because the per-chunk quality and metadata (node type, symbol name) are useful, but treat it as a moderate win, not the centerpiece. The centerpiece is hybrid retrieval.
- **Library:** `tree-sitter` + a prebuilt grammar bundle (e.g. `tree-sitter-language-pack`) so you don’t compile grammars per language.
- **Chunk sizing:** target ~32–64 lines / ~300–800 tokens per chunk with small overlap; never split a function below its signature. Store `{file_path, start_line, end_line, node_type, symbol_name, language}` as metadata on every chunk.
- **Alternative:** LangChain `RecursiveCharacterTextSplitter` with code separators. **Why not:** regex-ish; fooled by `def`/`class` inside strings/comments; loses node-type metadata.
- **Alternative:** whole-file chunks. **Why not:** dilutes embedding relevance and blows the context budget for large files (only competitive at very large context budgets).
### 4.5 Embedding model — **Nomic CodeRankEmbed (137M)** *(local default)*
 
- **Rationale:** It is **code-specialized**, **Apache-2.0**, and **small (~0.5 GB)** so it runs on CPU or a modest GPU — which matters for a tool meant to run locally on a developer machine. On CodeSearchNet it posts strong per-language scores (e.g., Python ~78.4, Go ~92.7), close to 7B models and well above general-purpose embedders, at a fraction of the footprint. It is instruction-aware (prepend the documented query prefix; embed documents without it).
- **Runtime:** `sentence-transformers` (loads CodeRankEmbed with `trust_remote_code`, broad model support, easy to swap models). Not `fastembed` for the default, because broad/code-model coverage is less certain there — confirm before relying on it.
- **Upgrade path (documented, not default):**
  - **Qwen3-Embedding-0.6B** (Apache-2.0, multilingual + code, Matryoshka dimensions) if NL queries in non-English or mixed prose/code dominate.
  - **Nomic Embed Code (7B)** only if you have an H100-class GPU and need top-tier recall; ~26 GB, overkill for MVP.
- **Why local, not a hosted API (Voyage/Gemini code embeddings):** the requirement is local hosting; hosted code embedders are stronger on some benchmarks but violate the constraint, add latency/cost, and ship your source to a third party. Keep the model pluggable so a hosted option can be added behind the same interface later.
- **Hard operational rule:** changing the embedding model invalidates the entire index (different vector space). Version the model name in metadata and force a full re-index on change. Do not silently mix vectors from two models.
### 4.6 Vector database — **Qdrant** *(local, single container)*
 
- **Rationale:** Qdrant supports **native hybrid search** — dense **and** sparse vectors in one collection with **server-side fusion (Reciprocal Rank Fusion)** — which is exactly the two-channel retrieval this design needs, without you hand-rolling fusion. It has rich **payload filtering** (filter by `project_id`, `language`, `path` prefix), strong concurrent-client handling (indexer writing while agents query), and runs as a single local Docker container or binary. Since the product already runs an HTTP service, “it needs a server process” is not a real cost here.
- **Alternative: LanceDB** (embedded, zero separate process, disk-based for larger-than-RAM, built-in versioning, excellent DX). **Why not as default:** documented multi-process concurrency limitations clash with this access pattern (indexer + dashboard + multiple agents hitting it at once). **Use LanceDB instead if** you want a true single-binary, zero-dependency embedded deployment and can serialize writes — it is the right call for a strictly single-process desktop build. This is a genuine fork; pick by deployment shape.
- **Alternative: ChromaDB.** **Why not:** simplest API but weakest hybrid/filtering story and the lowest scaling ceiling of the group — risky for large monorepos (millions of chunks).
- **Alternative: pgvector.** **Why not:** great if you already run Postgres and want ACID + SQL filtering; for a standalone code tool it adds a Postgres dependency and its hybrid story is more manual. Reasonable second choice if a relational store is wanted anyway.
### 4.7 Lexical / sparse channel — **BM25-style sparse vectors in Qdrant** *(non-negotiable)*
 
- **Rationale:** This is the channel that saves you from short-query collapse and exact-symbol lookups. Implement as sparse vectors stored alongside dense vectors in the same Qdrant collection so a single hybrid query covers both. (If you choose LanceDB instead, use its full-text search index for this channel.)
- **Why it’s separate from the embedding decision:** the lexical channel often contributes more to final quality than the choice of embedding model. Do not skip it to “ship the vector part first.”
### 4.8 Reranking — **cross-encoder reranker** *(phase 2, optional)*
 
- **Rationale:** After hybrid fusion returns ~50 candidates, a cross-encoder (e.g. a BGE reranker, or Nomic CodeRankLLM for code) reorders the top-k for precision before the agent sees them. Biggest single quality lever after hybrid itself.
- **Why phase 2:** adds latency and a second model to host; the MVP can ship with fusion-only ranking and add reranking once retrieval quality is measured.
### 4.9 Change detection — **per-file content hash + `watchdog` file watcher**
 
- **Rationale:** This is the staleness answer. On index, store a SHA-256 of each file’s content. On re-index (manual, scheduled, or watcher-triggered), hash again; only re-chunk/re-embed files whose hash changed; delete chunks for removed files. The `watchdog` library gives cross-platform filesystem events for near-real-time freshness on active projects.
- **Refinement:** roll per-file hashes up into a per-directory/per-project digest (Merkle-style) so “has anything changed under `src/auth/`” is a cheap comparison.
- **Alternative:** full re-index on a timer. **Why not:** wasteful and slow on large repos; defeats the latency/token advantage.
- **Alternative:** git-diff-based detection. **Why not for MVP:** assumes git and clean working trees; the hash approach works for any directory including uncommitted edits. (git integration is a fine phase-2 optimization.)
### 4.10 Metadata / state store — **SQLite**
 
- **Rationale:** Track projects (path, embedding-model version, created/updated), per-file state (path, hash, language, chunk count, last indexed), and index runs / job status (queued/running/failed, counts, errors). SQLite is zero-config, single-file, transactional, and ships with Python. The dashboard reads from it.
- **Alternative:** put this metadata only in Qdrant payloads. **Why not:** you need relational queries for the dashboard (“files per project,” “last run status,” “failed files”) and a clean source of truth for job state that is independent of the vector store.
- **Alternative:** Postgres. **Why not for MVP:** unnecessary operational weight for single-node local use.
### 4.11 Background work — **in-process async worker (asyncio), no Celery**
 
- **Rationale:** Indexing is the only long-running job. An in-process background task/queue with a bounded worker pool, status written to SQLite, is enough for single-node local use and keeps the deployment to “one process + one Qdrant container.”
- **Alternative:** Celery + Redis/RabbitMQ. **Why not for MVP:** adds a broker and worker processes for a workload that does not need distributed execution yet. Revisit only if you parallelize indexing across machines.
### 4.12 Dashboard — **server-rendered (Jinja2) or a single static page**, not a SPA
 
- **Rationale:** The dashboard’s job is *monitoring*, not interaction: list indexed projects, per-project file/chunk counts, last-index time, job status/progress, errors, and an optional manual “re-index” / “search test” box. FastAPI + Jinja2 (or one static HTML+JS page hitting the REST API) delivers this with near-zero build tooling.
- **Alternative:** React/Next SPA. **Why not for MVP:** build pipeline and state management you don’t need for a read-mostly status page. Upgrade later if the dashboard grows interactive features.
-----
 
## 5. Indexing pipeline (detailed)
 
1. **Register project:** record `{project_id, root_path, embedding_model_version}` in SQLite; create/ensure a Qdrant collection (or a per-project payload filter on a shared collection — see §6).
1. **Discover files:** walk the tree; respect `.gitignore`; skip binaries, generated dirs (`node_modules`, `dist`, `.git`), files over a size cap, and unsupported languages.
1. **Diff by hash:** for each file, compare current SHA-256 to the stored hash. Partition into *new*, *changed*, *unchanged*, *deleted*.
1. **Chunk changed/new files:** parse with tree-sitter → cAST split-then-merge → emit chunks with metadata `{file_path, start_line, end_line, language, node_type, symbol_name, file_hash}`.
1. **Embed + sparse-encode:** batch chunks through CodeRankEmbed (dense) and the sparse encoder (BM25). Batch sizes tuned to available RAM/GPU.
1. **Upsert:** write chunks to Qdrant with both vectors + payload. Use a deterministic `chunk_id` (e.g. `hash(project_id + file_path + start_line + file_hash)`) so re-runs are idempotent.
1. **Delete stale:** remove all chunks whose `file_hash` no longer matches (changed files) and all chunks for deleted files.
1. **Record run:** update SQLite file state + write an index-run row with counts, duration, and any per-file errors. Surface this to the dashboard.
**Idempotency & crash-safety:** because chunk IDs are content-derived and the file-state table is the source of truth, an interrupted run can be re-run safely — it re-processes only what’s still out of date.
 
-----
 
## 6. Retrieval pipeline (detailed)
 
1. **Input:** `query`, `project` (or “all”), `top_k`, optional `filters` (language, path prefix, node_type).
1. **Encode:** dense-embed the query (with the model’s query instruction prefix); sparse-encode the query tokens.
1. **Hybrid search:** issue one Qdrant query carrying both vectors + a payload filter for `project_id` and any user filters.
1. **Fuse:** Qdrant performs RRF over the dense and sparse result lists server-side, returning a single ranked list.
1. **(Phase 2) Rerank:** cross-encoder reorders the top ~50 → top_k.
1. **Return:** for each hit, `{file_path, start_line, end_line, language, symbol_name, score, snippet}`. The agent uses `file_path`+line range to read the *live* file before acting (index = candidate generator, file = ground truth).
**Collection strategy:** a single shared Qdrant collection with `project_id` in the payload (filter at query time) is simplest and supports cross-project search. Switch to a collection-per-project only if you need hard isolation or per-project lifecycle (drop a project = drop a collection). Document the choice; don’t mix silently.
 
-----
 
## 7. Data model
 
**SQLite**
 
```sql
projects(
  id TEXT PRIMARY KEY, root_path TEXT, embedding_model TEXT,
  created_at, updated_at
)
files(
  id TEXT PRIMARY KEY, project_id TEXT, path TEXT, language TEXT,
  content_hash TEXT, chunk_count INT, last_indexed_at,
  FOREIGN KEY(project_id) REFERENCES projects(id)
)
index_runs(
  id TEXT PRIMARY KEY, project_id TEXT, status TEXT,           -- queued|running|done|failed
  files_total INT, files_changed INT, chunks_written INT,
  started_at, finished_at, error TEXT
)
```
 
**Qdrant point**
 
```
id:       deterministic chunk_id
vectors:  { "dense": <float[]>, "sparse": <sparse vector> }
payload:  { project_id, file_path, start_line, end_line,
            language, node_type, symbol_name, file_hash,
            embedding_model }
```
 
-----
 
## 8. API surface
 
**MCP tools (primary, for agents)**
 
- `search_code(query, project?, top_k?, filters?) -> [hits]`
- `list_projects() -> [{project, files, chunks, last_indexed}]`
- `get_index_status(project) -> {status, progress, errors}`
- `get_chunk(chunk_id) -> {file_path, lines, content}`
- `reindex(project) -> {run_id}` *(action; guard appropriately)*
**REST (secondary, for dashboard/scripting)**
 
- `POST /projects` (register + index), `GET /projects`, `GET /projects/{id}/status`
- `POST /projects/{id}/reindex`
- `POST /search` (mirrors the MCP tool, for the dashboard test box)
- `GET /` (dashboard), `GET /healthz`
Keeping REST and MCP thin wrappers over the *same* core engine functions prevents the two surfaces from drifting.
 
-----
 
## 9. Staleness & correctness strategy (the highest-risk area)
 
This is where index-first tools die, so it gets its own section.
 
- **Freshness:** hash-diff incremental indexing + `watchdog` events. On a watched project, an edit triggers re-index of only that file within seconds.
- **Correctness guardrail:** the contract with the agent is “these are *candidates*; read the file before you trust the content.” The retriever returns precise line ranges to make that cheap. This is what makes approximate retrieval safe.
- **Model/version safety:** every chunk records `embedding_model`. Changing the model forces a full re-index; the system refuses to serve mixed-model results.
- **Observability:** the dashboard exposes last-index time per file, failed files, and run history so a human can see when the index is behind reality.
-----
 
## 10. Known limitations & failure modes (stated honestly)
 
1. **Cross-file reasoning is weak.** Chunk-level retrieval doesn’t follow imports/call graphs. “What calls this function across the repo” is a structural/graph query the MVP won’t answer well. → Phase 2: add `ast-grep`/structural search and/or a lightweight call graph.
1. **Very large monorepos** (tens of millions of chunks) will stress a single Qdrant node’s memory/quantization settings. MVP targets single-repo / moderate scale; document the ceiling.
1. **Embedding-model ceiling.** There is a proven mathematical limit on what any fixed-dimension embedding can represent; semantic recall has a hard cap. The lexical channel and agent verification are the hedge, not bigger vectors.
1. **First-index cost.** Initial full index of a large repo is slow (parse + embed everything). Communicate progress via the dashboard; it’s a one-time cost amortized over many queries.
1. **Language coverage** is bounded by available tree-sitter grammars; unsupported files fall back to line-based chunking (degraded but not broken).
-----
 
## 11. Build sequence (phased)
 
- **Milestone 1 — Spine:** project registration, file discovery + gitignore/binary filtering, hash-diff, SQLite state. No embeddings yet. Prove incremental detection works.
- **Milestone 2 — Index + dense search:** tree-sitter chunking → CodeRankEmbed → Qdrant dense-only search via REST. Prove NL→code retrieval returns sane spans.
- **Milestone 3 — Hybrid:** add the sparse/BM25 channel + RRF fusion. Measure quality on a handful of real queries (NL and keyword). This is where quality jumps.
- **Milestone 4 — MCP:** expose the retriever as MCP tools; connect a real agent and confirm it calls `search_code` and reads files.
- **Milestone 5 — Dashboard + watcher:** monitoring page + `watchdog` auto-reindex.
- **Phase 2 (post-MVP):** reranker, structural/AST-grep search, git-aware diffing, model-pluggability for hosted embedders, multi-repo.
-----
 
## 12. Decision log (ADR summary)
 
|# |Decision       |Chosen                                      |Key reason                                                                |Main alternative (why not)                                                |
|--|---------------|--------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
|1 |Overall shape  |Hybrid retrieval service, not pure vector DB|Pure semantic collapses on keyword queries; pure lexical misses NL queries|Vector-only (fails short-query case)                                      |
|2 |Agent interface|MCP (primary)                               |Agents consume retrieval over MCP in 2026                                 |REST-only (bespoke per-agent glue)                                        |
|3 |Language       |Python 3.11+                                |Ecosystem + maintainer fluency                                            |Rust (slow to MVP); Node (weak local embeddings)                          |
|4 |Web framework  |FastAPI                                     |Async + OpenAPI + Pydantic                                                |Flask (sync); Django (heavy)                                              |
|5 |Chunking       |tree-sitter AST (cAST)                      |Preserves function/class units + metadata                                 |Char splitter (breaks logic); whole-file (dilutes)                        |
|6 |Embedding model|CodeRankEmbed 137M (local)                  |Code-specialized, Apache-2.0, small/CPU-friendly                          |Nomic 7B (needs H100); hosted (violates local constraint)                 |
|7 |Vector DB      |Qdrant                                      |Native dense+sparse hybrid + RRF + filtering + concurrency                |LanceDB (concurrency limits) / Chroma (weak hybrid) / pgvector (extra dep)|
|8 |Lexical channel|BM25 sparse in Qdrant                       |Saves short-query/exact-symbol cases                                      |Skipping it (quality loss)                                                |
|9 |Reranker       |Cross-encoder, phase 2                      |Precision lever after fusion                                              |None (acceptable for MVP)                                                 |
|10|Freshness      |Hash-diff + watchdog                        |Only re-embed changed files                                               |Full re-index on timer (wasteful)                                         |
|11|State store    |SQLite                                      |Zero-config relational truth for dashboard/jobs                           |Qdrant-payload-only (no relational queries)                               |
|12|Background work|In-process asyncio worker                   |One workload, single node                                                 |Celery (broker overhead)                                                  |
|13|Dashboard      |Jinja2 / static page                        |Monitoring is read-mostly                                                 |React SPA (build overhead)                                                |
 
-----
 
## References (consulted during this design)
 
- cAST: Structural chunking via AST — arxiv.org/abs/2506.15655 ; reference impl github.com/yilinjz/astchunk
- Practical Code RAG at Scale (line-vs-AST, BM25 trade-offs) — arxiv.org/abs/2510.20609
- Nomic Embed Code / CodeRankEmbed (benchmarks, license) — nomic.ai/news/introducing-state-of-the-art-nomic-embed-code ; huggingface.co/nomic-ai/CodeRankEmbed
- Qwen3-Embedding (alt model, code+multilingual) — huggingface.co/Qwen/Qwen3-Embedding-8B
- Vector DB comparisons (Qdrant/LanceDB/Chroma/pgvector, local + hybrid) — encore.dev/articles/best-vector-databases ; 4xxi.com/articles/vector-database-comparison ; firecrawl.dev/blog/best-vector-databases ; callsphere.ai/blog/vector-database-benchmarks-2026
- Agentic search vs index, MCP as interface, staleness — morphllm.com/codebase-indexing ; morphllm.com/agentic-search ; zylos.ai/research/2026-04-19-codebase-intelligence ; ceaksan.com/en/code-search-for-ai-agents-which-tool-when (CoREB short-query finding)
- tree-sitter — tree-sitter.github.io
*Note: model versions, DB releases, and benchmark standings in this space move fast. Re-verify the embedding-model and vector-DB specifics against current docs before you commit code.*
