# Vector store

`src/noesis/core/vectorstore.py` wraps the Qdrant client with the hybrid dense + BM25-sparse collection layout and every rule the embedder boundary imposes on it.

## Role

One shared collection (default `noesis_chunks`) holds every project's chunks; queries scope to a project with a `project_id` payload filter. Each point carries two named vectors:

| Vector name | Kind | Config | Produced by |
|---|---|---|---|
| `dense` | dense | `size=embedder.dim`, `Distance.COSINE` | the active `Embedder` |
| `bm25` | sparse | `Modifier.IDF` (server-side IDF weighting) | client-side TF via `models.Document(model="Qdrant/bm25")` (fastembed) |

The dense size is read from `embedder.dim` at collection-creation time — never hardcoded — so a model swap is a config change plus a full re-embed, not a code change.

## Connection model (ADR-83)

`VectorStore` does not own one Qdrant connection; it is handed several, one set per role, and `runtime.py` is the only place allowed to construct them (CI greps for it). The production wiring is **K query connections + 1 index + 1 admin = K + 2**:

| Role | Used by | Pooled? |
|---|---|---|
| `client` (admin) | deletes, counts, `scroll`, `retrieve`, collection and payload-index setup | no |
| `index_client` | `upsert_chunks` only | no |
| `query_clients` | the sparse and hybrid channels of `search` — one connection checked out per query | **yes**, K of them |

**What the pool is for.** qdrant-client keeps its client-side inference bookkeeping — a `_batch_accumulator` dict and an `_embed_storage` FIFO — on the `QdrantClient` *object*, unsynchronized and keyed only by model name. Two threads that build a `models.Document` on one client can receive each other's embeddings, and the drain step recomputes only when the FIFO is empty and otherwise `pop(0)`s, so a single race leaves that client permanently offset: every later query silently returns another query's results, with no exception raised, until the process restarts. Reproduced 5/5 through this class's own methods, and measured at up to 15,864 wrong results in 16,000 queries against a live server. Independent of transport — REST and gRPC latch at the same rate — and unfixed upstream as of qdrant-client 1.19.

A pool makes that unreachable rather than merely guarded: no two threads ever meet on one client's inference state. It is sized by `[qdrant] query_connections` and paired **1:1** with the search executor's slots ([ADR-84](../project/decisions.md)), so a checkout finds a free connection whenever a thread is free. Checkout *blocks* when all are busy — the right back-pressure primitive at this layer; admission control that rejects rather than queues belongs one layer up, in the [search gate](runtime.md#search-concurrency-adr-8384).

**Why dense search is exempt.** Only `upsert_chunks` and the sparse and hybrid channels build a `models.Document`, which is what engages the racy inference path. A dense-only query does not, and neither does any other method here. Verified by instrumenting `ModelEmbedder._accumulate` and `_drain_accumulator`: **zero** calls for a dense query against one each for sparse and hybrid. Those paths are deliberately left unpooled and unlocked — there is nothing on them to protect, and making them contend would cost a slot for no safety.

**The lock floor.** Underneath the pool there is one `threading.Lock` per distinct client **object**, not per role. That distinction is load-bearing: two `QdrantClient(":memory:")` instances are two different *databases*, not two connections to one, so in-memory and embedded callers necessarily share a single client for all three roles. Keying locks by identity means every path then takes the *same* mutex; keying by role would leave search and upsert holding two different locks over one `ModelEmbedder` — [issue #48](https://github.com/behl1anmol/Noesis/issues/48)'s original race, reopened. In the production wiring the objects are distinct, so each lock is uncontended and costs an atomic.

`index_client` staying out of the pool means an index run can never consume a query connection however long its batch runs, and its own lock guards it against itself: `try_start_run` is per project, so two *different* projects can reach `upsert_chunks` at once (routine with `auto_reindex` on 2+ projects). `[indexing] max_concurrent_index_runs` ([ADR-85](../project/decisions.md)) bounds how many such runs exist at all, but that cap is machine-wide; this lock is what makes the overlap safe *within* a process.

## Payload

| Field | Meaning |
|---|---|
| `project_id` | owning project (keyword-indexed) |
| `file_path`, `start_line`, `end_line` | span location |
| `language` | canonical language (keyword-indexed) |
| `node_type`, `symbol_name` | AST metadata from the chunker |
| `file_hash` | SHA-256 of the source file at index time |
| `embedding_model` | versioning key — the model that produced the dense vector |
| `text` | full chunk text — serves search snippets (first 200 chars) and reranker input without re-opening files |

Keyword payload indexes on `project_id` and `language` are created immediately after the collection, before any bulk load.

## Design decisions

- **Deterministic point ids.** `chunk_point_id()` is a UUIDv5 over `"{project_id}:{file_path}:{start_line}:{file_hash}"` under the fixed namespace `CHUNK_NAMESPACE`. Re-indexing unchanged content rewrites the same points — idempotent by construction. The namespace constant must never change; doing so would orphan every existing point.
- **BM25 split between client and server ([ADR-32](../project/decisions.md)).** Local Qdrant has no server-side text inference, so the TF half of BM25 is computed client-side by qdrant-client's fastembed integration; IDF weighting is applied server-side via `Modifier.IDF`. Changing `BM25_MODEL_ID` changes tokenization and silently invalidates every stored sparse vector — treated like an embedding-model change (full re-index). Nothing leaves the machine ([ADR-25](../project/decisions.md)): the client talks only to localhost or `:memory:`.
- **Server-side RRF fusion.** Hybrid search issues two prefetches (dense + sparse, `max(prefetch_limit, top_k)` each, payload filter applied *inside* each prefetch so both candidate lists are project-scoped before fusion) and fuses with `FusionQuery(Fusion.RRF)`. The RRF constant is fixed by the server and not exposed — the design doc's k=60 is aspirational, not configured ([ADR-32](../project/decisions.md)). `dense` and `sparse` single-channel queries stay first-class because the M3 evaluation gate compares hybrid against exactly those baselines.
- **Mixed-model refusal.** `ensure_collection` raises if the collection exists with a different dense size, or without the `bm25` sparse vector (an M2-era collection whose points carry no sparse vectors — silently adding the config would leave the lexical channel empty). The fix is explicit: drop collection and state, re-index fully. A first-startup race between the HTTP and stdio processes is tolerated: the loser re-enters and verifies the winner's shape.
- **`close()` closes every connection.** De-duplicated by object identity, so the single-client default closes exactly once however many roles that one object fills. Each close is attempted even if an earlier one raised — one failure must never leak the rest — and the first exception is re-raised once all have been tried.
- **Ordered pruning.** `delete_file_chunks(..., exclude_file_hash=...)` spares points carrying the new hash — the indexer upserts a file's new chunks first, then prunes old ones, so a failure never leaves a file with zero searchable chunks.
- **Guarded orphan sweep.** `delete_orphan_points` (startup crash recovery, [ADR-48](../project/decisions.md)) deletes points whose `project_id` is not in the live set — but **refuses to run when the live set is empty**. An empty project table is indistinguishable from a misresolved state-DB path, and on the wrong DB "no live projects" would read as "delete the entire collection". The orphan count is logged before deletion, never silently.
- **Drift detection support ([ADR-49](../project/decisions.md)).** `count_project_points` (exact count) is the cheap drift gate the indexer compares against SQLite's expected chunk total; `per_file_point_counts` (payload-only scroll) is paid only after drift is confirmed.

## Key invariants

- One collection; project isolation is entirely by payload filter — deletes and counts are always project-filtered.
- `VectorStore` never constructs a `QdrantClient` and nothing outside it (bar `prefetch.py`) builds a `models.Document`. Both are CI-enforced greps: who owns the client objects *is* the concurrency design, and a client minted anywhere else would run a search outside the pool and outside its lock.
- `search` returns span dicts with `chunk_id`; chunk `text` is returned to core only when `with_text=True` (reranker path) and is stripped before results leave core.
- `get_chunk(chunk_id)` returns the *indexed* snapshot, not ground truth; malformed ids map to "unknown" identically for remote (HTTP 400) and in-process (ValueError) Qdrant, keeping the contract transport-independent.
- No server-side `path_prefix` filter is exposed: `MatchText` needs a full-text payload index that a real server would require, even though in-memory mode happens to accept a keyword index — a filter that only works in tests does not ship.
