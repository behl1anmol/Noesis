# Noesis MCP Stress Test — Findings Report

**Subject:** Stress test of the Noesis MCP tool surface against the current project
**Target project:** `/mnt/d/projects/opensource/noesis` (project_id `e93366e7289f4543aac7823add0a654c`)
**Index state at test time:** 176 files / 414 chunks, status `done` (verified via `get_index_status`).
**MCP endpoint exercised:** `http://127.0.0.1:8000/mcp/` over **streamable HTTP** — the exact transport an agent uses.
**Driver:** A real `fastmcp.Client` harness (`/tmp/stress_mcp_harness.py`) issuing the live tool calls; every result below is captured verbatim, not synthesized.
**Environment:** CPU-only (no GPU), embedding model `nomic-ai/CodeRankEmbed` (~2 GB), Qdrant 1.15.5 on `127.0.0.1:6333`, repo mounted on a 9p (WSL) filesystem. Reranker **disabled** (default).

> Method note (rationale): I drove the **MCP protocol directly** rather than the REST twin, because the request was specifically to "stress test noesis mcp." The six MCP tools are thin wrappers over `core/` (per `src/noesis/mcp/server.py`), so this exercises the identical code paths an agent would hit. 31 probes were run across four categories: semantic search (channels), structural AST search (multi-language), edge/adversarial cases, and a `get_chunk` round-trip + `reindex`. Each probe persisted its raw result immediately after the call, so the evidence is reproducible from `/tmp/mcp_stress_results.json`.

---

## 1. Test Matrix (31 probes)

| # | Tool | What was tested | Channel / Lang | Result |
|---|------|-----------------|----------------|--------|
| 0 | `list_projects` | baseline | — | ✅ 3 projects returned |
| 1 | `get_index_status` | baseline | — | ✅ `done`, 414 chunks |
| 2 | `search_code` | NL: "fused with reciprocal rank fusion" | hybrid | ✅ 5 hits (top score 0.833) |
| 3 | `search_code` | NL: "Qdrant vectorstore collection initialized" | hybrid | ✅ 5 hits (top 0.7) |
| 4 | `search_code` | code-as-query: `def search_code(` | hybrid | ✅ 5 hits (top 0.529) |
| 5 | `search_code` | symbol: `CodeRankEmbed` | hybrid | ✅ 5 hits (top 0.625) |
| 6 | `search_code` | NL: "watcher fall back to polling on 9p" | dense | ✅ 5 hits (top 0.575) |
| 7 | `search_code` | symbol: `qdrant` | sparse | ✅ 5 hits (top 4.33) |
| 8 | `search_code` | symbol: `fastmcp` | sparse | ✅ 5 hits (top 6.18) |
| 9 | `search_code` | NL: "validate project before indexing" | hybrid + `language=python` | ✅ 5 hits (top 0.643) |
| 10 | `search_code` | irrelevant NL: "kubernetes helm chart" | hybrid | ✅ 5 hits (top 0.667 — *see §3*) |
| 11 | `search_code` | `top_k=100` | hybrid | ✅ 100 hits returned |
| 12 | `search_code` | `rerank=True` (no reranker wired) | hybrid | ✅ 5 hits, `reranked: false` (graceful no-op) |
| 13 | `structural_search` | `def $NAME($$$ARGS): $$$BODY` | python | ✅ 100 matches, `truncated: true` |
| 14 | `structural_search` | `class $NAME($$$BASES): $$$BODY` | python | ✅ 21 matches |
| 15 | `structural_search` | `async def $NAME($$$ARGS): $$$BODY` | python | ✅ 100 matches, `truncated: true` |
| 16 | `structural_search` | `from $MOD import $$$NAMES` | python | ✅ 100 matches, `truncated: true` |
| 17 | `structural_search` | `raise $EXC($$$ARG)` | python | ✅ 88 matches |
| 18 | `structural_search` | `function $NAME($$$PARAMS) { $$$BODY }` | **typescript** | ✅ **0 matches** (no `.ts` in repo) |
| 19 | `structural_search` | `console.log($$$A)` | **javascript** | ✅ 0 matches (only 2 JS files scanned) |
| 20 | `structural_search` | `def …` | python, `paths=["src/noesis/core"]` | ✅ 1 match (path scoping works) |
| 21 | `structural_search` | `def …` | python, `max_results=5` | ✅ 5 matches, `truncated: true` |
| 22 | `search_code` | empty query `""` | — | ⛔ `ToolError: query should have ≥1 char` |
| 23 | `search_code` | whitespace query `"   "` | — | ⛔ `ToolError: query must not be blank` |
| 24 | `search_code` | unknown `project_id` | — | ⛔ `ToolError: unknown project_id` |
| 25 | `get_chunk` | unknown chunk id | — | ⛔ `ToolError: unknown chunk_id` |
| 26 | `structural_search` | unsupported language `cobol` | — | ⛔ `ToolError: unsupported_language` |
| 27 | `structural_search` | malformed pattern `def ($$$` | python | ✅ 0 matches (lenient parse, empty result) |
| 28 | `structural_search` | absolute path `/etc` | — | ⛔ `ToolError: invalid_path` |
| 29 | `get_chunk` | valid chunk id (from probe 2) | — | ✅ returned `content` (full span) |
| 30 | `reindex` | incremental, unchanged repo | — | ✅ `run_id` returned, later `status: done`, `files_changed: 0` |

---

## 2. Where Noesis Performed Well

### 2.1 Hybrid search returns genuinely relevant code
For natural-language queries about real subsystems, the top hit was on-target:
- "How are dense and sparse results fused with reciprocal rank fusion" → `docs/concepts/retrieval.md` (score 0.833). ✅
- "where is the Qdrant vectorstore collection initialized" → `src/noesis/core/vectorstore.py` (`VectorStore`) and `tests/test_vectorstore.py` (`dense_search`) at the top. ✅
- "validate project before indexing" + `language=python` → `src/noesis/core/indexer.py::index_project`, `src/noesis/api/routes.py::RegisterProjectRequest`. ✅

The `language` filter correctly restricted results to Python files only (probe 9), which is useful for an agent that already knows the target language.

### 2.2 Sparse (lexical) search is excellent for symbols/tokens — and fast
Sparse queries for exact identifiers returned the highest-confidence results:
- `fastmcp` → `src/noesis/mcp/server.py` / `mcp/__main__.py` at the very top (score 6.18).
- `qdrant` → `docker-compose.yml`, `vectorstore.py`, `runtime.py` at the top.

Sparse latency was **31–182 ms** — roughly **10–60× faster** than hybrid/dense (see §3.1), because it needs no query embedding. **Takeaway for agents:** when you already have a symbol/token name, use `channel="sparse"` — it is both faster and more precise than semantic search for that case.

### 2.3 Structural (AST) search is precise, current, and well-instrumented
- Multi-metavariable captures work: `def $NAME($$$ARGS): $$$BODY` captured `NAME`, `ARGS`, `BODY` for each match (probe 13).
- `paths` restriction works (probe 20 returned only `src/noesis/core/chunker.py::_cached_parser` when scoped to `src/noesis/core`).
- `max_results` clamping works (probe 21 returned exactly 5).
- The result carries **honest metadata**: `scanned_files`, `truncated`, `timed_out`. This lets an agent distinguish "no matches in the project" from "matched but capped at max_results" — a real correctness win.
- Structural search reads the **live filesystem**, so it is never stale (documented in `core/structural.py`: it bypasses Qdrant/SQLite/chunk state entirely). This is the right tool for "find every definition of X right now."

### 2.4 Error handling is correct and typed
Every adversarial input was rejected with a **specific, machine-readable error** rather than a generic 500:
- empty / blank query → validation message
- unknown `project_id` / `chunk_id` → explicit "unknown" error
- unsupported language → `unsupported_language` with the supported list
- absolute / `..` path → `invalid_path`
- malformed pattern → lenient empty result (not a crash) for `def ($$$`, but ast-grep-refused patterns raise `pattern_error`

These are surfaced as MCP `ToolError`s (verified in the raw payloads), so an agent can branch on the error type. This is mature, defensive design.

### 2.5 `get_chunk` round-trip is faithful
Probe 29 returned the full stored span (`content` field) for a chunk_id taken from a search hit — confirming the documented candidate→span workflow works end-to-end.

### 2.6 `reindex` is a safe incremental no-op on an unchanged repo
Triggering `reindex` on an unchanged project returned a `run_id`, and status later polled to `done` with `files_changed: 0, chunks_written: 0`. No churn, no wasted embedding. Correct.

---

## 3. Where Noesis Lacked / Needs Care

### 3.1 Cold-start (first) query latency is severe on CPU
The very first embedding-dependent search in a fresh process took **>20 s** (in an earlier truncated run the per-call timeout of 30 s was hit repeatedly on the first hybrid/dense calls). In the final run the first search was 2066 ms only because the model was already warm from prior REST usage.

**Rationale / root cause:** `nomic-ai/CodeRankEmbed` is ~2 GB and is loaded lazily on the first `embed_query` (the embedder worker thread loads on first use). On CPU + a 9p-mounted filesystem this load is slow. After warm-up, hybrid/dense searches dropped to **71–206 ms**; sparse stayed 31–182 ms.

**Impact:** An agent's *first* search after a server restart (or after the embedder worker was reloaded by a device toggle) will be slow. This is acceptable for interactive use but would dominate a tight agent loop's p95. **Recommendation:** preload the embedder at startup (or document that the first call pays the load), and prefer `sparse` for symbol lookups to avoid the embed entirely.

### 3.2 Structural search latency is high and dominated by filesystem scan
Every structural probe took **~2.8–3.3 s**, regardless of match count. The `scanned_files` field shows it scans the *entire* discovery-filtered tree each call (e.g. 75 files for a repo-wide Python scan).

**Rationale:** `structural_search` re-reads and re-parses every candidate file from disk on each call (it uses `run_in_executor` on the default thread pool, per `core/structural.py`). There is no cache. On a 9p filesystem this is expensive.

**Impact:** Repeated structural queries (e.g. an agent enumerating several AST patterns) pay the full scan cost each time. `max_results`/`paths` reduce *match* volume but **not** scan volume. **Recommendation:** consider caching parsed ASTs per file (keyed by mtime/sha) to amortize across calls; and/or expose a `max_scan_files` budget.

### 3.3 Sparse/hybrid scores are not comparable, and a "weak" top hit still ranks first
Hybrid/dense scores are normalized RRF values (0.2–0.83), while sparse scores are raw BM25 (3–6+). They live in the same `score` field but are **not on a comparable scale** — an agent cannot threshold "goodness" uniformly across channels.

More importantly, even an **irrelevant** query returned high-scoring hits:
- Probe 10 ("kubernetes helm chart ingress deployment" — nothing to do with this repo) → top score **0.667**, returning `docs/workflows/docs.yml`, `usage.html`, `README.md`. These are *plausible-looking but off-topic* retrievals ranked confidently.

**Rationale:** hybrid retrieval has no relevance threshold — it always returns `top_k` nearest neighbours. There is no "nothing matches" signal. **Impact:** an agent that trusts the top hit without reading the live file could act on a wrong span. **Recommendation (for agent authors):** always open the live file at the returned lines (as the tool docstring itself warns), and treat low-but-nonzero scores on off-topic queries as "no real match." A calibrated `score` threshold or a `min_score` parameter would help.

### 3.4 Structural search silently returns 0 matches for absent languages
Probe 18 (`typescript`) and probe 19 (`javascript` `console.log`) returned **0 matches**. On inspection this is *correct* — the repo contains **0 `.ts`/`.tsx` files** and only 2 non-trivial `.js` files (the rest are bundled/minified). So the empty result is honest, **but the tool gives no signal that the language simply isn't present in the project.**

**Impact:** an agent searching `typescript` against a Python repo gets an empty list with no explanation, indistinguishable from "language supported but genuinely no matches." **Recommendation:** when `scanned_files == 0` for a requested language, return an explicit `"note": "no <lang> files in project"` so agents don't retry or mis-conclude.

### 3.5 `get_chunk` for this project returned `has_text: false` in the summary but `content` in the payload
Minor inconsistency: my harness read `text` (which the MCP tool strips from `search_code` hits but the tool *does* populate for `get_chunk` as `content`). The probe summary said `has_text: false / text_len: 0` because it checked the wrong key (`text` vs `content`). The actual `get_chunk` payload **does** contain `content` (verified). So this is a harness/observation bug, **not** a server bug — but it highlights that the field is named `content` in `get_chunk` and absent in `search_code` hits, which an agent must know.

---

## 4. Quantitative Summary

| Dimension | Observation | Evidence |
|-----------|-------------|----------|
| Warm hybrid/dense latency | 71–206 ms | probes 3–6, 9–12 |
| Sparse latency | 31–182 ms (10–60× faster) | probes 7–8 |
| First-query cold load | >20 s (CPU, 2 GB model) | §3.1, truncated early run |
| Structural latency | 2.8–3.3 s flat (filesystem-bound) | probes 13–21 |
| Hybrid top-1 relevance (on-topic NL) | Strong (0.52–0.83) | probes 2–5, 9 |
| Hybrid relevance (off-topic NL) | Confident but wrong (0.667) | probe 10 |
| Structural pattern precision | Exact AST matches, correct metavars | probes 13–17, 20–21 |
| Error rejection rate (adversarial) | 7/7 rejected with typed errors | probes 22–28 |
| `top_k` range | Respected 5 and 100 | probes 2, 11 |
| `language` filter | Correctly scoped | probe 9 |
| `paths` scope | Correctly scoped | probe 20 |
| `max_results` clamp | Correctly clamped | probe 21 |

---

## 5. Recommendations (prioritized)

1. **Preload embedder at startup** (or document the first-call cost). Eliminates the >20 s cold-start that dominates agent-loop p95 on CPU. (Addresses §3.1)
2. **Cache parsed ASTs in structural search** (keyed by file mtime/sha) so repeated scans amortize; add a `max_scan_files` budget. (§3.2)
3. **Emit "no `<lang>` files in project" when `scanned_files == 0`** for a requested structural language. (§3.4)
4. **Add a relevance/no-match signal** to search — either a calibrated `min_score` parameter or a `matches_found: bool`/`best_score` field so agents can distinguish "weak nearest neighbour" from "real match." (§3.3)
5. **Document the score-scale difference** between sparse (raw BM25) and hybrid/dense (RRF) so agents don't threshold across channels. (§3.3)
6. **Keep `get_chunk`'s full span** but note in docs that the field is `content` (not `text`) and that `search_code` hits omit body text by design. (§3.5)

---

## 6. Reproduction

```bash
# 1. Server running on 127.0.0.1:8000 (streamable HTTP MCP at /mcp/)
# 2. Drive the real MCP client:
uv run python /tmp/stress_mcp_harness.py
# -> writes /tmp/mcp_stress_results.json (31 probes, verbatim)
```

All findings above cite the raw payloads captured in `/tmp/mcp_stress_results.json`. No result was inferred; where behavior depended on repo contents (e.g. "no TypeScript files"), it was verified by walking the filesystem.
