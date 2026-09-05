# `tests/perf` — cold-start harness

Reproduces the first-use stall: on a machine where `python -m noesis.prefetch`
was never run, the embedding weights are downloaded *inside* the first
user-facing MCP tool call, because `LocalSTEmbedder` loads its model lazily on
the first job its worker thread receives (`src/noesis/core/embedder.py`).

This directory holds the harness only. It does not fix anything, and nothing
here runs in CI.

## Run it

```bash
uv run python tests/perf/cold_start_harness.py                    # cold vs warm
uv run python tests/perf/cold_start_harness.py --scenarios cold,warm,warm,warm
uv run python tests/perf/cold_start_harness.py --scenarios cold,warm,warm,warm,prefetched
uv run python tests/perf/cold_start_harness.py --sequence mcp
uv run python tests/perf/cold_start_harness.py --qdrant-url http://127.0.0.1:6333
```

**Repeat `warm`.** Three warm repeats on a quiet 4-core box landed within 0.6s
of each other (10.72–11.33s); two on the same box under load landed 6.6s apart,
which is wider than the download delta itself. With one warm sample the harness
has no noise floor and refuses to call any seconds delta signal; with several it
measures the delta against the *slowest* warm sample and claims it only when it
clears the spread. The byte columns need no repeats — a cache either grew or it
did not.

Reports land in `dev/perf/cold-start/report_latest.{json,md}` (gitignored, along
with the whole workspace). Each run rewrites them.

Expect a full `cold,warm,prefetched` run to download the model **twice** (once
for `cold`, once for `prefetched`) — the scenarios are deliberately independent.

The default corpus is `src/noesis/mcp`, deliberately small: indexing is not what
this harness measures, and embedding the whole `src/noesis` tree on CPU costs
about 20 minutes *per scenario*. Pass `--corpus src/noesis` when you want the
index phase to be representative rather than incidental.

## What it measures

One **workload** (the real first-use sequence over this repo's own source tree)
inside a **scenario** (a controlled asset-cache state), each in a fresh
subprocess, reporting per phase:

- **wall seconds** — machine- and link-specific, only comparable *within one run*
- **bytes that appeared in each asset cache** — the machine-independent invariant

Read the byte columns to decide *what* is fetched and *which call pays*. Read
the seconds only against another scenario measured on the same machine.

### Scenarios

| scenario | cache state | answers |
| --- | --- | --- |
| `cold` | every asset cache empty | what a fresh install that skipped prefetch pays |
| `warm` | whatever the previous scenario left | the control |
| `prefetched` | empty, then `noesis.prefetch --skip-reranker` runs and is timed | whether the documented remedy actually works |

`warm` cannot run first — it is defined relative to the scenario before it, and
measuring it against a workspace of unknown provenance is the stale-artifact
trap `dev/LESSONS.md` lesson 8 records.

### Sequences

| sequence | order | why |
| --- | --- | --- |
| `query-first` (default) | register → `search_code` → `reindex` → drain → `search_code` | isolates the download onto one call, nothing else queued on the embedder worker. The first search returns zero hits by construction; the latency is the measurement |
| `mcp` | `reindex` → `search_code` → drain → `search_code` | the real agent path: `reindex` returns a run_id immediately and indexing continues in the background, so the agent's next `search_code` is what blocks |

`mcp` overlaps a hybrid search with an index run on one `QdrantClient`, which
qdrant-client 1.18 does not survive. Two distinct failures were reproduced:

- `RuntimeError: dictionary changed size during iteration` from
  `qdrant_client/embed/model_embedder.py`, whose `_batch_accumulator` carries no
  lock. Client-side inference runs before dispatch, so this is **not** an
  embedded-store artifact — a real Qdrant server does not protect against it.
- `IndexError: index N is out of bounds` from
  `qdrant_client/local/local_collection.py`, reading the deleted mask while an
  upsert grows the points. Embedded store only, and the damage persists: the
  next *non-concurrent* search fails too.

Both are separate defects, not the one being measured. The harness records them
on the phase rather than aborting — but because of the second, **run `mcp` with
`--qdrant-url`**; its embedded-store numbers are untrustworthy past the first
search.

### Asset caches

All three are redirected into the workspace, so a run never touches the
operator's real `~/.cache` and never deletes anything outside its own directory.

| cache | env var | holds |
| --- | --- | --- |
| `hf` | `HF_HOME` | CodeRankEmbed / reranker weights |
| `fastembed` | `FASTEMBED_CACHE_PATH` | `Qdrant/bm25` tokenizer assets |
| `xdg` | `XDG_CACHE_HOME` | tree-sitter grammars (the language pack has no env var of its own) |

The Python environment is **not** rebuilt — this measures asset fetch, not
`uv sync`.

## What it found (2026-09-05, commit 0417a52)

Linux, 4-core CPU, no CUDA, datacenter link, `--corpus src/noesis/mcp`,
`--scenarios cold,warm,warm,warm,prefetched`:

| phase | cold | warm (3 repeats) | prefetched |
| --- | --- | --- | --- |
| `import` | 1.54s | 1.44–1.50s | 1.52s |
| `startup` (`build_runtime_context`) | 0.01s | 0.01s | 0.01s |
| **`first_search`** | **20.75s, 548.1 MB** | 10.72–11.33s, 0 MB | 12.89s, 0 MB |
| `reindex_call` | 0.00s | 0.00s | 0.00s |
| `index_drain` | 8.87s, 19.3 MB | 6.72–7.00s, 0 MB | 6.76s, 0 MB |
| `second_search` | 0.10s | 0.10–0.12s | 0.10s |
| `prefetch` step | — | — | 33.87s, 614.7 MB |

Three things this makes hard to argue with:

1. **`startup` loads nothing.** `build_runtime_context` constructs the embedder
   but never forces a load, so the whole cost lands in a tool call.
2. **The first query is ~200× the second** (20.75s vs 0.10s), and 548 MB of that
   is fetched *inside* it. `reindex_call` returns in 1 ms, so on the real MCP
   path the wait is charged to `search_code`, not to indexing.
3. **Prefetching does not fix the stall, only the download.** The `prefetched`
   scenario still spent 12.89s in `first_search` with 0 bytes fetched. That
   residue is model construction, not query work: the same process's
   `second_search` — a real hybrid query with an embed, a BM25 encode and a
   store round-trip — took 0.10s. So roughly 10.6s of the warm 10.72s is
   `SentenceTransformer(...)` building the model, and no amount of prefetching
   removes it. Only warming the model does.

### The same experiment on the real MCP call order

`--sequence mcp --scenarios cold,warm` (one warm sample, so no seconds claim):

| phase | cold | warm |
| --- | --- | --- |
| `reindex_call` | 0.00s | 0.00s |
| **`first_search`** | **33.04s, 567.4 MB** | 16.87s, 0 MB |
| `second_search` | 0.23s | 0.14s |

This is the reported symptom exactly: `reindex` returns in under a millisecond,
and the agent's very next `search_code` blocks for 33s while **all 567 MB** —
weights, grammars and BM25 assets together — is fetched inside that one tool
call. In `query-first` the grammar fetch is charged to `index_drain` instead;
on the real call order it lands in the query too, because the background index
run is chunking while the query waits.

Note the byte asymmetry: `prefetch` fetches 614.7 MB where lazy loading fetched
567.4 MB, because it pulls grammars for every canonical language (66.5 MB) while
indexing Python only pulled 19.3 MB.

Seconds here are datacenter-link seconds. The transferable figure is the
548.1 MB: at 25 Mbit/s that same `first_search` carries roughly three minutes of
download instead of nine seconds.

## Mechanics tests

`test_cold_start_harness.py` runs in the default suite (no model, no network,
no Qdrant) and pins the things that would let the harness report a plausible
lie: symlink-free byte accounting, the delete guard, phase attribution, and
scenario ordering.

```bash
uv run pytest tests/perf -q
```
