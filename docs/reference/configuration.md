# Configuration reference

Noesis runs with zero config — every setting in `src/noesis/core/config.py` has a working default. A `config.toml` overrides per key; unknown keys are ignored, invalid values (wrong type, non-positive numbers) fail startup with a clear error.

## File resolution order

1. Explicit path passed by the embedding process (internal).
2. `$NOESIS_CONFIG` — environment variable, for hosts that can't control their cwd (typical for MCP host entries). `~` is expanded.
3. `./config.toml` — deliberate dev override when running from a checkout.
4. `$XDG_CONFIG_HOME/noesis/config.toml` (default `~/.config/noesis/config.toml`).

If no file is found, all defaults apply.

## Top level

| Key | Type | Default | Effect |
|---|---|---|---|
| `db_path` | path | `~/.local/share/noesis/noesis.sqlite` (respects `$XDG_DATA_HOME`) | SQLite state DB location. Anchored, never cwd-relative, so the HTTP server and a stdio MCP server always share one DB regardless of where each was launched ([ADR-44](../project/decisions.md)). A relative path resolves against the config file's own directory, not the process cwd. |

## `[embedder]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `model` | str | `nomic-ai/CodeRankEmbed` | Dense embedding model id. Changing it triggers the full-re-embed rule — the system refuses to serve mixed-model results. |
| `dim` | int > 0 | `768` | Vector dimension; the Qdrant collection's dense size is read from here at creation time. |
| `batch_size` | int > 0 | `32` | Documents per embed batch during indexing. |
| `device` | str | unset | Unset → auto-detect (`cuda` → `mps` → `cpu`). A pin here wins over the dashboard device setting. |

## `[qdrant]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `url` | str | `http://127.0.0.1:6333` | Qdrant server URL (localhost only by design). |
| `collection` | str | `noesis_chunks` | Collection name — one shared collection, filtered per project. |
| `query_connections` | int > 0 | unset | Size of the search connection pool, and 1:1 the number of searches that may run at once ([ADR-83](../project/decisions.md), [ADR-84](../project/decisions.md)). Unset → derive from the machine: `clamp(available CPUs, 2, 8)`, the same "unset means auto" convention `embedder.device` uses. The CPU count is read through `sched_getaffinity`, so a cgroup-limited container sizes for its own allowance rather than for the host's cores. Set too high, the cost is **latency for no throughput** — past the knee, queries queue inside Qdrant instead of in the pool; set to `1`, it gives up about a third of the throughput. Each connection costs one Qdrant connection and one thread. See the note below on what the validator can and cannot enforce. |
| `query_queue_depth` | int > 0 | unset | How many searches may **wait** for a free connection before the service refuses instead of queueing without limit. Unset → `4 × query_connections`. Beyond `query_connections + query_queue_depth` admitted, REST `/search` answers **429** with `Retry-After` and MCP `search_code` raises a `ToolError` naming the counts and this knob. That rejection is the point: under many agents the honest failure is a fast, legible refusal, not a request that sits for an unbounded time. Raising it does not make the service faster — it only lengthens the wait a caller can be made to sit through before being told no. |

!!! note "What the `query_connections > 0` check does and does not enforce"
    Load rejects anything that is not a positive integer, and nothing more. It
    cannot reject a *wrong* value, because "wrong" is a property of the host:
    the right number is the one that saturates this machine's cores, and
    nothing at load time knows how many agents will search at once or how much
    CPU the co-resident Qdrant is taking.

    The derived rule comes from a measured curve — 4-CPU box, 16 concurrent
    searches, 5,000 points, hybrid, prefetch 50, against a live Qdrant 1.18.3:

    | Connections | Throughput | p95 |
    |---|---|---|
    | 1 | 258 q/s | 4.7 ms |
    | 2 | 353 q/s | 8.1 ms |
    | 4 | **384 q/s** | 16.0 ms |
    | 6 | 375 q/s | 23.7 ms |
    | 8 | 375 q/s | 33.3 ms |
    | 12 | 379 q/s | 48.4 ms |

    Throughput saturates *exactly* at the CPU count and then stays flat while
    p95 climbs threefold, because Qdrant is co-resident and competing for the
    same cores. So the rule clamps to the CPU count, and the floor of `2` is
    measured: one connection cannot overlap a query with the next one's round
    trip and gives up ~33% of throughput for it.

    The **ceiling of 8 is a judgment call, not a measurement** — no larger box
    was available to test on — which is precisely why this knob exists. On a
    many-core machine serving many agents, raising it is a reasonable thing to
    do; the derivation just refuses to guess that far on your behalf.

## `[reranker]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `model` | str | `BAAI/bge-reranker-v2-m3` | Cross-encoder model id. |
| `enabled` | bool | `false` | Kill switch **and** per-request default: `false` never loads the model and requests cannot opt in; `true` makes `rerank` default on with per-request opt-out ([ADR-34](../project/decisions.md)). Default-off is the measured M4 gate decision ([ADR-35](../project/decisions.md)). |
| `preload` | bool | `false` | `true` loads the ~568M model at startup instead of on first reranked request. |
| `candidates` | int > 0 | `50` | Fused candidates passed to the reranker per request, and the per-channel prefetch depth. Zero is rejected at load. It does **not** empty the result set — `retriever.search_code` clamps both values with `max(top_k, candidates)` — but it collapses the rerank pool and the RRF prefetch to `top_k`, so a hit ranked just outside one channel's `top_k` loses that channel's RRF contribution entirely and can drop out of the fused results. The cost is **silent recall loss**, with nothing in the response to indicate it. See the note below on what the validator can and cannot enforce. |
| `batch_size` | int > 0 | `16` | Pairs scored per cross-encoder batch. |
| `device` | str | unset | Same semantics as `embedder.device`. |

!!! note "What the `candidates > 0` check does and does not enforce"
    The harm above is really about `candidates` falling below `top_k`, and the
    validator cannot check that: `top_k` is a **per-request** value
    (`1..100`), while `candidates` is read once at startup. There is no load
    time at which the relationship is known.

    So `> 0` rejects the one value that is never defensible under any `top_k`,
    and nothing more. `candidates=1` is accepted and, at any `top_k > 1`,
    behaves exactly like `0` would — same clamp, same collapsed pool. If you
    are tuning this, the rule that matters is **keep `candidates` at or above
    the largest `top_k` you expect to serve**; the default of `50` sits
    comfortably above the default `top_k` of `10`.

## `[structural]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `max_results` | int > 0 | `100` | Cap on matches per structural query. A request may lower it, never raise it. |
| `timeout_s` | float > 0 | `10.0` | Wall-clock scan budget, counted from *after* file discovery completes (issue #43) — discovery is bounded separately by `.gitignore`/skip-list/size filters, not by this timer. On expiry the scan stops and returns partial results with `timed_out: true` — partial matches are still actionable to an iterating agent. |

## `[git]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `fast_path` | bool | `true` | `false` disables git candidate narrowing entirely — every run does a full hash-walk (the correctness baseline the fast path must match). |

## `[watcher]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `poll_interval_s` | float > 0 | `1.0` | Snapshot cadence of the polling observer, used only for watched roots on inotify-blind filesystems (9p/cifs/nfs/fuse — e.g. WSL2's `/mnt/c`). Natively watched roots never poll. |

## `[indexing]`

Recovery policy: how a project gets back to a full, anchored walk on its own.
Every automated trigger (the watcher's two paths, the dashboard's two) runs
*scoped*, and only a full run drains the working-tree-dirty set or advances the
git anchor — so without these a project that hits a snag stays there until a
human clicks Reindex.

These only govern *when* a full walk happens. Whether that walk is allowed to
record its position — and what it carries forward if it could not read
everything — is not configurable and is decided per run
([ADR-60](../project/decisions.md)).

One key here is not recovery policy at all: `max_concurrent_index_runs` bounds
how much indexing the *machine* does at once, whatever triggered it.

| Key | Type | Default | Effect |
|---|---|---|---|
| `promote_after_scoped_runs` | int ≥ 0 | `20` | Promote a scoped run to a full walk once this many scoped runs have started since the last **completed** full one. A failed full run does not reset the counter — it did not drain anything. `0` disables. |
| `promote_candidate_fraction` | float 0–1 | `0.25` | Promote when the effective candidate set reaches this fraction of the indexed file count. Past that point a scoped run costs about what the full walk costs while delivering none of its drains, since discovery stats and binary-sniffs every file either way. `0` disables. The set is `pending`, plus the working-tree-dirty set **only when the last full walk completed without failures** — see below. |
| `unwalkable_quarantine_runs` | int ≥ 0 | `5` | Consecutive runs a directory must fail to be walked before the indexed paths it hides stop being re-queued for retry. `0` disables, restoring a permanently non-draining backlog. |
| `max_concurrent_index_runs` | int > 0 | `4` | Cap on index runs **across the whole machine**, not per project ([ADR-85](../project/decisions.md)) — one run per project was already guaranteed; this bounds how many *projects* may run at once. Enforced inside the same `BEGIN IMMEDIATE` transaction that claims a run, so it holds **across processes**: the documented deployment runs an HTTP server and a stdio MCP process against one state DB, and an in-process counter would bound one of them while the other proceeded unaware. At the cap nothing is silently queued — REST `/projects` and `/projects/{id}/reindex` answer **429** with `Retry-After`, MCP `reindex` raises a `ToolError` saying nothing was started, and the watcher defers and re-arms its quiet-period trigger. A project's own live run still reads as `already_running`, which is a truthful answer about work already happening. |

!!! info "Why the dirty set is conditional"
    A full walk that reported any failure — an unreadable directory, but
    equally an unreadable *file* — cannot empty the working-tree-dirty set. It
    drains to a **floor**: the paths it could not verify, which the next run
    re-derives identically while the fault is live
    ([ADR-60](../project/decisions.md)). Counting that floor made
    `promote_candidate_fraction` fire on every launch forever, on a number
    promotion is structurally incapable of moving. The set is therefore counted
    only while the last **full walk** was clean, which is when a promotion
    could take it to zero.

    Before ADR-60 the set did not drain at all on a faulted project, so the
    same guard was needed for a stronger reason. What changed is the size of
    the problem, not its shape — and the guard is worth keeping either way,
    since a broken subtree larger than this fraction would otherwise satisfy
    the threshold from its floor on every launch.

    "Full walk" and not "last run": a scoped run never hashes what is outside
    its candidate set, so a permanently unhashable file leaves scoped runs
    reporting clean while the fault is still live.

    One interaction worth knowing if you tune these: the condition is re-enabled
    by a later clean full walk, and full walks come from the other two triggers.
    Setting `promote_after_scoped_runs = 0` while leaving
    `promote_candidate_fraction` on therefore removes this trigger's own
    recovery path — a single transient failure during a full walk leaves the
    dirty half of it disabled until some other full run happens. At the shipped
    defaults it is self-healing, since the run-count trigger delivers a clean
    full walk and re-enables it.

!!! note "Quarantine never deletes anything"
    A quarantined directory's indexed content is **kept and stays searchable**.
    Quarantine bounds only the *retry* — it has no say in deletion, which
    still requires positive evidence that a file is gone. The run that finds
    the directory walkable again re-hashes everything under it, so an edit made
    while it was unreadable is picked up automatically. See
    [the dashboard reference](dashboard.md#unreadable-directories).

!!! note "`max_concurrent_index_runs = 4` is a judgment call, not a measurement"
    Concurrent runs serialize on the single embedder worker
    ([ADR-20](../project/decisions.md)) whichever way this is set, so a larger
    number buys no indexing throughput — it only multiplies simultaneous file
    walks, open descriptors and executor pressure. `1` would be a visible
    behaviour change in the other direction, since today a second project need
    not wait behind a long run. Four sits between those two, and no measurement
    picked it; raise it if your machine says otherwise.

## `[server]`

| Key | Type | Default | Effect |
|---|---|---|---|
| `port` | int > 0 | `8000` | Port the HTTP service listens on, **and** the port `python -m noesis.mcp --shared` looks for a shared server on ([ADR-86](../project/decisions.md)). Both sides read this one value, so the shim never has to discover an endpoint — an earlier draft published one and promptly proxied an operator who asked for port 8199 to a live server on 8123. The port is configured; liveness is probed. `--port` on the shim overrides it for that process. |

!!! note "There is no `host` key, deliberately"
    The bind address is fixed at `127.0.0.1` and is not configurable. A host
    knob would put a wildcard bind one config edit away, and nothing else in
    the design survives that: the service has no authentication because it is
    not reachable from off the machine
    ([ADR-25](../project/decisions.md)).

## Environment variables

| Variable | Effect |
|---|---|
| `NOESIS_CONFIG` | Explicit config-file path (resolution step 2). |
| `FASTEMBED_CACHE_PATH` | Where fastembed caches the BM25 assets. Set automatically by prefetch and the service to `$XDG_CACHE_HOME/noesis/fastembed` so runtime stays offline — override only if you know why. |

## Full example

```toml
db_path = "~/.local/share/noesis/noesis.sqlite"

[embedder]
model = "nomic-ai/CodeRankEmbed"
dim = 768
batch_size = 32
# device = "cuda"

[qdrant]
url = "http://127.0.0.1:6333"
collection = "noesis_chunks"
# query_connections / query_queue_depth: omit for auto —
# clamp(available CPUs, 2, 8) connections, and 4x that many waiters.
# query_connections = 4
# query_queue_depth = 16

[reranker]
model = "BAAI/bge-reranker-v2-m3"
enabled = false
preload = false
candidates = 50
batch_size = 16

[structural]
max_results = 100
timeout_s = 10.0

[git]
fast_path = true

[watcher]
poll_interval_s = 1.0

[indexing]
promote_after_scoped_runs = 20
promote_candidate_fraction = 0.25
unwalkable_quarantine_runs = 5
max_concurrent_index_runs = 4

[server]
port = 8000
```
