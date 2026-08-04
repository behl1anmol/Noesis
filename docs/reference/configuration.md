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
| `timeout_s` | float > 0 | `10.0` | Wall-clock scan budget. On expiry the scan stops and returns partial results with `timed_out: true` — partial matches are still actionable to an iterating agent. |

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

| Key | Type | Default | Effect |
|---|---|---|---|
| `promote_after_scoped_runs` | int ≥ 0 | `20` | Promote a scoped run to a full walk once this many scoped runs have started since the last **completed** full one. A failed full run does not reset the counter — it did not drain anything. `0` disables. |
| `promote_candidate_fraction` | float 0–1 | `0.25` | Promote when the effective candidate set reaches this fraction of the indexed file count. Past that point a scoped run costs about what the full walk costs while delivering none of its drains, since discovery stats and binary-sniffs every file either way. `0` disables. The set is `pending`, plus the working-tree-dirty set **only when the last full walk completed without failures** — see below. |
| `unwalkable_quarantine_runs` | int ≥ 0 | `5` | Consecutive runs a directory must fail to be walked before the indexed paths it hides stop being re-queued for retry. `0` disables, restoring a permanently non-draining backlog. |

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
```
