# Design decisions (ADRs)

Every design decision in Noesis carries a recorded rationale — the house rule is **"no rationale, no merge."** Decisions live in the devlog's `decisions` table and in the decision log of [`architecture-docs/code-indexer-expanded-architecture.md`](https://github.com/behl1anmol/Noesis/blob/main/architecture-docs/code-indexer-expanded-architecture.md) (Appendix A), which holds the full chosen/reason/alternatives table. This page is the index; ADRs 1–18 belong to the approved baseline documents.

| # | Title | Decision (one line) |
|---|---|---|
| 19 | Reranker sequencing | Build after the M3 evaluation gate; default-on only on a measured NDCG@10 win |
| 20 | Reranker concurrency | Second dedicated single-thread worker, separate from the embedder's |
| 21 | Structural search substrate | Live filesystem via ast-grep-py; the index is bypassed entirely |
| 22 | ast-grep capability exposure | Search only — the rewrite capability is never exposed |
| 23 | Git integration | subprocess git CLI; candidates only shrink the hash set; hash remains truth |
| 24 | Embedder pluggability | Tiny async Protocol encapsulating dim/model_id/prefix; local implementations only |
| 25 | **Hosted embedding: rejected** | No remote embedding of any kind; no HTTP client in `core/`; CI-grep enforced |
| 26 | MCP v2 handling | Scheduled migration checkpoint; FastMCP drives the `mcp` version range until then |
| 27 | Lessons & memories storage | Hybrid: SQLite as system of record + committed rendered `LESSONS.md`; 15-lesson cap |
| 28 | `.gitignore` matching | `pathspec` runtime dep — canonical gitwildmatch, not a hand-rolled matcher |
| 29 | M1 API surface | None — M1 is core-library only; HTTP lands with M2 |
| 30 | Grammar assets | Keep tree-sitter-language-pack 1.12.x + install-time prefetch of all grammars |
| 31 | Generated-lockfile skip-list | Discovery skips committed lockfiles (uv.lock alone was 53% of this repo's chunks) |
| 32 | BM25 TF encoding | `qdrant-client[fastembed]` client-side TF + server-side `Modifier.IDF`; RRF k is server-fixed |
| 33 | Reranker model-loading boundary | New `core/reranker.py`; `sentence_transformers` allowed in exactly two modules |
| 34 | Rerank request-flag default | `rerank` defaults to `reranker.enabled`; disabled reranker returns `reranked:false`, not an error |
| 35 | **M4 gate: reranker default-off** | Quality passed (+0.106 NDCG@10), latency failed (12.2 s p50 on T4) — ships opt-in |
| 36 | `chunk_id` in search hits | Every hit exposes its deterministic point id so `get_chunk` is discoverable |
| 37 | Git fast-path specifics | Anchor validity = `merge-base --is-ancestor`; `-z` parsing; directory entries match as prefixes |
| 38 | Roadmap reorder | Dashboard + watcher promoted to M8; MCP v2 checkpoint renumbered to M9 |
| 39 | M8 runtime deps | jinja2 3.1.6 + watchdog 6.0.0, decision row recorded before install |
| 40 | M8 scope expansion | Per-project watch/auto-reindex (default off), persisted device setting, metadata-only `query_log` |
| 41 | Per-file error capture | Runs continue past per-file failures; `run_file_errors` + `files_failed` surface them |
| 42 | Dashboard registration | Register-only vs register+index split; per-project index config; browse/preview endpoints |
| 43 | Dashboard project deletion | Cancel run → unwatch → delete points by filter → SQLite child-first; source files untouched |
| 44 | Anchored db_path + config lookup | Default DB at `$XDG_DATA_HOME/noesis/noesis.sqlite`; no cwd-relative resolution |
| 45 | Watcher polling fallback | inotify-blind mounts (9p/cifs/nfs/…) get a polling observer, detected via `/proc/mounts` |
| 46 | Polling walk pruning | `PollingObserverVFS` prunes excluded dirs at the walk source (~350 s → ~0.6 s per interval) |
| 47 | Cancellation vs in-flight upsert | `execute_run` shields the in-flight upsert and awaits it on cancel — no orphaned points |
| 48 | Startup orphan-point sweep | Delete points of dead projects at startup; refused when the project table is empty |
| 49 | SQLite↔Qdrant drift self-heal | Count gate per run → per-file scroll → re-embed drifted files + prune orphans |
| 50 | Documentation stack | MkDocs Material on GitHub Pages via Actions: static mkdocstrings (griffe — no runtime deps at docs build), official Pages actions (no gh-pages branch), stdlib-urllib releases generator with a committed empty-state placeholder, dev-only `docs` dependency group |

See also the [risk register](risks.md) and [milestones](milestones.md).
