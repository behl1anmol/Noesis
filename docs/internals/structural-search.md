# Structural search

`src/noesis/core/structural.py` answers queries like "every `def` with these arguments" by pattern-matching ASTs of **live files** with ast-grep — the index is never consulted.

## Role

`structural_search(conn, project_id, pattern, language, *, paths, max_results, settings)` scans the project's discovery-filtered files of the requested language and returns:

```python
class StructuralResult(TypedDict):
    matches: list[StructuralMatch]   # file_path, start_line, end_line (1-based), matched_text, meta_vars
    scanned_files: int
    truncated: bool   # stopped at max_results; more matches may exist
    timed_out: bool   # stopped at the timeout budget; matches are partial
```

Pattern syntax is ast-grep's: `$NAME` captures a single node, `$$$ARGS` a node list. Example — every Python function definition:

```
def $NAME($$$ARGS): $$$BODY
```

`meta_vars` in each match maps capture names to text (`$X` → `str | None`, `$$$X` → `list[str]`); capture names are extracted from the pattern text because ast-grep-py exposes captures only by name.

## The live-filesystem contract

This is Finding 5 of the architecture review: structural facts are **not** extracted at index time into Qdrant payloads. The scan reads the filesystem as it is right now, so results are never stale *by construction* — consistent with the system's core contract that the file is ground truth. Consequences:

- Bypasses Qdrant, SQLite chunk state, and the embedder entirely; shares almost no code with the retrieval pipeline.
- Reuses exactly two core surfaces: the project registry (`project_id` → `root_path`) and the discovery filter chain — so structural search **can never surface a file that indexing would have excluded** (`.gitignore`, secret skip-list, size caps, per-project scope).
- **Rejected alternative — index-time extraction:** it would duplicate ast-grep's job, add a second staleness surface in the system's highest-risk area, bloat payloads, and constrain queries to whatever was pre-extracted. Live scanning is slower per query but always correct and unbounded in expressiveness.

## Design decisions

- **ast-grep-py in-process ([ADR-21](../project/decisions.md)).** `SgRoot(src, language).root().find_all(pattern=...)` per candidate file; the engine is self-contained Rust with its own grammars. Language identifiers differ from tree-sitter's, so `core/languages.py` keeps one explicit `LANGUAGE_MAP` from canonical names to both grammar names; unmapped languages (e.g. `toml`, `sql`) return a clean `unsupported_language` error.
- **Search only, never rewrite ([ADR-22](../project/decisions.md)).** ast-grep's rewrite capability is deliberately not exposed — a tool that mutates user source is a different risk class.
- **Typed errors.** `StructuralSearchError.error_type` is one of `unknown_project` / `unsupported_language` / `pattern_error` / `invalid_path`, so adapters map failures without parsing messages. Pattern errors carry ast-grep's own diagnostic — agents iterate on patterns, and that loop is kept cheap by probing the pattern against empty source before any file is read. Most malformed patterns simply match nothing; only patterns the engine refuses outright are errors.
- **Executor placement.** The scan is blocking I/O plus native compute, so it runs via `run_in_executor` on the **default thread pool** — never the embedder or reranker workers, which must not queue behind a scan.
- **Bounded cost.** `max_results` (request may lower the configured cap, never raise it; clamped to ≥ 1) stops the scan mid-file; `timeout_s` is a wall-clock budget checked per file — a single file can overrun by its own parse+match time, which discovery's 1 MiB size cap keeps small. Both terminations are reported explicitly (`truncated`, `timed_out`), never silent.
- **Path restrictions are validated.** `paths` entries must be project-relative without `..`; absolute or escaping paths raise `invalid_path` rather than silently scanning nothing.

## Key invariants

- Results reflect the filesystem at scan time — a file deleted mid-scan is skipped, not an error.
- The secret skip-list and all discovery filters apply identically to indexing and structural search: one filter chain, two consumers.
- Line numbers are 1-based inclusive, matching the search-hit span shape.
- No state is written anywhere; structural search is a pure read of the tree (plus a metadata-only telemetry row).
