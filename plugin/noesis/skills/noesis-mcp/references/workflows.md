# Noesis task recipes

Concrete, copyable sequences for the common jobs. All assume a `project_id` from
`list_projects`.

## 1. Find where something is implemented, then read it

The default retrieval task. Describe intent, confirm on the live file.

```
search_code(
  query = "where the hybrid search fuses dense and sparse channels",
  project_id = PID,
  top_k = 10,
)
→ take the top 1–3 hits
→ Read tool on each hit's file_path at start_line..end_line   (live file!)
```

Tips:
- Phrase the query as a description of behavior, not a symbol name. Hybrid
  retrieval still catches exact identifiers via the sparse channel.
- If results look off-topic, narrow with `language=` or bump `top_k`.
- Only fall back to `channel="sparse"` when you want a pure keyword/BM25 match
  (e.g. hunting an exact rare string), or `channel="dense"` for pure semantic.

## 2. Safe refactor — enumerate every call site before editing

Search finds *where to look*; structural search finds *every* site exactly, so a
rename/signature change misses nothing.

```
search_code("definition of foo and its role", PID)     → locate the definition
structural_search(
  pattern = "foo($$$ARGS)",
  language = "python",
  project_id = PID,
)                                                       → every call site, live
→ if truncated:true, raise max_results or narrow with paths, and re-run
→ edit each site; re-run the structural_search to confirm zero remain
```

Structural search reads the live filesystem, so it stays correct as you edit —
re-run it as a completeness check after your changes.

## 3. Pull a hit's full text

Search `snippet`s can be truncated. For the complete indexed span:

```
hits = search_code(...)                     → grab hits[0].chunk_id
get_chunk(hits[0].chunk_id)                 → full stored text + metadata
```
Remember `get_chunk` returns the *indexed snapshot*. If it disagrees with the
live file, the index is stale — reindex (recipe 5) or just trust the live file.

## 4. Structural patterns cheat-sheet

| Goal | `language` | `pattern` |
|------|-----------|-----------|
| All function defs | `python` | `def $NAME($$$ARGS): $$$BODY` |
| Calls to a function | `python` | `my_func($$$ARGS)` |
| Method calls on an object | `python` | `$OBJ.save($$$A)` |
| Decorator usage | `python` | `@$DEC` |
| Console logging | `typescript` | `console.log($$$A)` |
| React hook usage | `typescript` | `useEffect($$$A)` |

`$NAME` binds one node; `$$$ARGS` binds zero or more. Scope with `paths=["src/"]`.
Unsupported language → the error lists the languages that *are* supported.

## 5. Refresh a stale index after files change

```
reindex(PID)                    → {run_id, status:"accepted"}   (async, incremental)
loop:
  get_index_status(PID)         → until status == "done"
                                  (on "failed", read .error and report it)
```
Don't block on the run — poll `get_index_status`. `structural_search` never needs
this; it already reads live files. Only `search_code`/`get_chunk` see stale data.

## 6. First-contact checklist (unfamiliar setup)

```
1. list_projects
   ├─ empty        → register (scripts/register_project.py … --wait), then continue
   └─ has the repo → note its id
2. get_index_status(PID)
   ├─ never_indexed / failed → reindex or re-register; wait for "done"
   └─ done                    → proceed
3. search_code / structural_search as the task needs
```
If step 1 errors because no `noesis:` tools exist at all, the service/connection
is down — run `scripts/healthcheck.py` and see troubleshooting.md.
