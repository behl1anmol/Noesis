# Git fast-path

`src/noesis/core/gitfast.py` asks git "what changed since the last indexed commit?" so a re-index of a large repo hashes a handful of candidates instead of every file. It is an optimization only — the SHA-256 hash remains the sole authority on change.

## Role

`compute_candidates(root, anchor)` returns the candidate changed set since the stored anchor commit, or `None`, which means: do the full hash-walk. The set is the union of:

- `git diff --name-status -z --no-renames <anchor>..HEAD` — committed changes, and
- `git status --porcelain=v1 -z --untracked-files=all --no-renames` — staged, unstaged, and untracked files.

Deleted paths are included: they are not on disk, so discovery never yields them and the hash-diff partition marks them deleted exactly as a full walk would.

## The correctness rule

**The fast path may only shrink the set of files that get hashed. It never marks a file unchanged on its own authority** — that stays with the SHA-256 comparison in `core/hashdiff.py`. `CandidatePathSet` even widens membership to ancestor directories, because git collapses an untracked nested repository to a single `dir/` entry and a submodule change to one gitlink path; treating those as directory prefixes means more files get hashed, never fewer.

## Fallback conditions

Any uncertainty returns `None` → silent full hash-walk (logged for the operator, invisible to the API caller):

| Condition | Detection |
|---|---|
| not a git repository / git binary absent | `git rev-parse` fails to spawn or exits non-zero |
| in-progress operation | any of `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, `BISECT_LOG`, `rebase-apply`, `rebase-merge` present in the git dir |
| detached HEAD | `git symbolic-ref -q HEAD` non-zero |
| unresolvable HEAD (e.g. no commits yet) | `git rev-parse HEAD` non-zero |
| anchor missing or not an ancestor of HEAD | `git merge-base --is-ancestor` non-zero ([ADR-37](../project/decisions.md)) — merely existing in the object store is not enough; a rebased-away commit lingers until gc and diffing against it would miss rewritten history |
| `git diff` / `git status` failure | non-zero exit |
| timeout | any git call exceeding 30 s (`_GIT_TIMEOUT_S`) |

## Design decisions

- **Subprocess CLI, not pygit2 ([ADR-23](../project/decisions.md)).** Zero Python dependencies, license-clean (GPLv2 unlinked), and the ~10 ms spawn cost is noise next to embedding. pygit2 remains the documented upgrade if profiling ever demands it.
- **`-z` and `--no-renames` parsing.** NUL-separated raw paths are immune to quoting/escaping; disabling rename detection makes a rename surface as delete-old + add-new, so both paths become candidates with a trivial parser regardless of the user's `diff.renames` config ([ADR-37](../project/decisions.md)).
- **Dirty-path carryover (H1).** The working-tree-dirty subset is persisted alongside the anchor (`projects.dirty_paths`), so a file dirty at run N is re-admitted as a candidate at run N+1 even if it was reverted to HEAD in between — the diff against the new anchor would otherwise never mention it.

## Anchor lifecycle

- `projects.last_indexed_commit` is written **only after a run completes cleanly** with a resolved HEAD; a failed run never advances it.
- `resolve_head` runs even on full-walk runs of a git worktree, so the *next* run has an anchor to fast-path from.
- **Watcher-scoped runs never advance the anchor**: a scoped run only re-examines the files the watcher saw, so advancing the anchor would let the next full pass skip files the watcher missed.

## Telemetry

Each run records `fast_path_used` (1/0) and `candidate_count` (`NULL` on full walks) in `index_runs`. Compared with `files_total`, this measures the optimization's actual value per run — the design principle that an optimization must be audited by its own telemetry rather than assumed.

## Key invariants

- A fast-path run and a full-walk run of the same tree produce identical index state — the fast path changes cost, never outcome.
- All fallbacks are silent to callers and logged for operators; the fast path can never surface an error to an API client.
- The candidate set only ever grows relative to git's answer (ancestor-directory matching), never shrinks.
