"""Evaluation harness (doc §6.2; M3 gate, extended for the M4 gate).

Loads the human-labeled golden set (``tests/eval/golden.yaml``), runs a
search function per query, and reports Recall@5, Recall@10 and NDCG@10 per
query category (nl / symbol / structural) plus overall. The gate compares the
run against the living reference at ``tests/eval/baselines/reference.json``
(ADR-65) — numbers or it didn't happen. ``baselines/m2_dense.json`` is a
frozen historical record of what M2 measured and is read by no code path;
this docstring named it as the gate's baseline for two milestones after that
stopped being true (PR #42 review round 4).

M4 (§3.8): every row also carries search latency p50/p95 in milliseconds —
wall time of the full search call per query, nearest-rank percentiles — so
the reranker's cost is visible next to its NDCG gain. Latency is measured,
never compared by the quality-delta view: quality gates and latency budgets
are separate stakeholder decisions.

Labels are content-addressed (ADR-64, issue #38): a relevant item names an
``anchor`` — a substring occurring exactly once in the labeled file — which
``load_golden`` resolves to a line range against the tree being measured,
optionally widened by an ``anchor_end``. Stored line numbers were tried first
and rotted: 22 of 46 labels had drifted off their own ranges by 2026-08, and
because ``matches`` requires span overlap, each rotted label silently forced
its query to zero. An anchor that stops resolving raises instead.

Scoring rules (deliberate, stated so the numbers are reproducible):

- A result matches a relevant item iff the ``file_path`` is equal and the
  result span ``[start_line, end_line]`` overlaps the item's resolved range.
- Results are grouped by ``file_path`` before scoring, in first-appearance
  order — several chunks of one file count as one retrieval and occupy one
  rank slot, but every retrieved chunk of that file stays available for
  matching (ADR-67). Collapsing to the best-ranked chunk alone discarded
  correct answers whenever a non-matching chunk of the right file outranked
  the matching one.
- Recall@k = fraction of a query's relevant items matched, using only the
  chunks in the top k rank slots, averaged over queries. Chunks are assigned
  to items by maximum bipartite matching (ADR-68), each chunk crediting at
  most one item: first-fit assignment could spend a chunk on an item another
  chunk was the only one able to cover, losing credit for chunks that had in
  fact been retrieved.
- NDCG@10 uses binary gains: a rank slot gains 1 when it grows the matching,
  whatever number of items it adds, so a file scores one slot however many of
  its labels it answers; IDCG assumes the best case ranks first as many
  distinct FILES as the query has relevant items in, capped at NDCG_K — the
  same one-slot-per-file model DCG uses, not one slot per item (PR #42 review
  round 5, finding 1: an item-counted IDCG could not be reached by a query
  with two labels in one file, capping its best possible NDCG@10 at 0.6131).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

CATEGORIES = ("nl", "symbol", "structural")
NDCG_K = 10
LATENCY_KEYS = ("latency_p50_ms", "latency_p95_ms")

SearchFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class RelevantItem:
    path: str
    lines: tuple[int, int] | None = None
    # Both anchors as written. Kept after resolution — not only for error
    # messages, but because the default-suite tripwire re-checks them against a
    # fresh read of the file, and the END is the half that drifts: it is usually
    # the last line of a body that grows (PR #42 review). Dropping anchor_end
    # left 35 of 47 labels with only their start re-checked.
    anchor: str | None = None
    anchor_end: str | None = None


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    category: str
    query: str
    relevant: tuple[RelevantItem, ...]


def resolve_anchor(lines: list[str], anchor: str, where: str) -> int:
    """Line number (1-based) of the single line containing *anchor*.

    Substring match, so an anchor survives re-indentation. Zero matches or
    more than one both raise: an ambiguous anchor would silently pick a line
    the labeler never meant, which is the failure mode ADR-64 exists to end.
    """
    hits = [i + 1 for i, line in enumerate(lines) if anchor in line]
    if not hits:
        raise ValueError(
            f"{where}: anchor {anchor!r} matches no line — the labeled code "
            f"moved, was renamed or was deleted. Re-anchor the label."
        )
    if len(hits) > 1:
        raise ValueError(
            f"{where}: anchor {anchor!r} matches {len(hits)} lines {hits} — "
            f"an anchor must identify exactly one line. Lengthen it."
        )
    return hits[0]


# The complete key vocabulary of golden.yaml's `queries` section. Anything
# outside these sets is a typo, and a typo is refused rather than dropped.
_QUERY_KEYS = frozenset({"id", "category", "query", "relevant"})
_ITEM_KEYS = frozenset({"path", "anchor", "anchor_end"})
# The two sections golden.yaml is allowed to carry at the top level. This is
# the same class of guard round 4 added at query and item indent — a typo'd
# top-level key was still silently dropped (PR #42 review round 5, finding 25).
_TOP_LEVEL_KEYS = frozenset({"queries", "structural_patterns"})


def _reject_unknown_keys(
    where: str, mapping: dict[str, Any], allowed: frozenset[str]
) -> None:
    """Refuse keys the loader does not read, instead of silently dropping them.

    A label is data the gate's numbers are computed from, and the silent
    failure is not hypothetical: a mistyped ``anchor_end`` (``anchor-end``,
    ``anchorend``, or the correct spelling at the wrong indent) was accepted
    and dropped, collapsing the span to the single anchor line. ``matches`` is
    a span-overlap test, so that narrows what the label credits and can turn a
    hit into a miss — with nothing raised and no line of any file changed.

    ``lines`` gets its own message because it is the one key a reader of the
    pre-ADR-64 format would reach for.
    """
    unknown = set(mapping) - allowed
    if not unknown:
        return
    if "lines" in unknown:
        raise ValueError(
            f"{where} carries 'lines' — ADR-64 replaced stored line numbers "
            f"with anchors, and a 'lines' key is silently ignored rather than "
            f"honoured. Delete it; the span comes from anchor/anchor_end."
        )
    # repr() rather than the bare keys: YAML permits a non-string key (`1: 2`
    # beside `oops: 3`), and sorting int against str raises TypeError — still
    # loud, but not the message this was written to deliver.
    raise ValueError(
        f"{where} has unknown key(s) {sorted(map(repr, unknown))} — allowed: "
        f"{', '.join(sorted(allowed))}. A typo here is silently dropped, and a "
        f"dropped 'anchor_end' shrinks the span its label credits."
    )


def load_golden(path: str | Path, root: str | Path) -> list[GoldenQuery]:
    """Parse and validate golden.yaml, resolving anchors against *root*.

    Bad labels fail loudly — a silently skipped query would corrupt the gate
    numbers. ADR-64: labels address content (``anchor`` / ``anchor_end``), not
    line numbers, and are resolved against the tree being measured. *root* is
    explicit rather than inferred because the golden run indexes a **copy** of
    the tracked tree, and the line numbers must describe the copy that was
    actually chunked.
    """
    root = Path(root)
    with open(path, "rb") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    _reject_unknown_keys(str(path), raw, _TOP_LEVEL_KEYS)
    if not isinstance(raw.get("queries"), list):
        raise ValueError(f"{path}: expected a top-level 'queries' list")
    file_cache: dict[str, list[str]] = {}
    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw["queries"]):
        qid = entry.get("id")
        if not qid or qid in seen_ids:
            raise ValueError(f"{path}: query #{i} has a missing or duplicate id")
        seen_ids.add(qid)
        # Checked at BOTH levels. The item-level guard alone was escapable by
        # one indent: an `anchor_end` written one level out lands here, is
        # dropped, and the label's span silently collapses to its anchor line —
        # the very failure the guard was added for, surviving at the level
        # above it (PR #42 review round 4).
        _reject_unknown_keys(f"{path}: query {qid!r}", entry, _QUERY_KEYS)
        category = entry.get("category")
        if category not in CATEGORIES:
            raise ValueError(
                f"{path}: query {qid!r} has category {category!r}, "
                f"expected one of {CATEGORIES}"
            )
        text = entry.get("query")
        if not text or not isinstance(text, str):
            raise ValueError(f"{path}: query {qid!r} has no query text")
        rel_raw = entry.get("relevant")
        if not rel_raw:
            raise ValueError(f"{path}: query {qid!r} has no relevant items")
        relevant: list[RelevantItem] = []
        for item in rel_raw:
            rel_path = item.get("path")
            if not rel_path:
                raise ValueError(f"{path}: query {qid!r} has a relevant item with no path")
            _reject_unknown_keys(
                f"{path}: query {qid!r} item {rel_path!r}", item, _ITEM_KEYS
            )
            anchor = item.get("anchor")
            if not anchor or not isinstance(anchor, str):
                raise ValueError(
                    f"{path}: query {qid!r} item {rel_path!r} has no 'anchor' "
                    f"(ADR-64: labels are content-addressed, not line ranges)"
                )
            if rel_path not in file_cache:
                target = root / rel_path
                if not target.is_file():
                    raise ValueError(
                        f"{path}: query {qid!r} labels {rel_path!r}, which does not "
                        f"exist under {root}"
                    )
                file_cache[rel_path] = target.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            body = file_cache[rel_path]
            where = f"{path}: query {qid!r} item {rel_path!r}"
            start = resolve_anchor(body, anchor, where)
            anchor_end = item.get("anchor_end")
            if anchor_end is None:
                end = start
            else:
                end = resolve_anchor(body, anchor_end, f"{where} anchor_end")
                if end < start:
                    raise ValueError(
                        f"{where}: anchor_end resolves to line {end}, before "
                        f"anchor's line {start}"
                    )
            relevant.append(
                RelevantItem(
                    path=rel_path,
                    lines=(start, end),
                    anchor=anchor,
                    anchor_end=anchor_end,
                )
            )
        queries.append(
            GoldenQuery(id=qid, category=category, query=text, relevant=tuple(relevant))
        )
    return queries


@dataclass(frozen=True)
class StructuralPattern:
    """M5 golden entry (§3.8): a structural_search pattern with its exact
    expected per-file match counts. Evaluated pass/fail, deliberately outside
    the retrieval metrics — pattern matching is exact, so partial credit
    would only hide regressions."""

    id: str
    pattern: str
    language: str
    expected: dict[str, int]  # repo-relative path -> match count


_STRUCTURAL_PATTERN_KEYS = frozenset({"id", "pattern", "language", "expected"})


def load_structural_patterns(path: str | Path) -> list[StructuralPattern]:
    """Parse and validate the golden ``structural_patterns`` section. Same
    fail-loudly rule as load_golden: a silently skipped entry corrupts the
    exit-criterion check."""
    with open(path, "rb") as fh:
        raw = yaml.safe_load(fh)
    entries = raw.get("structural_patterns") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty 'structural_patterns' list")
    patterns: list[StructuralPattern] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        pid = entry.get("id")
        if not pid or pid in seen_ids:
            raise ValueError(f"{path}: pattern #{i} has a missing or duplicate id")
        seen_ids.add(pid)
        # This docstring already claimed load_golden's fail-loudly rule; this
        # call is what makes that true rather than aspirational — a mistyped
        # key ("expectd") used to load clean and silently drop what it named
        # (PR #42 review round 5, finding 11).
        _reject_unknown_keys(f"{path}: pattern {pid!r}", entry, _STRUCTURAL_PATTERN_KEYS)
        if not entry.get("pattern") or not entry.get("language"):
            raise ValueError(f"{path}: pattern {pid!r} needs pattern and language")
        expected = entry.get("expected")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"{path}: pattern {pid!r} has no expected match counts")
        patterns.append(
            StructuralPattern(
                id=pid,
                pattern=entry["pattern"],
                language=entry["language"],
                expected={str(k): int(v) for k, v in expected.items()},
            )
        )
    return patterns


def matches(result: dict[str, Any], item: RelevantItem) -> bool:
    if result.get("file_path") != item.path:
        return False
    if item.lines is None:
        return True
    start, end = item.lines
    return result["start_line"] <= end and result["end_line"] >= start


def dedupe_by_path(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best-ranked result per file (input is rank-ordered).

    Not called by any current scoring or reporting path — scoring uses
    :func:`group_by_path` instead (ADR-67: keeping only the best-ranked chunk
    per file silently discarded correct answers). Kept as a small,
    independently tested utility rather than removed: collapsing to one row
    per file is a real view a caller may still want, it was previously
    claimed as a reporting-path caller this function does not have (PR #42
    review round 5, finding 24).
    """
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        path = result.get("file_path")
        if path in seen:
            continue
        seen.add(path)
        deduped.append(result)
    return deduped


def group_by_path(results: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rank-ordered results by file, in first-appearance order.

    ADR-67. One file still occupies one rank slot — that was always the point
    of collapsing by path, and it stops a single file's chunks flooding recall
    — but every retrieved chunk of that file stays available for matching.

    Keeping only the best-ranked chunk was measurably discarding correct
    answers: on the 2026-08-08 run, the query ``chunk_point_id`` retrieved the
    chunk holding its definition at rank 2 and scored 0, because a *usage*
    chunk of the same file ranked 1 and displaced it. Three of forty queries
    were pinned at zero across all five channels by that alone, so they could
    never register a regression or an improvement.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(result.get("file_path"), []).append(result)
    return list(groups.values())


def score_query(
    results: list[dict[str, Any]],
    relevant: tuple[RelevantItem, ...],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Recall@k for each k plus NDCG@10 for one query (rules in module doc).

    Chunk-to-item assignment is a maximum bipartite matching, not a first-fit
    walk. Each chunk may still credit at most one item — that cap is what stops
    one wide chunk sweeping every label in its file — but *which* item it
    credits is chosen so the query scores as well as its retrieved chunks
    allow. First-fit could not do that, and lost credit for chunks it had
    already retrieved:

        labels  L0 = f.py[10,12]        L1 = f.py[40,42]
        chunks  A  = f.py[1,50]  (overlaps L0 and L1)
                B  = f.py[9,13]  (overlaps L0 only)

    First-fit gave A to L0 because L0 comes first, leaving B with nothing to
    credit and L1 unmatched: recall 0.5, with both labels sitting in retrieved
    chunks. Matching gives A to L1 and B to L0: recall 1.0. That is ADR-67's
    own defect — a retrieved, matching chunk earning no credit — one level
    below where ADR-67 fixed it.

    Exposure, as an upper bound rather than a live hit. Only a query with two
    labels in ONE file can be affected at all, and the golden set has exactly
    two: structural-03 (two async methods in embedder.py) and structural-08
    (two endpoints in routes.py). At today's chunk boundaries neither is
    actually reachable — chunking both labeled files and testing every chunk
    against both of its query's labels, no chunk overlaps more than one (5
    chunks examined, max overlap 1), so first fit and maximum matching are
    provably identical on the whole set right now. Corroborated by
    measurement: every Recall aggregate is bit-identical across the scorer
    change. This is hardening against a chunk/label shape the corpus does not
    currently produce, not a recall improvement (PR #42 review round 4).

    The matching is built incrementally in rank order, so the size after slot
    *r* is the maximum matching over slots 0..r — which is what recall@k needs,
    and it stays monotone in k. NDCG gain is still 1 per slot that adds any
    match: the file occupies one rank position however many items it answers.
    """
    if not relevant:
        # `load_golden` already refuses an empty `relevant` list at parse time,
        # so the gate can never reach this — but score_query is exported and
        # directly unit-tested, and an empty tuple otherwise divides by zero at
        # the recall line below rather than naming what went wrong (PR #42
        # review round 5, finding 23).
        raise ValueError("score_query: relevant must be non-empty")
    # Dedupe by chunk identity before grouping: two "chunks" that are really
    # one physical retrieval (same chunk_id) must not each get their own rank
    # slot in the matching below, or one wide chunk retrieved twice could
    # answer two different labels — defeating the one-chunk-one-item cap this
    # function exists to enforce. Latent today: the current retriever fuses by
    # point id, so it cannot produce a literal duplicate (PR #42 review round
    # 5, finding 26) — this is robustness against a future retriever, not a
    # live bug.
    seen_chunks: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for chunk in results:
        identity = chunk.get("chunk_id")
        if identity is None:
            identity = (chunk.get("file_path"), chunk.get("start_line"), chunk.get("end_line"))
        if identity in seen_chunks:
            continue
        seen_chunks.add(identity)
        deduped.append(chunk)

    groups = group_by_path(deduped)
    # Flat chunk list with, for each chunk, the items it could credit.
    chunk_items: list[list[int]] = []
    for group in groups:
        for chunk in group:
            chunk_items.append(
                [i for i, item in enumerate(relevant) if matches(chunk, item)]
            )

    item_chunk: dict[int, int] = {}  # relevant index -> flat chunk index

    def augment(chunk_idx: int, seen: set[int]) -> bool:
        """Kuhn's augmenting path: try to give *chunk_idx* an item, displacing
        an earlier chunk onto a different item of its own where that frees one
        up. Chunk capacity stays 1; the sets are tiny (<=10 slots, <=3 items)."""
        for i in chunk_items[chunk_idx]:
            if i in seen:
                continue
            seen.add(i)
            if i not in item_chunk or augment(item_chunk[i], seen):
                item_chunk[i] = chunk_idx
                return True
        return False

    matched_after: list[int] = []  # matching size after each rank slot
    flat = 0
    for group in groups:
        for _ in group:
            augment(flat, set())
            flat += 1
        matched_after.append(len(item_chunk))

    scores: dict[str, float] = {}
    for k in ks:
        # `matched_after[min(k, n) - 1]` is "the matching after the first k rank
        # slots". k <= 0 is refused rather than allowed to index from the end:
        # k=0 would read matched_after[-1], the FULL matching, and report every
        # item as found "in the top 0 slots" (PR #42 review round 4). No caller
        # passes it today; the guard is here because `ks` is a parameter.
        if k <= 0:
            raise ValueError(f"score_query: k must be >= 1, got {k}")
        found = matched_after[min(k, len(matched_after)) - 1] if matched_after else 0
        scores[f"recall@{k}"] = found / len(relevant)

    gains = [
        1 if size > (matched_after[r - 1] if r else 0) else 0
        for r, size in enumerate(matched_after)
    ]
    dcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(gains[:NDCG_K]))
    # IDCG must imagine the same rank-slot model DCG uses: one slot per FILE,
    # not per item (PR #42 review round 5). Before this fix, two labels in one
    # file made IDCG assume two ideal slots when DCG can only ever fill one —
    # a perfect answer to structural-03/structural-08 (every label hit, at the
    # best possible rank) scored a flat 0.6131, indistinguishable from a half
    # answer. `{item.path for item in relevant}` is the most slots the DCG
    # model can ever fill, which is what "ideal" has to mean here.
    ideal = min(len({item.path for item in relevant}), NDCG_K)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
    scores[f"ndcg@{NDCG_K}"] = dcg / idcg if idcg else 0.0
    return scores


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation): the smallest value with at
    least ``pct`` percent of the sample at or below it. Deterministic and
    honest at eval-set sizes (~40 queries)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(rank, 1) - 1]


def _mean_rows(per_query: list[dict[str, Any]], metric_keys: list[str]) -> dict:
    row: dict[str, Any] = {"n_queries": len(per_query)}
    for key in metric_keys:
        row[key] = sum(q[key] for q in per_query) / len(per_query) if per_query else 0.0
    latencies = [q["latency_ms"] for q in per_query if "latency_ms" in q]
    row["latency_p50_ms"] = percentile(latencies, 50)
    row["latency_p95_ms"] = percentile(latencies, 95)
    return row


async def evaluate(
    search_fn: SearchFn,
    golden: list[GoldenQuery],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Run every golden query through *search_fn* and aggregate the report:
    ``{"overall": row, "categories": {cat: row}, "queries": [...]}``."""
    metric_keys = [f"recall@{k}" for k in ks] + [f"ndcg@{NDCG_K}"]
    per_query: list[dict[str, Any]] = []
    for gq in golden:
        started = time.perf_counter()
        results = await search_fn(gq.query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        scores = score_query(results, gq.relevant, ks=ks)
        per_query.append(
            {
                "id": gq.id,
                "category": gq.category,
                **scores,
                "latency_ms": elapsed_ms,
            }
        )

    report: dict[str, Any] = {
        "overall": _mean_rows(per_query, metric_keys),
        "categories": {
            cat: _mean_rows([q for q in per_query if q["category"] == cat], metric_keys)
            for cat in CATEGORIES
        },
        "queries": per_query,
    }
    return report


def save_reference(
    reports: dict[str, dict[str, Any]],
    path: str | Path,
    provenance: dict[str, Any],
) -> None:
    """Write the living reference: every channel's rows plus full provenance.

    ADR-65 layer 3. There is deliberately no write-if-missing convenience and
    no unconditional write — both existed before and both recorded whichever
    run got there first as the standard every later run was judged against
    (lesson 8). The only caller is the explicit re-baseline path.

    Per-query rows are stored beside the aggregates (ADR-69, lesson 19). The
    aggregates alone could confirm every aggregate claim and refute no
    per-query one: when a per-channel table for `structural-08` went out with
    two wrong rows in the PR #42 round-1 reply, the only artifact that could
    have falsified it was `report_latest.json`, which is gitignored. A claim
    about one query had nowhere to live. Lesson 19's own remedy is this — put
    the per-item view into the artifact of record — and it is what makes the
    gate's own numbers re-derivable by a reader holding only the repository.

    Latency is deliberately NOT stored per query: it is device- and
    load-specific, so it would churn all 200 rows on every re-baseline even
    when quality is bit-identical, and drown the rows that actually moved. The
    p50/p95 summary in each row above keeps the cost visible.
    """
    for channel, report in reports.items():
        # Refuse rather than write `"queries": []`. Without this a re-baseline
        # ran the full ~15 GPU-minutes, exited green, and left a reference the
        # DEFAULT suite then rejected as rows-less — the cost paid, the artifact
        # unusable, and the failure surfacing on someone else's next `pytest`
        # (PR #42 review round 4). ADR-69: the rows ARE the artifact.
        rows = report.get("queries")
        if not rows:
            raise ValueError(
                f"save_reference: channel {channel!r} carries no per-query rows, "
                f"so the reference could not support a per-query claim (ADR-69). "
                f"Refusing to write {path}."
            )
        # `id` and `category` are as load-bearing as the three metrics below:
        # they're what a per-query row is even identified and grouped by. Only
        # _METRICS was checked here, so a row missing `id` or `category` raised
        # a bare KeyError two lines down instead of this named refusal (PR #42
        # review round 5, finding 22).
        required_keys = (*_METRICS, "id", "category")
        missing = sorted({k for k in required_keys for row in rows if k not in row})
        if missing:
            raise ValueError(
                f"save_reference: channel {channel!r} has rows missing {missing}, "
                f"so the reference could not support a per-query claim (ADR-69). "
                f"Refusing to write {path}."
            )
    payload = {
        "provenance": provenance,
        "channels": {
            channel: {
                "overall": report["overall"],
                "categories": report["categories"],
                # No `if m in row` guard: a row missing a metric is a harness
                # defect, and dropping it here would surface as a bare KeyError
                # inside test_reference_integrity instead.
                "queries": [
                    {
                        "id": row["id"],
                        "category": row["category"],
                        **{m: row[m] for m in _METRICS},
                    }
                    for row in report["queries"]
                ],
            }
            for channel, report in reports.items()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_reference(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


_METRICS = ("recall@5", "recall@10", f"ndcg@{NDCG_K}")

# ADR-70: the scoring rules are part of a reference's provenance.
#
# `reference_mismatches` already refuses a reference measured on different
# models, a different store or different labels. Until PR #42 round 4 it had no
# term for the SCORER, so numbers produced under one scoring rule could be
# gated against numbers produced under another — and that is not hypothetical:
# ADR-67 and ADR-68 each changed what a retrieved chunk credits, and both times
# the only thing that forced a re-baseline was a human remembering to.
#
# Bump this whenever a change to `score_query`, `group_by_path` or `matches`
# can move a stored number. A reference written before the field existed
# carries no version, reads as None, and correctly refuses to gate.
#
# Policy (PR #42 review round 5, finding 7): bumping this turns the DEFAULT
# suite red on every pull request until someone spends the ~15 GPU-minutes to
# re-baseline (test_the_stored_provenance_could_gate_a_run), which is the
# same incentive ADR-69 explicitly refused to build for labels. Kept anyway,
# deliberately: a label edit is frequent and cheap and must not be
# discouraged (issue #38's own failure mode), while a scorer change is rare
# and inherently invalidates the measurement it touches — silently comparing
# across it is exactly the mistake this field exists to prevent. The rule
# this implies: a scorer change and its re-baseline land in the same commit.
#   1  grouped scoring, credit assigned by first fit (ADR-67)
#   2  credit assigned by maximum bipartite matching (ADR-68)
#   3  NDCG's ideal counts unique FILES, not items — matching what DCG can
#      actually fill (PR #42 review round 5, finding 1)
SCORER_VERSION = 3


# --- ADR-65: provenance, comparability, regression ------------------------
#
# A drop fires only when it exceeds BOTH a relative band and one query's
# worth of absolute movement.
#
# The relative band alone is not enough, and the arithmetic is the reason: the
# gate measures (before - after) / before, while one query flipping moves the
# mean by 1/n absolutely. As a relative figure that is (1/n) / before, which
# grows without limit as `before` falls — at n=12 a single flipped query is a
# 28% relative drop against a 0.30 baseline but only 10% against 0.79. So a
# fixed relative band silently changes its meaning with the score level, and
# post-repair category levels are not yet measured.
#
# The absolute floor is what actually encodes "one query is noise": 1/n_queries
# is the largest a single query can move any of these metrics (recall and NDCG
# are both bounded to [0,1] per query), and it is read from the row at runtime
# rather than hard-coded, so it stays correct if the golden set is ever resized.
OVERALL_REGRESSION_TOLERANCE = 0.10
CATEGORY_REGRESSION_TOLERANCE = 0.20
# Corpus drift beyond this makes the stored reference incomparable rather
# than merely older: more documents against a fixed label set depress recall
# mechanically, independent of retrieval quality (issue #38, Finding B).
CORPUS_DRIFT_TOLERANCE = 0.20

# Recorded in the reference and reported, never gated. A diagnostic channel's
# job is to EXPLAIN a regression the gated channels already detect — the
# dense -> dense(python-only) gap separates "the haystack grew" from
# "retrieval got worse". Gating it would double-count evidence `dense`
# already carries and add a threshold with no exit criterion behind it.
DIAGNOSTIC_CHANNELS = frozenset({"dense (python-only)"})


def manifest_sha256(rel_paths: Iterable[str]) -> str:
    """Stable fingerprint of a corpus: sha256 over its sorted relative paths.

    Names *which* files were measured, not their contents — content drift is
    already covered by the chunk count and the commit sha, and a content hash
    would invalidate the reference on every unrelated edit.

    Recorded as evidence, deliberately NOT used as a comparability predicate
    (see ``reference_mismatches``). Any repo gains and loses files constantly,
    so a reference gated on an exact path-set match would be incomparable
    after almost every commit — which is Finding A rebuilt in a new costume: a
    gate that never fires. Its job is to let a reader of two reports see
    exactly which corpus produced each number.
    """
    return hashlib.sha256("\n".join(sorted(rel_paths)).encode("utf-8")).hexdigest()


def golden_digest(path: str | Path) -> str:
    """Fingerprint of the golden set's *questions*, not its file bytes.

    Hashes the parsed ``queries`` list, so re-wording a comment or reflowing
    the YAML leaves the reference comparable while any change to a query,
    category or label invalidates it. Hashing the file would tie the gate to
    its own formatting and force a 2 h re-baseline over a typo fix.
    """
    with open(path, "rb") as fh:
        raw = yaml.safe_load(fh)
    payload = json.dumps(raw.get("queries"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_mismatches(reference: dict[str, Any], run: dict[str, Any]) -> list[str]:
    """Reasons the stored reference cannot gate *run*, most decisive first.

    Empty list means comparable. A reference measured on different models, a
    different store implementation or different labels is not a weaker
    reference — it is a different experiment, and comparing against it is the
    provenance-blind mistake lesson 8 was recorded for.
    """
    # `or {}` at every section, not `.get(section, {})` alone: a section
    # present but explicitly `null` (a malformed or hand-edited provenance
    # block) must refuse the same way a MISSING section does, not raise
    # AttributeError out of the next `.get()` (PR #42 review round 5, finding
    # 12). `.get(section, {})` only covers absence; `null` is present.
    reasons: list[str] = []
    ref_models, run_models = reference.get("models") or {}, run.get("models") or {}
    for key in ("embedding_model", "reranker_model"):
        if ref_models.get(key) != run_models.get(key):
            reasons.append(
                f"{key}: reference {ref_models.get(key)!r} != run {run_models.get(key)!r}"
            )
    ref_scorer = (reference.get("scoring") or {}).get("version")
    run_scorer = (run.get("scoring") or {}).get("version")
    if ref_scorer != run_scorer:
        reasons.append(
            f"scorer version: reference {ref_scorer!r} != run {run_scorer!r} — "
            f"the two sets of numbers were produced by different scoring rules "
            f"(ADR-70), so a difference between them measures the scorer, not "
            f"retrieval"
        )
    ref_store, run_store = reference.get("store") or {}, run.get("store") or {}
    if ref_store.get("kind") != run_store.get("kind"):
        reasons.append(
            f"store kind: reference {ref_store.get('kind')!r} != "
            f"run {run_store.get('kind')!r}"
        )
    elif ref_store.get("server_version") != run_store.get("server_version"):
        reasons.append(
            f"server version: reference {ref_store.get('server_version')!r} != "
            f"run {run_store.get('server_version')!r}"
        )
    ref_labels, run_labels = reference.get("labels") or {}, run.get("labels") or {}
    if ref_labels.get("golden_sha256") != run_labels.get("golden_sha256"):
        reasons.append("golden.yaml changed since the reference was recorded")
    if ref_labels.get("n_queries") != run_labels.get("n_queries"):
        reasons.append(
            f"query count: reference {ref_labels.get('n_queries')} != "
            f"run {run_labels.get('n_queries')}"
        )
    ref_corpus, run_corpus = reference.get("corpus") or {}, run.get("corpus") or {}
    ref_chunks, run_chunks = ref_corpus.get("chunks"), run_corpus.get("chunks")
    if not ref_chunks:
        reasons.append("reference records no chunk count")
    elif not run_chunks:
        # Symmetric with the line above (PR #42 review): a run that cannot say
        # how big its corpus was is exactly as uncomparable as a reference that
        # cannot, and skipping the check instead would gate against an unknown
        # corpus — Finding B with the numbers hidden one level deeper.
        reasons.append("run records no chunk count")
    else:
        drift = abs(run_chunks - ref_chunks) / ref_chunks
        if drift > CORPUS_DRIFT_TOLERANCE:
            # Report the file counts beside the chunk counts: chunks are what
            # the band is computed on, but files are the number a human can
            # act on ("the repo grew by 90 docs"), and a message that only
            # says "chunks" invites re-deriving that from scratch.
            ref_files = ref_corpus.get("files_indexed")
            run_files = run_corpus.get("files_indexed")
            reasons.append(
                f"corpus drift {drift:.1%} exceeds {CORPUS_DRIFT_TOLERANCE:.0%} "
                f"({ref_chunks} -> {run_chunks} chunks, "
                f"{ref_files} -> {run_files} files indexed)"
            )
    return reasons


def relational_failures(reports: dict[str, dict[str, Any]]) -> list[str]:
    """M3's exit criterion, asserted from the same run (ADR-65 layer 1).

    'Hybrid beats dense-only on Recall@10, especially the symbol subset'
    (docs/project/milestones.md). Same corpus, labels, models and run, so
    corpus growth cannot invalidate it by construction — which is exactly why
    this layer, unlike the regression layer, always asserts.

    On the two operators, because they look inverted and are not. Overall is
    strict: "beats" is the milestone's own word, and it is measured over all
    40 queries, where a tie is a real result. The symbol subset is n=14, where
    one query is 7 points, and §6.2 of the approved draft says in terms "at
    that sample size, ignore sub-point differences — they are noise"; a tie
    there is the resolution limit, not a regression, so requiring "must not
    lose" is the honest reading of the emphasis. The measured margin is
    reported either way, so a shrinking lead is visible before it inverts.

    The third relation is ADR-35's shipped claim (docs/internals/evaluation.md
    quotes rerank as a uniform NDCG win). If it stops holding, a published
    number has become false and the owner needs to know from the gate rather
    than from a reader.
    """
    failures: list[str] = []
    dense, hybrid = reports["dense"], reports["hybrid"]
    rerank = reports.get("hybrid+rerank")
    metric = "recall@10"
    if hybrid["overall"][metric] <= dense["overall"][metric]:
        failures.append(
            f"M3 exit criterion: hybrid {metric} {hybrid['overall'][metric]:.4f} "
            f"must beat dense {dense['overall'][metric]:.4f} (overall)"
        )
    sym_h = hybrid["categories"]["symbol"][metric]
    sym_d = dense["categories"]["symbol"][metric]
    if sym_h < sym_d:
        failures.append(
            f"M3 exit criterion: hybrid {metric} {sym_h:.4f} must not lose to "
            f"dense {sym_d:.4f} on the symbol subset — the sparse channel's "
            f"reason to exist"
        )
    if rerank is not None:
        ndcg = f"ndcg@{NDCG_K}"
        if rerank["overall"][ndcg] < hybrid["overall"][ndcg]:
            failures.append(
                f"ADR-35: hybrid+rerank {ndcg} {rerank['overall'][ndcg]:.4f} must "
                f"not lose to same-run hybrid {hybrid['overall'][ndcg]:.4f} — the "
                f"reranker's only justification for its latency cost"
            )
    return failures


def regression_failures(
    reports: dict[str, dict[str, Any]],
    reference_channels: dict[str, dict[str, Any]],
) -> list[str]:
    """Relative-drop breaches against the stored reference (ADR-65 layer 2)."""
    failures: list[str] = []
    for channel, ref in sorted(reference_channels.items()):
        if channel in DIAGNOSTIC_CHANNELS:
            continue
        run = reports.get(channel)
        if run is None:
            failures.append(f"{channel}: reference has this channel, run does not")
            continue
        for scope, tol in (("overall", OVERALL_REGRESSION_TOLERANCE),) + tuple(
            (cat, CATEGORY_REGRESSION_TOLERANCE) for cat in CATEGORIES
        ):
            ref_row = ref["overall"] if scope == "overall" else ref["categories"][scope]
            run_row = run["overall"] if scope == "overall" else run["categories"][scope]
            n_queries = run_row.get("n_queries") or ref_row.get("n_queries") or 0
            one_query = 1.0 / n_queries if n_queries else 0.0
            for metric in _METRICS:
                before, after = ref_row[metric], run_row[metric]
                if before <= 0:
                    continue
                drop_abs = before - after
                if drop_abs <= 0:
                    continue
                if drop_abs > max(tol * before, one_query):
                    failures.append(
                        f"{channel}/{scope}/{metric}: {before:.4f} -> {after:.4f} "
                        f"(-{drop_abs:.4f} = {drop_abs / before:.1%}; limit is the "
                        f"greater of {tol:.0%} relative and one query "
                        f"({one_query:.4f}) over n={n_queries})"
                    )
    return failures


def zero_recall_queries(reports: dict[str, dict[str, Any]]) -> list[str]:
    """Query ids that scored 0 recall@10 on **every** channel.

    Dense, sparse, hybrid and rerank fail for different reasons, so a query
    that all four miss is far likelier to be a broken label than a retrieval
    failure. The intersection covers the diagnostic channel too — all five, not
    the four gated ones — and that is deliberate rather than an oversight: every
    label is a ``.py`` file, so the python-only filter can remove distractors
    and never a true positive, and a query it *can* answer is by definition not
    one whose label is unreachable. Including it only ever shortens this list,
    and shortens it by exactly the queries that do not belong on it.
    Surfacing this is the loop back to Finding C: had the harness
    printed it, `nl-10` and `symbol-12` — which could never match anything —
    would have been visible from the very first M2 run instead of being found
    two milestones later by hand-reading a JSON file.
    """
    per_channel = [
        {row["id"] for row in report["queries"] if row["recall@10"] == 0.0}
        for report in reports.values()
        if "queries" in report
    ]
    if not per_channel:
        return []
    return sorted(set.intersection(*per_channel))


def rerank_query_deltas(
    reports: dict[str, dict[str, Any]], metric: str = "recall@10"
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Per-query ``(gained, lost)`` for hybrid+rerank against same-run hybrid.

    The gate already asserts the reranker's *aggregate* win (ADR-35, layer 1),
    and the aggregate is what the docs quote. What no artifact recorded was the
    per-query cost underneath it: against the CURRENT reference — the one that
    stores the per-query rows this claim is re-derivable from, not the
    2026-08-09 one, which stored none (PR #42 review round 5, finding 18) — the
    reranker gains 5 queries (nl-11, symbol-04, symbol-06, structural-02,
    structural-04) and loses 3 (nl-01, structural-08, structural-10), netting
    +0.05 on Recall@10 — a real win paid for by demoting three answers it had
    already retrieved (PR #42 round-2 review, which found this by hand after a
    per-query claim went out wrong).

    Reported, never gated. A cross-encoder re-ordering some queries downward
    while winning overall is expected behaviour, not a regression; the reason to
    surface it is that "which queries does rerank hurt" was previously
    answerable only by re-reading raw JSON.
    """
    gained: list[tuple[str, float, float]] = []
    lost: list[tuple[str, float, float]] = []
    hybrid, rerank = reports.get("hybrid"), reports.get("hybrid+rerank")
    if not hybrid or not rerank or "queries" not in hybrid or "queries" not in rerank:
        return gained, lost
    before = {row["id"]: row for row in hybrid["queries"]}
    for row in rerank["queries"]:
        prior = before.get(row["id"])
        if prior is None or metric not in prior or metric not in row:
            continue
        if row[metric] > prior[metric]:
            gained.append((row["id"], prior[metric], row[metric]))
        elif row[metric] < prior[metric]:
            lost.append((row["id"], prior[metric], row[metric]))
    return gained, lost


def reference_query_deltas(
    reports: dict[str, dict[str, Any]],
    reference_channels: dict[str, Any],
    metric: str = "recall@10",
) -> list[tuple[str, str, float, float]]:
    """Queries that crossed zero against the reference: ``(channel, id, was, now)``.

    ADR-71. ADR-69 committed the per-query rows so a reader could check a
    per-query claim by hand — but nothing *compared* them between runs, and the
    aggregates cannot: permuting 12 of 40 hybrid rows among same-category
    queries leaves every stored aggregate bit-identical and both gate layers
    passing (PR #42 review round 4). A channel can therefore swap which queries
    it answers, completely, and the gate sees a flat line.

    Crossing zero is the signal worth naming: a query that went from answered
    to unanswered is either a retrieval regression or a rotted label, and both
    are worth a human look. A query moving 1.0 -> 0.5 is ordinary rank churn
    and is left to the aggregate.

    Reported, never gated — the same standing as :func:`rerank_query_deltas`.
    Per-query retrieval is noisy, the cost of a false red here is a 15-minute
    re-run, and the gate already has a layer that fires on the aggregate.

    Includes DIAGNOSTIC_CHANNELS deliberately, unlike layer 2's
    :func:`regression_failures` — this view is reported, not gated, so there
    is no double-counting risk to avoid, and a crossing on the python-only
    channel is still useful evidence of what changed (same reasoning
    :func:`zero_recall_queries` states for its own choice to include it; PR
    #42 review round 5, finding 20).
    """
    crossings: list[tuple[str, str, float, float]] = []
    for channel in sorted(reports):
        run_rows = reports[channel].get("queries") or []
        ref_rows = (reference_channels.get(channel) or {}).get("queries") or []
        before = {row["id"]: row for row in ref_rows}
        for row in run_rows:
            prior = before.get(row["id"])
            if prior is None or metric not in prior or metric not in row:
                continue
            was, now = prior[metric], row[metric]
            if (was > 0) != (now > 0):
                crossings.append((channel, row["id"], was, now))
    return crossings


def render_report(
    reports: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    relational: list[str],
    mismatches: list[str],
    regression: list[str] | None,
    rebaseline: bool = False,
    wrote_reference: bool = False,
    reference_channels: dict[str, Any] | None = None,
) -> str:
    """The run, as markdown — verdicts first, so `-q` can no longer bury them.

    *regression* is None when the reference was missing or incomparable; that
    case renders the tables under an explicit NOT A GATE banner rather than
    letting a reader mistake an ungated table for a passing one.

    *rebaseline* is True when this run was ASKED to re-baseline;
    *wrote_reference* is True only once the file has actually been written.
    They are separate because they can disagree, and the report used to assume
    they could not: the report is rendered before the gate's assertions run, so
    a re-baseline run that failed layer 1 left a persisted report announcing
    "THIS RUN WROTE THE REFERENCE ... now holds these numbers" for a file it
    never touched (PR #42 review round 4). The stamp has to reach the file at
    all — re-baselining is the most consequential thing a run can do, the
    persisted report is the artifact of record (`.claude/commands/eval.md` sends
    the reader here rather than to the terminal), and before round 2 it was
    announced only by a stdout `print` that pytest's capture eats — but it has
    to say which of the two things happened.

    *reference_channels* enables the ADR-71 per-query block; None omits it.
    """
    lines = ["# Golden run", "", "## Verdicts", ""]
    lines.append(
        "- L1 relational (same-run): "
        + ("PASS" if not relational else f"**FAIL** ({len(relational)})")
    )
    for failure in relational:
        lines.append(f"    - {failure}")
    if regression is None:
        # Keyed off OUTCOME (wrote_reference), not REQUEST (rebaseline): a
        # re-baseline that was asked for but never written — because L1 failed
        # or because the write itself was refused — must not read as "this run
        # re-baselined" one line above the L3 stamp that says otherwise (PR #42
        # review round 5, finding 3; round 4's own fix left this sibling line).
        if rebaseline and wrote_reference:
            l2_reason = "this run re-baselined"
        elif rebaseline:
            l2_reason = "a re-baseline was requested, so nothing was gated"
        else:
            l2_reason = "no comparable reference"
        lines.append("- L2 regression vs reference: **NOT RUN** — " + l2_reason)
        # Label them. Under a re-baseline the line above already gave a reason,
        # so bare bullets read as if they explained it; they are a second,
        # independent fact (PR #42 review round 4).
        if mismatches and rebaseline:
            lines.append("    - the stored reference was also incomparable:")
        for reason in mismatches:
            lines.append(f"    {'    ' if rebaseline else ''}- {reason}")
    else:
        lines.append(
            "- L2 regression vs reference: "
            + ("PASS" if not regression else f"**FAIL** ({len(regression)})")
        )
        for failure in regression:
            lines.append(f"    - {failure}")
    if rebaseline and wrote_reference:
        lines.append(
            "- L3 re-baseline: **THIS RUN WROTE THE REFERENCE** "
            "(`NOESIS_EVAL_REBASELINE=1`)"
        )
        lines.append(
            "    - `tests/eval/baselines/reference.json` now holds these numbers. "
            "Nothing measured them against a standard — they *are* the new "
            "standard, and L1 above is the only assertion they passed."
        )
    elif rebaseline:
        lines.append(
            "- L3 re-baseline: **REQUESTED BUT NOT WRITTEN** "
            "(`NOESIS_EVAL_REBASELINE=1`)"
        )
        # "a gate layer above failed" was true only for the L1 case. When L1
        # PASSES and the write still didn't happen, the only remaining cause
        # is save_reference itself refusing (e.g. ADR-69's rows-less guard) —
        # `relational` is empty precisely then, since the assertion above this
        # branch already stopped the run before reaching `if rebaseline:` on
        # any L1 failure (PR #42 review round 5, finding 3b).
        if relational:
            lines.append(
                "    - the relational gate (L1) failed above, so "
                "`tests/eval/baselines/reference.json` was NOT touched and "
                "still holds the previous numbers. A run that fails its own "
                "exit criterion is not a measurement standard (lesson 8)."
            )
        else:
            lines.append(
                "    - the write itself was refused (see the error above) — "
                "`tests/eval/baselines/reference.json` was NOT touched and "
                "still holds the previous numbers."
            )
    zero = zero_recall_queries(reports)
    lines.append(
        "- Zero-recall on every channel: "
        + ("none" if not zero else f"**{len(zero)}** — {', '.join(zero)}")
    )
    if zero:
        lines.append(
            "    - A query no channel can answer is a label to inspect before it "
            "is a retrieval regression (issue #38, Finding C)."
        )
    gained, lost = rerank_query_deltas(reports)
    if gained or lost:
        lines.append(
            f"- Reranker vs same-run hybrid (recall@10, reported not gated): "
            f"**+{len(gained)} / −{len(lost)}** queries"
        )
        for qid, before_v, after_v in lost:
            lines.append(f"    - lost: {qid} {before_v:.2f} -> {after_v:.2f}")
    has_reference_rows = any(
        (channel or {}).get("queries") for channel in (reference_channels or {}).values()
    )
    if has_reference_rows:
        # ADR-71: the aggregates cannot see a channel swap WHICH queries it
        # answers. Reported, never gated. Skipped against a reference recorded
        # before ADR-69, which stores no rows to compare — otherwise the
        # zero-recall line below would report every query as newly changed.
        crossings = reference_query_deltas(reports, reference_channels)
        lines.append(
            "- Per-query vs reference (recall@10 crossing zero, reported not "
            "gated): " + (f"**{len(crossings)}**" if crossings else "none")
        )
        for channel, qid, was, now in crossings:
            verb = "lost" if was > now else "gained"
            lines.append(
                f"    - {verb}: {qid} on {channel} {was:.2f} -> {now:.2f}"
            )
        ref_zero = zero_recall_queries(reference_channels)
        if ref_zero != zero:
            lines.append(
                "    - zero-recall set changed: was "
                + (", ".join(ref_zero) or "none")
                + " / now "
                + (", ".join(zero) or "none")
            )
    if regression is None:
        lines.append("")
        lines.append("> **THE TABLES BELOW ARE NOT A GATE.**")
    lines += ["", "## Provenance", "", format_provenance(provenance)]
    lines += ["", "## Channels (quality + latency)", "", format_table(reports)]
    return "\n".join(lines) + "\n"


def format_provenance(provenance: dict[str, Any]) -> str:
    """Markdown key/value table — what this run measured, so a reader a month
    later can tell whether two numbers were ever comparable."""
    lines = ["| field | value |", "|---|---|"]
    for section, values in sorted(provenance.items()):
        if isinstance(values, dict):
            for key, value in sorted(values.items()):
                lines.append(f"| {section}.{key} | `{value}` |")
        else:
            lines.append(f"| {section} | `{values}` |")
    return "\n".join(lines)


def format_table(reports: dict[str, dict[str, Any]]) -> str:
    """Markdown table: one row per (category, channel), overall last.
    Latency columns appear when the rows carry them (fresh runs do; stored
    pre-M4 baselines may not)."""
    channels = list(reports)
    columns = list(_METRICS)
    # ALL channels, not just the first: sampling one channel to decide whether
    # latency columns exist, then indexing every channel with them, KeyErrors
    # the moment a later channel is the one without them — the exact mixed
    # case this function's own docstring describes (PR #42 review round 5,
    # finding 28).
    if all(
        all(key in report["overall"] for key in LATENCY_KEYS)
        for report in reports.values()
    ):
        columns += list(LATENCY_KEYS)
    lines = [
        "| category | n | channel | " + " | ".join(columns) + " |",
        "|---|---|---|" + "---|" * len(columns),
    ]
    for cat in (*CATEGORIES, "overall"):
        for channel in channels:
            report = reports[channel]
            row = report["overall"] if cat == "overall" else report["categories"][cat]
            cells = " | ".join(
                f"{row[c]:.1f}" if c in LATENCY_KEYS else f"{row[c]:.3f}"
                for c in columns
            )
            lines.append(f"| {cat} | {row['n_queries']} | {channel} | {cells} |")
    return "\n".join(lines)


def format_delta(
    challenger: dict[str, Any],
    baseline: dict[str, Any],
    challenger_label: str = "hybrid",
    baseline_label: str = "m2 dense (stored)",
) -> str:
    """Challenger-vs-baseline quality delta per category (gate view: M3 used
    hybrid vs stored M2 dense; M4 uses hybrid+rerank vs same-run hybrid).
    Quality metrics only — latency is reported, not gated, by this table."""
    lines = [
        f"| category | metric | {baseline_label} | {challenger_label} | delta |",
        "|---|---|---|---|---|",
    ]
    for cat in (*CATEGORIES, "overall"):
        cha = (
            challenger["overall"] if cat == "overall" else challenger["categories"][cat]
        )
        base = baseline["overall"] if cat == "overall" else baseline["categories"][cat]
        for metric in _METRICS:
            delta = cha[metric] - base[metric]
            lines.append(
                f"| {cat} | {metric} | {base[metric]:.3f} | {cha[metric]:.3f} "
                f"| {delta:+.3f} |"
            )
    return "\n".join(lines)
