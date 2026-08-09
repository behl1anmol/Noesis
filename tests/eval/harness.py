"""Evaluation harness (doc §6.2; M3 gate, extended for the M4 gate).

Loads the human-labeled golden set (``tests/eval/golden.yaml``), runs a
search function per query, and reports Recall@5, Recall@10 and NDCG@10 per
query category (nl / symbol / structural) plus overall. The M3 gate compares
the hybrid channel against the stored M2 dense-only baseline
(``tests/eval/baselines/m2_dense.json``) — numbers or it didn't happen.

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
- Recall@k = fraction of a query's relevant items matched by at least one
  result in the top k (after dedup), averaged over queries.
- NDCG@10 uses binary gains with greedy credit: walking the deduped ranking,
  a result gains 1 only the first time it matches a not-yet-credited
  relevant item; IDCG assumes all relevant items ranked first.
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
    anchor: str | None = None  # the label as written; kept for error messages


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
    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise ValueError(f"{path}: expected a top-level 'queries' list")
    file_cache: dict[str, list[str]] = {}
    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw["queries"]):
        qid = entry.get("id")
        if not qid or qid in seen_ids:
            raise ValueError(f"{path}: query #{i} has a missing or duplicate id")
        seen_ids.add(qid)
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
                RelevantItem(path=rel_path, lines=(start, end), anchor=anchor)
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

    Retained for reporting a one-row-per-file view. Scoring uses
    :func:`group_by_path` instead — see ADR-67 for why keeping only the
    best-ranked chunk silently discarded correct answers.
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
    """Recall@k for each k plus NDCG@10 for one query (rules in module doc)."""
    groups = group_by_path(results)
    matched_rank: dict[int, int] = {}  # relevant index -> rank credited
    gains: list[int] = []
    for rank, group in enumerate(groups):
        gain = 0
        # Each chunk may credit at most one not-yet-credited item, so a file
        # holding two distinct relevant items can satisfy both — via two
        # different chunks — which the golden set does have (structural-08
        # labels two endpoints in routes.py). The NDCG gain stays 1 for the
        # slot regardless: the file occupies one rank position however many
        # items it answers.
        #
        # The cap is per CHUNK, so two same-file labels inside one chunk still
        # credit only one and cap that query at 1/len(relevant) (ADR-67,
        # pinned by test_two_relevant_greedy_credit). Deliberate: without it a
        # single wide chunk would sweep every label in its file and score 1.0
        # for retrieving one thing.
        for chunk in group:
            for i, item in enumerate(relevant):
                if i in matched_rank:
                    continue
                if matches(chunk, item):
                    matched_rank[i] = rank
                    gain = 1
                    break
        gains.append(gain)

    scores: dict[str, float] = {}
    for k in ks:
        found = sum(1 for rank in matched_rank.values() if rank < k)
        scores[f"recall@{k}"] = found / len(relevant)

    dcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(gains[:NDCG_K]))
    ideal = min(len(relevant), NDCG_K)
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
    """
    payload = {
        "provenance": provenance,
        "channels": {
            channel: {
                "overall": report["overall"],
                "categories": report["categories"],
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
    reasons: list[str] = []
    ref_models, run_models = reference.get("models", {}), run.get("models", {})
    for key in ("embedding_model", "reranker_model"):
        if ref_models.get(key) != run_models.get(key):
            reasons.append(
                f"{key}: reference {ref_models.get(key)!r} != run {run_models.get(key)!r}"
            )
    ref_store, run_store = reference.get("store", {}), run.get("store", {})
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
    ref_labels, run_labels = reference.get("labels", {}), run.get("labels", {})
    if ref_labels.get("golden_sha256") != run_labels.get("golden_sha256"):
        reasons.append("golden.yaml changed since the reference was recorded")
    if ref_labels.get("n_queries") != run_labels.get("n_queries"):
        reasons.append(
            f"query count: reference {ref_labels.get('n_queries')} != "
            f"run {run_labels.get('n_queries')}"
        )
    ref_corpus, run_corpus = reference.get("corpus", {}), run.get("corpus", {})
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
    failure. Surfacing this is the loop back to Finding C: had the harness
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


def render_report(
    reports: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    relational: list[str],
    mismatches: list[str],
    regression: list[str] | None,
) -> str:
    """The run, as markdown — verdicts first, so `-q` can no longer bury them.

    *regression* is None when the reference was missing or incomparable; that
    case renders the tables under an explicit NOT A GATE banner rather than
    letting a reader mistake an ungated table for a passing one.
    """
    lines = ["# Golden run", "", "## Verdicts", ""]
    lines.append(
        "- L1 relational (same-run): "
        + ("PASS" if not relational else f"**FAIL** ({len(relational)})")
    )
    for failure in relational:
        lines.append(f"    - {failure}")
    if regression is None:
        lines.append("- L2 regression vs reference: **NOT RUN** — no comparable reference")
        for reason in mismatches:
            lines.append(f"    - {reason}")
        lines.append("")
        lines.append("> **THE TABLES BELOW ARE NOT A GATE.**")
    else:
        lines.append(
            "- L2 regression vs reference: "
            + ("PASS" if not regression else f"**FAIL** ({len(regression)})")
        )
        for failure in regression:
            lines.append(f"    - {failure}")
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
    sample = next(iter(reports.values()))["overall"]
    if all(key in sample for key in LATENCY_KEYS):
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
