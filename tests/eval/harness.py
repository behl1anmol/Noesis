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

Scoring rules (deliberate, stated so the numbers are reproducible):

- A result matches a relevant item iff the ``file_path`` is equal and, when
  the item carries a ``lines: [start, end]`` range, the result span
  ``[start_line, end_line]`` overlaps it.
- Results are deduplicated by ``file_path`` before scoring, keeping the best
  (lowest) rank — several chunks of one file count as one retrieval.
- Recall@k = fraction of a query's relevant items matched by at least one
  result in the top k (after dedup), averaged over queries.
- NDCG@10 uses binary gains with greedy credit: walking the deduped ranking,
  a result gains 1 only the first time it matches a not-yet-credited
  relevant item; IDCG assumes all relevant items ranked first.
"""

from __future__ import annotations

import json
import math
import time
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


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    category: str
    query: str
    relevant: tuple[RelevantItem, ...]


def load_golden(path: str | Path) -> list[GoldenQuery]:
    """Parse and validate golden.yaml. Bad labels fail loudly — a silently
    skipped query would corrupt the gate numbers."""
    with open(path, "rb") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise ValueError(f"{path}: expected a top-level 'queries' list")
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
            lines = item.get("lines")
            if lines is not None:
                if len(lines) != 2 or lines[0] > lines[1]:
                    raise ValueError(
                        f"{path}: query {qid!r} has bad lines range {lines!r}"
                    )
                lines = (int(lines[0]), int(lines[1]))
            relevant.append(RelevantItem(path=item["path"], lines=lines))
        queries.append(
            GoldenQuery(
                id=qid, category=category, query=text, relevant=tuple(relevant)
            )
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
    """Keep the best-ranked result per file (input is rank-ordered)."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        path = result.get("file_path")
        if path in seen:
            continue
        seen.add(path)
        deduped.append(result)
    return deduped


def score_query(
    results: list[dict[str, Any]],
    relevant: tuple[RelevantItem, ...],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Recall@k for each k plus NDCG@10 for one query (rules in module doc)."""
    deduped = dedupe_by_path(results)
    matched_rank: dict[int, int] = {}  # relevant index -> rank credited
    gains: list[int] = []
    for rank, result in enumerate(deduped):
        gain = 0
        for i, item in enumerate(relevant):
            if i in matched_rank:
                continue
            if matches(result, item):
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
        row[key] = (
            sum(q[key] for q in per_query) / len(per_query) if per_query else 0.0
        )
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
            cat: _mean_rows(
                [q for q in per_query if q["category"] == cat], metric_keys
            )
            for cat in CATEGORIES
        },
        "queries": per_query,
    }
    return report


def save_baseline(
    report: dict[str, Any], path: str | Path, meta: dict[str, Any]
) -> None:
    payload = {
        "meta": meta,
        "overall": report["overall"],
        "categories": report["categories"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_baseline(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


_METRICS = ("recall@5", "recall@10", f"ndcg@{NDCG_K}")


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
            row = (
                report["overall"] if cat == "overall" else report["categories"][cat]
            )
            cells = " | ".join(
                f"{row[c]:.1f}" if c in LATENCY_KEYS else f"{row[c]:.3f}"
                for c in columns
            )
            lines.append(
                f"| {cat} | {row['n_queries']} | {channel} | {cells} |"
            )
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
        base = (
            baseline["overall"] if cat == "overall" else baseline["categories"][cat]
        )
        for metric in _METRICS:
            delta = cha[metric] - base[metric]
            lines.append(
                f"| {cat} | {metric} | {base[metric]:.3f} | {cha[metric]:.3f} "
                f"| {delta:+.3f} |"
            )
    return "\n".join(lines)
