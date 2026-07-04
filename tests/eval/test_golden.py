"""Golden-set evaluation (doc §6.2; M3 + M4 gates) plus harness metric tests.

The ``golden``-marked test self-indexes this repository with the real
embedder and an in-memory Qdrant, evaluates the dense / sparse / hybrid /
hybrid+rerank channels against ``tests/eval/golden.yaml``, prints the
comparison tables (quality + latency p50/p95), and writes
``tests/eval/report_latest.json``. It asserts only harness mechanics — the
gates themselves (M3: hybrid beats stored M2 dense; M4: rerank default-on
only on a measured NDCG@10 win over same-run hybrid, Finding 2) are
stakeholder decisions made from the printed numbers. The M4 comparison is
same-run by design: both channels share the corpus, models and labels, so
no stale stored baseline can skew it (lesson 3).

The unmarked tests cover the metric math with fabricated results and run in
the default suite.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.embedder import LocalSTEmbedder
from noesis.core.indexer import index_project
from noesis.core.reranker import LocalCrossEncoderReranker
from noesis.core.retriever import search_code
from noesis.core.vectorstore import VectorStore

from .harness import (
    CATEGORIES,
    LATENCY_KEYS,
    GoldenQuery,
    RelevantItem,
    dedupe_by_path,
    evaluate,
    format_delta,
    format_table,
    load_baseline,
    load_golden,
    percentile,
    save_baseline,
    score_query,
)

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN_PATH = EVAL_DIR / "golden.yaml"
BASELINE_PATH = EVAL_DIR / "baselines" / "m2_dense.json"
M3_BASELINE_PATH = EVAL_DIR / "baselines" / "m3_hybrid.json"
M4_BASELINE_PATH = EVAL_DIR / "baselines" / "m4_hybrid_rerank.json"
REPORT_PATH = EVAL_DIR / "report_latest.json"


# --- metric math (default suite, no model) ---------------------------------


def result(path: str, start: int = 1, end: int = 10) -> dict:
    return {"file_path": path, "start_line": start, "end_line": end}


def test_score_query_perfect_and_miss():
    relevant = (RelevantItem("a.py"),)
    perfect = score_query([result("a.py")], relevant)
    assert perfect == {"recall@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0}
    miss = score_query([result("b.py")], relevant)
    assert miss == {"recall@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
    empty = score_query([], relevant)
    assert empty["recall@10"] == 0.0 and empty["ndcg@10"] == 0.0


def test_score_query_rank_positions():
    # Relevant at deduped rank 5: outside @5, inside @10; NDCG discounted.
    results = [result(f"noise{i}.py") for i in range(5)] + [result("a.py")]
    scores = score_query(results, (RelevantItem("a.py"),))
    assert scores["recall@5"] == 0.0
    assert scores["recall@10"] == 1.0
    # gain 1 at rank index 5 -> dcg = 1/log2(7), idcg = 1
    assert scores["ndcg@10"] == pytest.approx(0.3562, abs=1e-3)


def test_line_overlap_required_when_lines_given():
    relevant = (RelevantItem("a.py", lines=(100, 120)),)
    assert score_query([result("a.py", 1, 50)], relevant)["recall@10"] == 0.0
    assert score_query([result("a.py", 118, 140)], relevant)["recall@10"] == 1.0
    # Touching the boundary counts as overlap.
    assert score_query([result("a.py", 90, 100)], relevant)["recall@10"] == 1.0


def test_dedupe_keeps_best_rank_and_counts_file_once():
    assert [r["file_path"] for r in dedupe_by_path(
        [result("a.py", 1, 5), result("b.py"), result("a.py", 50, 60)]
    )] == ["a.py", "b.py"]
    # Two chunks of the same relevant file are one retrieval, not two.
    relevant = (RelevantItem("a.py"), RelevantItem("b.py"))
    scores = score_query([result("a.py", 1, 5), result("a.py", 9, 20)], relevant)
    assert scores["recall@10"] == 0.5


def test_two_relevant_greedy_credit():
    relevant = (RelevantItem("a.py", lines=(1, 10)), RelevantItem("a.py", lines=(50, 60)))
    # One deduped result can credit only one of two same-file items.
    scores = score_query([result("a.py", 1, 60)], relevant)
    assert scores["recall@10"] == 0.5


async def test_evaluate_aggregates_per_category():
    golden = [
        GoldenQuery("nl-1", "nl", "q1", (RelevantItem("a.py"),)),
        GoldenQuery("sym-1", "symbol", "q2", (RelevantItem("b.py"),)),
        GoldenQuery("st-1", "structural", "q3", (RelevantItem("c.py"),)),
    ]

    async def search_fn(query: str) -> list[dict]:
        return [result("a.py")] if query == "q1" else [result("zzz.py")]

    report = await evaluate(search_fn, golden)
    assert report["categories"]["nl"]["recall@10"] == 1.0
    assert report["categories"]["symbol"]["recall@10"] == 0.0
    assert report["overall"]["n_queries"] == 3
    assert report["overall"]["recall@10"] == pytest.approx(1 / 3)


def test_percentile_nearest_rank():
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 95) == 5.0
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0  # unsorted input


async def test_evaluate_reports_latency_percentiles():
    golden = [
        GoldenQuery("nl-1", "nl", "q1", (RelevantItem("a.py"),)),
        GoldenQuery("nl-2", "nl", "q2", (RelevantItem("a.py"),)),
    ]

    async def search_fn(query: str) -> list[dict]:
        return [result("a.py")]

    report = await evaluate(search_fn, golden)
    for row in (report["overall"], report["categories"]["nl"]):
        assert row["latency_p50_ms"] > 0
        assert row["latency_p95_ms"] >= row["latency_p50_ms"]
    # Empty categories report zero latency, not a crash.
    assert report["categories"]["symbol"]["latency_p50_ms"] == 0.0


def test_load_golden_rejects_bad_entries(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text("queries:\n  - id: x\n    category: nope\n    query: q\n")
    with pytest.raises(ValueError, match="category"):
        load_golden(bad)
    bad.write_text("queries:\n  - id: x\n    category: nl\n    query: q\n")
    with pytest.raises(ValueError, match="relevant"):
        load_golden(bad)


# --- golden-set run (opt-in: uv run pytest tests/eval/ -m golden) -----------


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Self-index this repo once: real embedder, in-memory Qdrant.

    The corpus is the git-tracked tree copied to a tmp dir, minus
    ``tests/eval/`` — golden.yaml contains every query string verbatim, so
    indexing it would hand the lexical channel its own answer key and
    poison the gate numbers. Tracked-files-only also keeps the corpus
    reproducible (no local junk, honors the same set CI would see)."""
    import shutil
    import subprocess

    eval_tmp = tmp_path_factory.mktemp("eval")
    corpus_root = eval_tmp / "corpus"
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    for rel in filter(None, tracked.split("\0")):
        if rel.startswith("tests/eval/"):
            continue
        dst = corpus_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)

    conn = state.connect(eval_tmp / "state.sqlite")
    state.init_db(conn)
    embedder = LocalSTEmbedder()
    store = VectorStore(QdrantClient(":memory:"))
    store.ensure_collection(embedder)
    indexed = asyncio.run(index_project(conn, store, embedder, str(corpus_root)))
    assert indexed.chunks_written > 0
    yield store, embedder, indexed.project_id
    embedder.close()
    conn.close()


@pytest.mark.golden
@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="tests/eval/golden.yaml not present — golden set not labeled yet",
)
async def test_golden_set_gate_numbers(corpus):
    store, embedder, project_id = corpus
    golden = load_golden(GOLDEN_PATH)
    reranker = LocalCrossEncoderReranker()

    def channel_fn(channel: str, rerank: bool = False):
        async def fn(query: str) -> list[dict]:
            result = await search_code(
                store,
                embedder,
                query,
                project_id,
                top_k=10,
                channel=channel,
                reranker=reranker if rerank else None,
                rerank=rerank or None,
            )
            return result["hits"]

        return fn

    # Warm the reranker outside the timed loop: the first reranked call pays
    # the ~568M model load, which is startup cost, not per-query latency.
    await reranker.preload()

    try:
        reports = {
            "dense": await evaluate(channel_fn("dense"), golden),
            "sparse": await evaluate(channel_fn("sparse"), golden),
            "hybrid": await evaluate(channel_fn("hybrid"), golden),
            "hybrid+rerank": await evaluate(
                channel_fn("hybrid", rerank=True), golden
            ),
        }
    finally:
        reranker.close()

    baseline = load_baseline(BASELINE_PATH)
    baseline_note = ""
    if baseline is None:
        save_baseline(
            reports["dense"],
            BASELINE_PATH,
            meta={
                "embedding_model": embedder.model_id,
                "date": date.today().isoformat(),
                "milestone": "M2",
                "channel": "dense",
            },
        )
        baseline = load_baseline(BASELINE_PATH)
        baseline_note = (
            "NOTE: no stored baseline found — this dense run was recorded as "
            f"the M2 baseline at {BASELINE_PATH}.\n"
        )

    # Record both M4-gate channels from this same clean run (lesson 3:
    # baselines carry their provenance; the gate comparison itself is
    # same-run and never reads these files).
    for path, channel, milestone in (
        (M3_BASELINE_PATH, "hybrid", "M3"),
        (M4_BASELINE_PATH, "hybrid+rerank", "M4"),
    ):
        save_baseline(
            reports[channel],
            path,
            meta={
                "embedding_model": embedder.model_id,
                "reranker_model": reranker.model_id,
                "date": date.today().isoformat(),
                "milestone": milestone,
                "channel": channel,
            },
        )

    REPORT_PATH.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")

    print("\n\n== Channel comparison (quality + latency) ==")
    print(format_table(reports))
    print("\n== M3 gate: hybrid vs stored M2 dense baseline ==")
    print(baseline_note + format_delta(reports["hybrid"], baseline))
    print("\n== M4 gate: hybrid+rerank vs same-run hybrid (Finding 2) ==")
    print(
        format_delta(
            reports["hybrid+rerank"],
            reports["hybrid"],
            challenger_label="hybrid+rerank",
            baseline_label="hybrid (same run)",
        )
    )

    # Harness mechanics only — the gate decisions are made from the numbers.
    for report in reports.values():
        assert set(report["categories"]) == set(CATEGORIES)
        for row in (report["overall"], *report["categories"].values()):
            for key, value in row.items():
                if key == "n_queries" or key in LATENCY_KEYS:
                    assert value >= 0
                else:
                    assert 0.0 <= value <= 1.0
    assert reports["hybrid+rerank"]["overall"]["n_queries"] == len(golden)
    assert reports["hybrid"]["overall"]["n_queries"] == len(golden)
