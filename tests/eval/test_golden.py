"""Golden-set evaluation (doc §6.2; M3 + M4 gates) plus harness metric tests.

The ``golden``-marked test self-indexes this repository with the real
embedder and Qdrant (embedded by default, a real server under ADR-66),
evaluates five channels against ``tests/eval/golden.yaml``, and writes
``report_latest.json`` plus a rendered ``report_latest.md``.

It now **gates** (ADR-65), in three layers:

1. Relational, same-run: M3's exit criterion (hybrid beats dense on
   Recall@10, especially the symbol subset) and ADR-35's rerank NDCG claim.
   Same corpus, labels and models, so corpus growth cannot skew it.
2. Regression against one living ``baselines/reference.json``, and only when
   that reference's provenance says it is comparable — same models, same
   store, same questions, corpus within a band. Otherwise the run fails with
   the re-baseline command and the tables are marked NOT A GATE.
3. No run writes a baseline implicitly. ``NOESIS_EVAL_REBASELINE=1`` is the
   only write path, it refuses a dirty tree, and layer 1 still asserts.

Before issue #38 the assertions here were harness mechanics only, so a run
where every metric had halved reported ``1 passed``; the actual gate was a
human reading a table that the documented ``-q`` invocation suppresses, in a
tier no CI job runs. None of these layers run in CI either — the cheap part,
label integrity, lives in ``test_golden_labels.py`` in the default suite.

The unmarked tests here cover the metric math and the gate logic itself with
fabricated reports, and run in the default suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from noesis.core import state
from noesis.core.discovery import discover_files
from noesis.core.embedder import LocalSTEmbedder
from noesis.core.indexer import IndexResult, index_project
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
    golden_digest,
    load_golden,
    load_reference,
    manifest_sha256,
    percentile,
    reference_mismatches,
    regression_failures,
    relational_failures,
    render_report,
    save_reference,
    score_query,
    zero_recall_queries,
)

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN_PATH = EVAL_DIR / "golden.yaml"
REFERENCE_PATH = EVAL_DIR / "baselines" / "reference.json"
REPORT_PATH = EVAL_DIR / "report_latest.json"
REPORT_MD_PATH = EVAL_DIR / "report_latest.md"

# ADR-65 layer 3. An environment variable rather than a pytest flag, for one
# property worth having: `.claude/commands/eval.md` may only run
# `Bash(uv run pytest *:*)`, which cannot carry an env prefix — so /eval is
# structurally incapable of re-baselining, and recording a new measurement
# standard stays a deliberate human act with a decision row behind it.
REBASELINE_ENV = "NOESIS_EVAL_REBASELINE"
# ADR-66. Unset -> embedded Qdrant, exactly as before.
QDRANT_URL_ENV = "NOESIS_EVAL_QDRANT_URL"


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
    assert [
        r["file_path"]
        for r in dedupe_by_path(
            [result("a.py", 1, 5), result("b.py"), result("a.py", 50, 60)]
        )
    ] == ["a.py", "b.py"]
    # Two chunks of the same relevant file are one retrieval, not two.
    relevant = (RelevantItem("a.py"), RelevantItem("b.py"))
    scores = score_query([result("a.py", 1, 5), result("a.py", 9, 20)], relevant)
    assert scores["recall@10"] == 0.5


def test_two_relevant_greedy_credit():
    relevant = (
        RelevantItem("a.py", lines=(1, 10)),
        RelevantItem("a.py", lines=(50, 60)),
    )
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


def _golden_yaml(body: str) -> str:
    return "queries:\n  - id: x\n    category: nl\n    query: q\n" + body


def test_load_golden_rejects_bad_entries(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text("queries:\n  - id: x\n    category: nope\n    query: q\n")
    with pytest.raises(ValueError, match="category"):
        load_golden(bad, tmp_path)
    bad.write_text("queries:\n  - id: x\n    category: nl\n    query: q\n")
    with pytest.raises(ValueError, match="relevant"):
        load_golden(bad, tmp_path)


def test_load_golden_resolves_anchors(tmp_path):
    (tmp_path / "a.py").write_text("import os\n\n\ndef target():\n    return 1\n")
    good = tmp_path / "golden.yaml"
    good.write_text(
        _golden_yaml(
            '    relevant:\n      - path: a.py\n        anchor: "def target():"\n'
        )
    )
    item = load_golden(good, tmp_path)[0].relevant[0]
    assert item.lines == (4, 4)  # anchor alone -> a one-line span

    good.write_text(
        _golden_yaml(
            "    relevant:\n      - path: a.py\n"
            '        anchor: "def target():"\n        anchor_end: "return 1"\n'
        )
    )
    assert load_golden(good, tmp_path)[0].relevant[0].lines == (4, 5)


def test_load_golden_rejects_unresolvable_anchors(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\ndef target():\n    return 1\n")
    bad = tmp_path / "golden.yaml"

    # ADR-64's whole point: a label that stopped matching raises instead of
    # silently scoring its query 0 forever (issue #38, Finding C).
    bad.write_text(
        _golden_yaml('    relevant:\n      - path: a.py\n        anchor: "gone()"\n')
    )
    with pytest.raises(ValueError, match="matches no line"):
        load_golden(bad, tmp_path)

    bad.write_text(
        _golden_yaml('    relevant:\n      - path: a.py\n        anchor: "x = 1"\n')
    )
    with pytest.raises(ValueError, match="matches 2 lines"):
        load_golden(bad, tmp_path)

    bad.write_text(
        _golden_yaml(
            "    relevant:\n      - path: a.py\n"
            '        anchor: "def target():"\n        anchor_end: "x = 1"\n'
        )
    )
    with pytest.raises(ValueError, match="matches 2 lines"):
        load_golden(bad, tmp_path)

    bad.write_text(
        _golden_yaml('    relevant:\n      - path: missing.py\n        anchor: "z"\n')
    )
    with pytest.raises(ValueError, match="does not exist"):
        load_golden(bad, tmp_path)

    # The rotting form is rejected outright rather than tolerated: there is one
    # golden.yaml and it is migrated in the same commit, so accepting `lines:`
    # would only keep the loaded gun on the table.
    bad.write_text(
        _golden_yaml("    relevant:\n      - path: a.py\n        lines: [1, 2]\n")
    )
    with pytest.raises(ValueError, match="anchor"):
        load_golden(bad, tmp_path)


def test_load_golden_rejects_reversed_anchor_end(tmp_path):
    (tmp_path / "a.py").write_text("first = 1\ndef target():\n    return 1\n")
    bad = tmp_path / "golden.yaml"
    bad.write_text(
        _golden_yaml(
            "    relevant:\n      - path: a.py\n"
            '        anchor: "def target():"\n        anchor_end: "first = 1"\n'
        )
    )
    with pytest.raises(ValueError, match="before"):
        load_golden(bad, tmp_path)


# --- gate logic (default suite, fabricated reports, no model) --------------


def _row(n: int, r5: float, r10: float, ndcg: float) -> dict:
    return {
        "n_queries": n,
        "recall@5": r5,
        "recall@10": r10,
        f"ndcg@{10}": ndcg,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 2.0,
    }


def _report(r5: float, r10: float, ndcg: float, symbol_r10: float | None = None) -> dict:
    categories = {
        "nl": _row(14, r5, r10, ndcg),
        "symbol": _row(14, r5, symbol_r10 if symbol_r10 is not None else r10, ndcg),
        "structural": _row(12, r5, r10, ndcg),
    }
    return {"overall": _row(40, r5, r10, ndcg), "categories": categories}


def _provenance(**overrides) -> dict:
    base = {
        "corpus": {"chunks": 5000, "files_indexed": 184},
        "models": {
            "embedding_model": "nomic-ai/CodeRankEmbed",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        },
        "store": {"kind": "embedded", "server_version": None},
        "labels": {"golden_sha256": "abc", "n_queries": 40},
    }
    base.update(overrides)
    return base


# The numbers issue #38 actually observed: a full golden run whose metrics had
# roughly halved against the stored M2 baseline, which the old gate passed.
_DEGRADED = {
    "dense": _report(0.3750, 0.4125, 0.2992),
    "hybrid": _report(0.4375, 0.5000, 0.3404),
    "hybrid+rerank": _report(0.5000, 0.5875, 0.4196),
}
_M2_REFERENCE_CHANNELS = {"dense": _report(0.6500, 0.7375, 0.6149)}


def test_regression_gate_catches_the_halving_that_shipped_green():
    """The keystone: the exact run issue #38 was filed about must now FAIL.

    Without this the new gate would be exactly as unverified as the one it
    replaces (lesson 2: never assert before seeing the losing outcome).
    """
    failures = regression_failures(_DEGRADED, _M2_REFERENCE_CHANNELS)
    assert failures, "the gate must reject a run where every metric halved"
    assert any("dense/overall/recall@10" in f for f in failures)


def test_relational_gate_alone_would_have_passed_the_halving():
    """Why layer 2 is not redundant, stated as a test rather than a comment.

    In that same degraded run dense 0.4125 < hybrid 0.5000 < rerank, so every
    relational claim still held. Uniform degradation is invisible to a
    same-run comparison by construction — which is exactly the blind spot the
    provenance-guarded reference exists to cover.
    """
    assert relational_failures(_DEGRADED) == []


def test_relational_gate_catches_hybrid_losing_to_dense():
    reports = {
        "dense": _report(0.65, 0.74, 0.61),
        "hybrid": _report(0.60, 0.70, 0.58),
        "hybrid+rerank": _report(0.62, 0.72, 0.60),
    }
    failures = relational_failures(reports)
    assert any("must beat dense" in f for f in failures)


def test_relational_gate_catches_symbol_subset_and_rerank_losses():
    symbol_loss = {
        "dense": _report(0.65, 0.74, 0.61, symbol_r10=0.90),
        "hybrid": _report(0.66, 0.75, 0.62, symbol_r10=0.80),
        "hybrid+rerank": _report(0.67, 0.76, 0.63, symbol_r10=0.80),
    }
    assert any("symbol subset" in f for f in relational_failures(symbol_loss))

    rerank_loss = {
        "dense": _report(0.65, 0.74, 0.61),
        "hybrid": _report(0.66, 0.75, 0.62),
        "hybrid+rerank": _report(0.66, 0.75, 0.55),
    }
    assert any("ADR-35" in f for f in relational_failures(rerank_loss))


def test_regression_gate_passes_an_identical_run_and_tolerates_one_query():
    same = {"dense": _report(0.65, 0.7375, 0.6149)}
    assert regression_failures(same, same) == []

    # One query of 40 is 0.025 absolute. A relative band alone would call that
    # a 3.4% drop and pass it; the absolute floor is what makes "one query is
    # noise" true regardless of the score level.
    one_query_worse = {"dense": _report(0.65, 0.7375 - 0.025, 0.6149)}
    assert regression_failures(one_query_worse, same) == []

    three_queries_worse = {"dense": _report(0.65, 0.7375 - 0.075, 0.6149)}
    assert regression_failures(three_queries_worse, same)


def test_regression_gate_absolute_floor_holds_at_low_scores():
    """The bug a purely relative band would have: at a low baseline, one
    query's movement exceeds any sane relative percentage."""
    low = {"dense": _report(0.20, 0.25, 0.20)}
    # 1/12 = 0.083 on the smallest category — a 33% relative drop, but still
    # one query.
    one_query = {"dense": _report(0.20, 0.25, 0.20)}
    one_query["dense"]["categories"]["structural"]["recall@10"] = 0.25 - (1 / 12)
    assert regression_failures(one_query, low) == []


def test_reference_mismatches_names_every_incomparability():
    run = _provenance()
    assert reference_mismatches(run, run) == []

    other_model = _provenance(
        models={"embedding_model": "other", "reranker_model": "BAAI/bge-reranker-v2-m3"}
    )
    assert any("embedding_model" in r for r in reference_mismatches(other_model, run))

    server = _provenance(store={"kind": "server", "server_version": "1.18.3"})
    assert any("store kind" in r for r in reference_mismatches(server, run))

    relabeled = _provenance(labels={"golden_sha256": "zzz", "n_queries": 40})
    assert any("golden.yaml changed" in r for r in reference_mismatches(relabeled, run))

    grown = _provenance(corpus={"chunks": 10000, "files_indexed": 380})
    reasons = reference_mismatches(grown, run)
    assert any("corpus drift" in r and "files indexed" in r for r in reasons)


def test_reference_mismatches_tolerates_ordinary_corpus_growth():
    run = _provenance(corpus={"chunks": 5500, "files_indexed": 195})
    assert reference_mismatches(_provenance(), run) == []


def test_zero_recall_queries_flags_only_universal_misses():
    def with_queries(*pairs):
        report = _report(0.5, 0.5, 0.5)
        report["queries"] = [{"id": qid, "recall@10": r} for qid, r in pairs]
        return report

    reports = {
        "dense": with_queries(("a", 0.0), ("b", 0.0), ("c", 1.0)),
        "hybrid": with_queries(("a", 0.0), ("b", 1.0), ("c", 1.0)),
    }
    assert zero_recall_queries(reports) == ["a"]


def test_render_report_marks_an_ungated_run():
    reports = {"dense": _report(0.6, 0.7, 0.6)}
    rendered = render_report(
        reports, _provenance(), [], ["no reference at x"], None
    )
    assert "NOT A GATE" in rendered
    assert "NOT RUN" in rendered

    gated = render_report(reports, _provenance(), [], [], [])
    assert "NOT A GATE" not in gated
    assert "L2 regression vs reference: PASS" in gated


# --- golden-set run (opt-in: uv run pytest tests/eval/ -m golden) -----------


@dataclass(frozen=True)
class CorpusRun:
    """What one indexed corpus offers the gate — including the provenance
    inputs, so the gate never has to re-derive what it measured."""

    store: VectorStore
    embedder: LocalSTEmbedder
    project_id: str
    corpus_root: Path
    indexed: IndexResult
    store_meta: dict


def _server_version_or_skip(url: str) -> str:
    """Live server version, or skip when nothing is listening.

    Same shape as ``tests/test_qdrant_server_smoke.py:_server_version_or_skip``
    and for the same reason — a plain REST GET, because the client runs its
    compatibility check on a daemon thread and anything built on catching that
    warning would race. Not imported from there: that module is ``server``-
    marked and keys off ``QDRANT_URL``, while this tier is opt-in through its
    own variable so a developer who merely has a server running does not
    silently get their golden numbers measured somewhere else.
    """
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — any failure means "not available"
        pytest.skip(f"no Qdrant server at {url} ({exc}) — `docker compose up -d`")
    version = response.json().get("version")
    if not version:
        pytest.fail(f"{url} answered but reported no version: {response.text!r}")
    return version


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Self-index this repo once: real embedder, embedded or real Qdrant.

    The corpus is the git-tracked tree copied to a tmp dir, minus
    ``tests/eval/`` — golden.yaml contains every query string verbatim, so
    indexing it would hand the lexical channel its own answer key and
    poison the gate numbers. Tracked-files-only also keeps the corpus
    reproducible (no local junk, honors the same set CI would see).

    ADR-66: with ``NOESIS_EVAL_QDRANT_URL`` set the run measures against a real
    server on a throwaway collection instead of ``:memory:``. ADR-62 left that
    open deliberately — the embedded client is a Python reimplementation of the
    query semantics, and sparse IDF plus RRF fusion are computed server-side,
    so ranking can move between the two with nothing failing. The store is
    recorded in provenance and is a hard comparability term, so numbers from
    one store can never gate the other.
    """
    import shutil

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

    qdrant_url = os.environ.get(QDRANT_URL_ENV)
    if qdrant_url:
        server_version = _server_version_or_skip(qdrant_url)
        # Unique, dropped afterwards: this talks to whatever server the
        # developer is running, which is very likely their real index.
        collection = f"noesis_eval_{uuid.uuid4().hex[:12]}"
        client = QdrantClient(url=qdrant_url)
        store = VectorStore(client, collection_name=collection)
        store_meta = {"kind": "server", "server_version": server_version}
    else:
        client = QdrantClient(":memory:")
        store = VectorStore(client)
        store_meta = {"kind": "embedded", "server_version": None}

    conn = state.connect(eval_tmp / "state.sqlite")
    state.init_db(conn)
    embedder = LocalSTEmbedder()
    try:
        store.ensure_collection(embedder)
        indexed = asyncio.run(index_project(conn, store, embedder, str(corpus_root)))
        assert indexed.chunks_written > 0
        yield CorpusRun(
            store=store,
            embedder=embedder,
            project_id=indexed.project_id,
            corpus_root=corpus_root,
            indexed=indexed,
            store_meta=store_meta,
        )
    finally:
        embedder.close()
        conn.close()
        if qdrant_url:
            client.delete_collection(collection_name=store.collection_name)
        client.close()


@pytest.mark.golden
@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason="tests/eval/golden.yaml not present — golden set not labeled yet",
)
async def test_golden_set_gate_numbers(corpus):
    """The gate (ADR-65). Three layers, and every one of them asserts.

    Before this, the only assertions here were harness mechanics — categories
    present, values inside [0,1] — so a run where every metric had roughly
    halved reported ``1 passed`` (issue #38, Finding A). The real gate was a
    human reading a printed table that the documented ``-q`` invocation
    suppresses, in a tier no CI job runs.
    """
    store, embedder, project_id = corpus.store, corpus.embedder, corpus.project_id
    rebaseline = os.environ.get(REBASELINE_ENV) == "1"
    # Tracked changes only. The corpus fixture builds from `git ls-files`, so
    # untracked files are not in the measurement and cannot affect whether it
    # reproduces — counting them would block a re-baseline over a stray .idea/
    # directory while proving nothing.
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    # Checked before the two-hour measurement, not after it: a reference that
    # cannot be reproduced from a commit is not a measurement standard, and
    # learning that at the end would waste the whole run.
    if rebaseline and dirty:
        pytest.fail(
            "refusing to record a reference from a dirty tree — commit first, "
            "so the numbers name a reproducible state of the corpus"
        )
    # Anchors resolve against the copy that was actually chunked, not the
    # working tree, so the line numbers describe the thing being measured.
    golden = load_golden(GOLDEN_PATH, corpus.corpus_root)
    reranker = LocalCrossEncoderReranker()

    def channel_fn(channel: str, rerank: bool = False, language: str | None = None):
        async def fn(query: str) -> list[dict]:
            result = await search_code(
                store,
                embedder,
                query,
                project_id,
                top_k=10,
                language=language,
                channel=channel,
                reranker=reranker if rerank else None,
                rerank=rerank or None,
            )
            return result["hits"]

        return fn

    # Warm the reranker outside the timed loop: the first reranked call pays
    # the ~568M model load, which is startup cost, not per-query latency.
    await reranker.preload()

    # Record the compute device every model actually loaded on — a latency
    # benchmark without a recorded device is uninterpretable (lesson 4). The
    # embedder loaded during the corpus fixture's index; the reranker just now.
    import torch

    device = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "embedder": embedder.resolved_device,
        "reranker": reranker.resolved_device,
    }
    print(f"\n== compute device ==\n{device}")

    try:
        reports = {
            "dense": await evaluate(channel_fn("dense"), golden),
            # Diagnostic channel, reported and never gated. Every label in the
            # golden set is a .py file, so filtering to python can only remove
            # distractors and never a true positive: the dense -> dense
            # (python-only) gap separates "the haystack grew" from "retrieval
            # got worse", which is the question Finding B could not answer
            # without it. One extra pass over the already-built index.
            "dense (python-only)": await evaluate(
                channel_fn("dense", language="python"), golden
            ),
            "sparse": await evaluate(channel_fn("sparse"), golden),
            "hybrid": await evaluate(channel_fn("hybrid"), golden),
            "hybrid+rerank": await evaluate(channel_fn("hybrid", rerank=True), golden),
        }

        # Assert the filter's contract rather than a ranking relation. The
        # monotonicity you would expect (python-only recall >= unfiltered) is
        # true in principle but can flip on a tie-break, and a flaky assertion
        # on a two-hour run is worse than none. This cannot flake, and it is
        # what catches the filter being dropped from one prefetch.
        filtered = await search_code(
            store, embedder, golden[0].query, project_id, top_k=10, language="python"
        )
        assert filtered["hits"], "python-only filter returned nothing at all"
        assert {hit["language"] for hit in filtered["hits"]} == {"python"}
    finally:
        reranker.close()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    indexed = corpus.indexed
    provenance = {
        "corpus": {
            "files_total": indexed.files_total,
            "files_indexed": indexed.files_indexed,
            "chunks": indexed.chunks_written,
            "commit": head,
            "dirty": dirty,
            "manifest_sha256": manifest_sha256(discover_files(str(corpus.corpus_root))),
        },
        "models": {
            "embedding_model": embedder.model_id,
            "reranker_model": reranker.model_id,
        },
        "store": corpus.store_meta,
        "labels": {
            "golden_sha256": golden_digest(GOLDEN_PATH),
            "n_queries": len(golden),
        },
        "device": device,
        "date": date.today().isoformat(),
    }

    reference = load_reference(REFERENCE_PATH)
    if reference is None:
        mismatches = [f"no reference at {REFERENCE_PATH}"]
    else:
        mismatches = reference_mismatches(reference["provenance"], provenance)
    # A run that just wrote the reference cannot also be judged against it.
    gated = reference is not None and not mismatches and not rebaseline
    regression = regression_failures(reports, reference["channels"]) if gated else None
    relational = relational_failures(reports)

    REPORT_PATH.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    REPORT_MD_PATH.write_text(
        render_report(reports, provenance, relational, mismatches, regression)
    )

    print("\n\n== Channel comparison (quality + latency) ==")
    print(format_table(reports))
    if reference is not None:
        print("\n== hybrid vs stored reference ==")
        print(
            format_delta(
                reports["hybrid"],
                reference["channels"]["hybrid"],
                baseline_label="reference (stored)",
            )
        )
    print("\n== M4: hybrid+rerank vs same-run hybrid (Finding 2) ==")
    print(
        format_delta(
            reports["hybrid+rerank"],
            reports["hybrid"],
            challenger_label="hybrid+rerank",
            baseline_label="hybrid (same run)",
        )
    )
    print(f"\n== verdicts ==\n{REPORT_MD_PATH}")

    # Harness mechanics first: a malformed report would make every verdict
    # below meaningless.
    for report in reports.values():
        assert set(report["categories"]) == set(CATEGORIES)
        for row in (report["overall"], *report["categories"].values()):
            for key, value in row.items():
                if key == "n_queries" or key in LATENCY_KEYS:
                    assert value >= 0
                else:
                    assert 0.0 <= value <= 1.0
    assert reports["hybrid"]["overall"]["n_queries"] == len(golden)
    assert reports["hybrid+rerank"]["overall"]["n_queries"] == len(golden)

    # Layer 1 before layer 2, deliberately: a corpus change big enough to make
    # the reference incomparable must never be able to mask a fusion
    # regression. Layer 1 holds even while re-baselining — a run that fails
    # M3's own exit criterion is not a measurement standard (lesson 8).
    assert not relational, "M3/M4 relational gate failed:\n  " + "\n  ".join(relational)

    if rebaseline:
        save_reference(reports, REFERENCE_PATH, provenance)
        print(
            f"\nREBASELINED: wrote {REFERENCE_PATH}. Layer 2 was skipped for this "
            f"run. Record it: python .claude/scripts/devlog.py decision add "
            f'--title "golden reference {provenance["date"]}"'
        )
        return

    assert reference is not None and not mismatches, (
        "the stored golden reference cannot gate this run:\n  "
        + "\n  ".join(mismatches)
        + f"\nNothing was gated. The tables in {REPORT_MD_PATH} are NOT A GATE.\n"
        f"Review them, then re-baseline deliberately:\n"
        f"  {REBASELINE_ENV}=1 uv run pytest tests/eval/ -m golden"
    )
    assert not regression, "quality regressed against the stored reference:\n  " + (
        "\n  ".join(regression)
    )
