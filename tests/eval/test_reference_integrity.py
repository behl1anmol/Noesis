"""The committed reference must be able to refute a per-query claim — DEFAULT suite.

Unmarked, no model, no GPU: it reads `baselines/reference.json` and nothing else.

Why this exists (ADR-69, lesson 19). The reference used to store only aggregates.
That is enough to confirm any aggregate — and enough to refute nothing about a
single query. The PR #42 round-1 reply published a five-row per-channel table for
`structural-08` reconstructed from memory; two rows were wrong, and the mechanism
built on them was falsified on both premises. The evidence that could have caught
it lived only in `tests/eval/report_latest.json`, which is gitignored, so a reader
holding the repository could not check the claim at all.

Storing the rows is half the fix. This test is the other half: it re-derives every
stored aggregate from the stored per-query rows, so the two halves of the artifact
can never drift apart and a reader can trust either one. Had a row been edited to
match a claim, the aggregate would stop reconstructing here.

Deliberately NOT asserted: that the query ids match today's `golden.yaml`. The
reference is a *past* measurement, and a label edit is supposed to make it stale —
`reference_mismatches` already refuses to gate across a `golden_sha256` change.
Coupling this test to the live golden set would mean every label edit turned the
default suite red until someone found a GPU and spent 11 minutes, which is a
strong incentive to stop editing labels — the exact pressure issue #38 was about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .harness import CATEGORIES, _METRICS

REFERENCE_PATH = Path(__file__).resolve().parent / "baselines" / "reference.json"

REFERENCE = json.loads(REFERENCE_PATH.read_text())
CHANNELS = sorted(REFERENCE["channels"])

_MISSING_ROWS = (
    "the reference stores no per-query rows, so no per-query claim about this "
    "run can be checked against a committed artifact (ADR-69). Re-baseline with "
    "a harness that writes them: NOESIS_EVAL_REBASELINE=1 uv run pytest "
    "tests/eval/ -m golden -s"
)


def _rows(channel: str) -> list[dict]:
    """The channel's per-query rows, or a named failure — never a KeyError."""
    rows = REFERENCE["channels"][channel].get("queries")
    assert rows, f"{channel}: {_MISSING_ROWS}"
    return rows


def test_reference_records_per_query_rows():
    assert CHANNELS, f"{REFERENCE_PATH} records no channels"
    for channel in CHANNELS:
        _rows(channel)


@pytest.mark.parametrize("channel", CHANNELS)
def test_stored_aggregates_reconstruct_from_the_stored_rows(channel):
    """Every aggregate in the reference is the mean of its own per-query rows."""
    stored = REFERENCE["channels"][channel]
    rows = _rows(channel)

    scopes = {"overall": rows}
    for category in CATEGORIES:
        scopes[category] = [r for r in rows if r["category"] == category]

    for scope, subset in scopes.items():
        row = stored["overall"] if scope == "overall" else stored["categories"][scope]
        assert row["n_queries"] == len(subset), (
            f"{channel}/{scope}: n_queries is {row['n_queries']} but "
            f"{len(subset)} per-query rows carry that category"
        )
        for metric in _METRICS:
            recomputed = sum(r[metric] for r in subset) / len(subset) if subset else 0.0
            assert row[metric] == pytest.approx(recomputed, abs=1e-9), (
                f"{channel}/{scope}/{metric}: stored {row[metric]!r} is not the "
                f"mean of the stored rows ({recomputed!r}) — the aggregates and "
                f"the per-query rows in {REFERENCE_PATH.name} disagree, so one "
                f"of them was edited by hand"
            )


def test_every_channel_measured_the_same_queries():
    """Per-query comparisons across channels (the reranker delta the report
    prints) are only meaningful if the channels ran the same set."""
    id_sets = {channel: sorted(r["id"] for r in _rows(channel)) for channel in CHANNELS}
    first = id_sets[CHANNELS[0]]
    for channel, ids in id_sets.items():
        assert ids == first, f"{channel} measured a different query set than {CHANNELS[0]}"
    assert len(first) == len(set(first)), "duplicate query ids in the reference"
    assert len(first) == REFERENCE["provenance"]["labels"]["n_queries"], (
        "provenance n_queries disagrees with the number of stored rows"
    )
