"""Golden label integrity (ADR-64, issue #38 Finding C) — DEFAULT suite.

Unmarked on purpose. This is the whole point of the change: the retrieval
labels in ``golden.yaml`` had no check, and rotted to 22 of 46 broken over a
month, while ``structural_patterns`` in the same file — pinned by
``test_structural_golden.py``, which is also unmarked — was correctly updated
by three separate commits. Maintenance follows enforcement, not intent. A
label check that only ran behind ``-m golden`` would inherit exactly the
blindness it exists to remove, because that tier runs neither in CI nor
unattended.

No model, no Qdrant, no index: it reads the live working tree, so it costs
milliseconds and fails on the pull request that breaks a label.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noesis.core.languages import detect_language

from .harness import CATEGORIES, load_golden

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
GOLDEN = EVAL_DIR / "golden.yaml"

# Import-time load is deliberate, matching test_structural_golden.py: an
# unresolvable anchor fails COLLECTION loudly rather than green-skipping the
# labels the gate numbers are computed from.
QUERIES = load_golden(GOLDEN, REPO_ROOT)

LABELS = [(q, item) for q in QUERIES for item in q.relevant]

# Pinned because a silent change to the set's shape invalidates every stored
# reference and every number quoted in docs/internals/evaluation.md.
EXPECTED_COUNTS = {"nl": 14, "symbol": 14, "structural": 12}


def test_golden_set_shape():
    assert len(QUERIES) == sum(EXPECTED_COUNTS.values())
    counts = {cat: sum(1 for q in QUERIES if q.category == cat) for cat in CATEGORIES}
    assert counts == EXPECTED_COUNTS


@pytest.mark.parametrize(
    ("query", "item"), LABELS, ids=[f"{q.id}:{i.path}" for q, i in LABELS]
)
def test_label_resolves_inside_its_file(query, item):
    """The anchor is still on the line it resolved to — on a FRESH read.

    This used to assert ``1 <= start <= end <= total``, which cannot fail:
    ``resolve_anchor`` returns ``i + 1`` over that same file's ``splitlines()``
    and ``load_golden`` already rejects ``end < start``, so 47 parametrized
    cases carried no coverage (PR #42 review).

    Re-reading the file and locating the anchor is falsifiable. It covers a
    resolution bug, and it covers the one real gap the bounds form had:
    ``load_golden`` caches each file's body at COLLECTION time, so a tree that
    changes between collection and execution would otherwise go unnoticed.
    """
    body = (
        (REPO_ROOT / item.path).read_text(encoding="utf-8", errors="replace").splitlines()
    )
    start, end = item.lines
    # Bounds first, so a file that shrank names the label instead of raising a
    # bare IndexError from the line below.
    assert 1 <= start <= end <= len(body), (
        f"{query.id}: anchor {item.anchor!r} resolved to [{start},{end}] but "
        f"{item.path} now has {len(body)} lines"
    )
    assert item.anchor in body[start - 1], (
        f"{query.id}: anchor {item.anchor!r} resolved to {item.path}:{start}, "
        f"but that line now reads {body[start - 1]!r}"
    )


@pytest.mark.parametrize(
    ("query", "item"), LABELS, ids=[f"{q.id}:{i.path}" for q, i in LABELS]
)
def test_label_points_at_a_file_the_corpus_indexes(query, item):
    """A label the corpus cannot reach scores 0 forever, whatever retrieval does.

    This is the general form of nl-10's failure. Both conditions are the
    corpus fixture's own rules: it skips ``tests/eval/`` so golden.yaml cannot
    hand the lexical channel its own answer key, and discovery only indexes
    files with a detected language.
    """
    assert not item.path.startswith("tests/eval/"), (
        f"{query.id}: labels a file the corpus deliberately excludes"
    )
    assert detect_language(item.path) is not None, (
        f"{query.id}: labels {item.path}, which discovery does not index"
    )


@pytest.mark.parametrize(
    "query",
    [q for q in QUERIES if q.category == "symbol"],
    ids=[q.id for q in QUERIES if q.category == "symbol"],
)
def test_symbol_query_identifier_lives_inside_its_label(query):
    """For an identifier lookup, the identifier must be in the labeled span.

    The second, independent route that caught the rot in issue #38, made
    permanent: symbol-02/05/06/07/12 all had their identifier outside their
    labeled range, and this assertion would have gone red on the commit that
    moved each one rather than two milestones later.
    """
    identifier = query.query.strip()
    for item in query.relevant:
        body = (REPO_ROOT / item.path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        start, end = item.lines
        span = body[start - 1 : end]
        assert any(identifier in line for line in span), (
            f"{query.id}: {identifier!r} does not appear in {item.path} "
            f"[{start},{end}] (anchor {item.anchor!r}) — the label no longer "
            f"points at the definition the query asks for"
        )
