"""The label migration must stay re-derivable (ADR-64, PR #42 review).

``migrate_labels.py`` is committed on the claim that anyone can re-derive the
46 anchors instead of trusting a hand-edited YAML. That claim decayed
silently: the script re-parsed the ``golden.yaml`` it had itself rewritten and
died on ``item["lines"]``, so as shipped it re-derived nothing. A claim nobody
executes is a claim that rots — the same lesson as the labels themselves.

So the claim is executed here. Both inputs are pinned to commits, which makes
the output a fixed artifact rather than something that drifts with the tree.

This runs in the DEFAULT suite: no model, no Qdrant, two ``git show`` calls.
It skips when the objects are unreachable, because ``actions/checkout@v4``
clones at depth 1 — the alternative is a test that reddens CI for a reason
that has nothing to do with the code under review. The trade-off is stated
rather than hidden: on a shallow clone this pins nothing.
"""

from __future__ import annotations

import subprocess

import pytest

from .migrate_labels import (
    GOLDEN_REL,
    MIGRATION_BASE,
    MIGRATION_COMMIT,
    REPO_ROOT,
    blob_at,
    compute_anchors,
    pre_migration_golden,
    rewrite,
)

# The migration derived 43 labels mechanically and 3 from OVERRIDES.
EXPECTED_LABELS = 46


def _reachable(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


HAS_HISTORY = _reachable(MIGRATION_BASE) and _reachable(MIGRATION_COMMIT)

pytestmark = pytest.mark.skipif(
    not HAS_HISTORY,
    reason=(
        f"needs commits {MIGRATION_BASE} and {MIGRATION_COMMIT}; "
        f"a shallow clone does not have them"
    ),
)


def test_migration_reproduces_its_own_committed_output():
    """Re-running the migration must yield commit 3307997's golden.yaml, byte
    for byte.

    Not today's golden.yaml, deliberately: ``structural-08`` gained a second
    endpoint label in 13fcd9d and three prose anchors were hardened onto
    signature lines in review. Those are signed-off edits made ON TOP of the
    migration, so the migration reproducing them would mean it was no longer
    reproducing the migration.
    """
    source = pre_migration_golden()
    resolved, unresolved = compute_anchors(source)

    assert unresolved == [], "every label must resolve or the migration is a guess"
    assert len(resolved) == EXPECTED_LABELS

    expected = blob_at(MIGRATION_COMMIT, GOLDEN_REL)
    assert expected is not None
    assert rewrite(resolved, source) == expected


def test_reproduced_anchors_are_the_ones_the_loader_would_accept():
    """Each derived anchor occurs exactly once in its own file at MIGRATION_BASE.

    ``resolve_anchor``'s rule, applied to the migration's output — so a change
    to anchor selection that produced an ambiguous anchor would fail here
    rather than at some future ``load_golden``.
    """
    resolved, _ = compute_anchors()
    for record in resolved:
        body = blob_at(MIGRATION_BASE, record["new_path"])
        assert body is not None, record
        lines = body.splitlines()
        for key in ("anchor", "anchor_end"):
            text = record.get(key)
            if text is None:
                continue
            hits = sum(1 for line in lines if text in line)
            assert hits == 1, (
                f"{record['id']}: {key} {text!r} matches {hits} lines in "
                f"{record['new_path']} at {MIGRATION_BASE}"
            )
