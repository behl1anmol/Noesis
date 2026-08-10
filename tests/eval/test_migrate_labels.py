"""The label migration must stay re-derivable (ADR-64, PR #42 review).

``migrate_labels.py`` is committed on the claim that anyone can re-derive the
46 anchors instead of trusting a hand-edited YAML. That claim decayed
silently: the script re-parsed the ``golden.yaml`` it had itself rewritten and
died on ``item["lines"]``, so as shipped it re-derived nothing. A claim nobody
executes is a claim that rots — the same lesson as the labels themselves.

So the claim is executed here. Both inputs are pinned to commits, which makes
the output a fixed artifact rather than something that drifts with the tree.

This runs in the DEFAULT suite: no model, no Qdrant, two ``git show`` calls.
CI checks out with ``fetch-depth: 0`` so it actually runs there (PR #42 round-2
review: with the default depth-1 checkout it skipped on every pull request, so
the pin was inert everywhere except the author's machine — Finding C's own root
cause, maintenance following enforcement rather than intent).

**The skip condition is shallowness itself, not object reachability.** It used
to be "``git cat-file -e`` failed", which cannot tell a shallow clone from a
commit that a rebase orphaned or a hand typo mangled: flipping one hex digit of
``MIGRATION_BASE`` turned both tests into ``2 skipped`` reading *"a shallow
clone does not have them"* inside a full clone, and under ``-q`` a skip is an
anonymous ``s``. ``git rev-parse --is-shallow-repository`` answers the question
actually being asked, so a missing object in a full clone now fails loudly —
which is what hard rule 9 says a check has to be able to do.
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


def _no_history_reason() -> str | None:
    """Why this environment cannot pin the migration, or None when it can.

    Only two environments legitimately lack the objects: a shallow clone, and a
    tree that is not a git checkout at all (an sdist, an extracted archive).
    Both are stated in the skip reason in their own words. Anything else — an
    orphaned or mistyped commit — is a defect, and falls through to a real
    failure rather than being absorbed here.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return f"{REPO_ROOT} is not a git checkout, so the history is unavailable"
    if proc.stdout.strip() == "true":
        return (
            f"shallow clone: commits {MIGRATION_BASE} and {MIGRATION_COMMIT} are "
            f"not present. CI uses fetch-depth: 0 so this runs there."
        )
    return None


NO_HISTORY_REASON = _no_history_reason()

pytestmark = pytest.mark.skipif(
    NO_HISTORY_REASON is not None, reason=NO_HISTORY_REASON or ""
)


def test_migration_reproduces_its_own_committed_output():
    """Re-running the migration must yield commit 3307997's golden.yaml, byte
    for byte.

    Not today's golden.yaml, deliberately: ``structural-08`` gained a second
    endpoint label in 13fcd9d, three prose anchors were hardened onto signature
    lines in review round 1, and ``nl-09``'s ``anchor_end`` was moved off an
    indent-unique fragment in round 2. Those are signed-off edits made ON TOP of
    the migration, so the migration reproducing them would mean it was no longer
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
