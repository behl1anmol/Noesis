"""M5 structural golden evaluation (§3.8, M5 exit criterion).

Every ``structural_patterns`` entry in golden.yaml must return exactly its
labeled per-file match counts when run against this repository. Unlike the
retrieval golden tests, this needs no model, no Qdrant and no index — the
scan reads the live working tree — so it runs in the default suite rather
than behind the ``golden`` marker: any commit that breaks a labeled pattern
fails CI immediately.

Scans are scoped to ``paths=["src"]`` (see golden.yaml) so churn in agent
tooling under .claude/ cannot flake the labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noesis.core import state
from noesis.core.structural import structural_search

from .harness import load_structural_patterns

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests" / "eval" / "golden.yaml"

PATTERNS = load_structural_patterns(GOLDEN)


@pytest.fixture()
def project(tmp_path):
    conn = state.connect(tmp_path / "state.sqlite")
    state.init_db(conn)
    pid = state.register_project(conn, REPO_ROOT, "fake-model")
    yield conn, pid
    conn.close()


@pytest.mark.parametrize("gp", PATTERNS, ids=[p.id for p in PATTERNS])
async def test_structural_pattern_exact_counts(project, gp):
    conn, pid = project
    result = await structural_search(
        conn, pid, gp.pattern, gp.language, paths=["src"]
    )
    got: dict[str, int] = {}
    for m in result["matches"]:
        got[m["file_path"]] = got.get(m["file_path"], 0) + 1
    assert got == gp.expected, (
        f"{gp.id}: pattern {gp.pattern!r} returned {got}, expected {gp.expected}"
    )
    assert result["truncated"] is False
    assert result["timed_out"] is False
