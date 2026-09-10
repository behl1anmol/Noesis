"""The plugin's standalone diagnostic script (issue #52).

``plugin/noesis/skills/noesis-mcp/scripts/healthcheck.py`` is stdlib-only and
lives outside the package, so it is loaded here by path. It had no test at all
before this file; the branch under test is the reranker one added for issue
#52, and the embedder branch it mirrors is pinned alongside it so a future
edit to either cannot silently drop the other.

Exit-code contract: 0 healthy, 1 unhealthy. "Missing assets" is unhealthy on
purpose (ADR-78) — green here previously meant nothing about whether the next
call pays a multi-minute download.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "plugin/noesis/skills/noesis-mcp/scripts/healthcheck.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("noesis_plugin_healthcheck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, health: dict, projects=None):
    """Run the script's main() against a stubbed service; returns
    (exit_code, stdout)."""
    module = _load_script()

    def fake_get(url: str, timeout: float):
        if url.endswith("/healthz"):
            return 200, health
        return 200, projects if projects is not None else []

    monkeypatch.setattr(module, "_get", fake_get)
    monkeypatch.setattr(sys, "argv", ["healthcheck.py"])
    code = 0
    try:
        module.main()
    except SystemExit as exc:  # the script's own exit path
        code = exc.code or 0
    return code, module


def _capture(capsys) -> str:
    return capsys.readouterr().out


HEALTHY_EMBEDDER = {"status": "ok", "assets": "ready", "embedder_ready": True}


def test_missing_reranker_assets_fails_the_healthcheck(monkeypatch, capsys):
    code, _ = _run(
        monkeypatch,
        {**HEALTHY_EMBEDDER, "reranker_assets": "missing", "reranker_ready": False},
    )
    out = _capture(capsys)
    assert code == 1, "reranking is on with no weights cached — must not be green"
    assert "reranker" in out.lower()
    assert "noesis.prefetch" in out


def test_disabled_reranker_is_silent_and_healthy(monkeypatch, capsys):
    # The shipped default (reranker.enabled=false). Reporting anything here
    # would be a false alarm about a feature the operator turned off.
    code, _ = _run(
        monkeypatch,
        {
            **HEALTHY_EMBEDDER,
            "reranker_assets": "disabled",
            "reranker_ready": "disabled",
        },
        projects=[{"id": "p1", "root_path": "/tmp/repo", "embedding_model": "m"}],
    )
    out = _capture(capsys)
    assert code == 0
    assert "reranker" not in out.lower()


def test_ready_reranker_is_reported_as_ok(monkeypatch, capsys):
    code, _ = _run(
        monkeypatch,
        {**HEALTHY_EMBEDDER, "reranker_assets": "ready", "reranker_ready": True},
        projects=[{"id": "p1", "root_path": "/tmp/repo", "embedding_model": "m"}],
    )
    out = _capture(capsys)
    assert code == 0
    assert "reranker" in out.lower()
    assert "[ OK ]" in out


def test_a_server_that_reports_no_reranker_fields_is_not_an_error(monkeypatch, capsys):
    # An older service (or one behind a proxy that strips fields) simply omits
    # them. Absent is not "missing" — inventing a failure there would make the
    # plugin's diagnostic wrong about a server that is fine.
    code, _ = _run(
        monkeypatch,
        dict(HEALTHY_EMBEDDER),
        projects=[{"id": "p1", "root_path": "/tmp/repo", "embedding_model": "m"}],
    )
    out = _capture(capsys)
    assert code == 0
    assert "reranker" not in out.lower()


def test_missing_embedder_assets_still_fails(monkeypatch, capsys):
    # Regression guard on the pre-existing ADR-78 branch: the reranker
    # addition must not shadow or reorder it.
    code, _ = _run(
        monkeypatch,
        {"status": "ok", "assets": "missing", "embedder_ready": False},
    )
    out = _capture(capsys)
    assert code == 1
    assert "embedding model assets" in out


def test_script_has_no_third_party_imports():
    """The script runs under a bare ``python3`` with no venv (its own
    docstring promises stdlib only) — a dependency here would make the
    diagnostic fail exactly when the environment is broken."""
    source = SCRIPT.read_text()
    for banned in ("import httpx", "import requests", "from noesis"):
        assert banned not in source
