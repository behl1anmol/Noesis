"""Settings parsing — defaults with no file, per-key fallback, M4 [reranker]."""

from __future__ import annotations

from noesis.core.config import RerankerSettings, load_settings


def test_defaults_without_config_file(tmp_path):
    settings = load_settings(tmp_path / "missing.toml")
    assert settings.reranker == RerankerSettings()
    assert settings.reranker.enabled is False  # pre-gate default (Finding 2)
    assert settings.reranker.candidates == 50


def test_reranker_section_parsed(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[reranker]\n"
        "enabled = true\n"
        "preload = true\n"
        "candidates = 25\n"
        "batch_size = 8\n"
        'model = "some/other-reranker"\n'
    )
    reranker = load_settings(cfg).reranker
    assert reranker.enabled is True
    assert reranker.preload is True
    assert reranker.candidates == 25
    assert reranker.batch_size == 8
    assert reranker.model == "some/other-reranker"


def test_partial_reranker_section_keeps_defaults(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[reranker]\nenabled = true\n")
    reranker = load_settings(cfg).reranker
    assert reranker.enabled is True
    assert reranker.candidates == 50
    assert reranker.model == "BAAI/bge-reranker-v2-m3"
