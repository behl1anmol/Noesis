"""Tests for the no-network asset-readiness check (PR #50 review finding 2:
config.json alone reported ``ready`` after a download interrupted before the
weight file arrived)."""

from __future__ import annotations

from unittest.mock import patch

from noesis.prefetch import embedder_assets_ready

MODEL_ID = "nomic-ai/CodeRankEmbed"


def _cache_fake(present: dict[str, str], absent_confirmed: set[str] = frozenset()):
    """Mimics ``huggingface_hub.try_to_load_from_cache``: a cached filename
    returns its path (str); a filename HF has probed and confirmed missing
    from the repo returns a non-None sentinel object (not a str) — this is
    real observed behavior, not a guess (checked against a live HF cache:
    ``adapter_config.json`` on a repo that doesn't ship one comes back as
    ``<object object at ...>``, not ``None``); anything else (never probed)
    returns None."""
    sentinel = object()

    def fake(model_id: str, filename: str):
        assert model_id == MODEL_ID
        if filename in present:
            return present[filename]
        if filename in absent_confirmed:
            return sentinel
        return None

    return fake


def test_missing_config_is_not_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache", _cache_fake(present={})
    ):
        assert embedder_assets_ready(MODEL_ID) is False


def test_config_without_any_weight_file_is_not_ready():
    # The exact scenario PR #50 review flagged: a download interrupted after
    # metadata but before weights. Must fail against the pre-fix
    # config.json-only check for the reason stated above, not vacuously.
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(present={"config.json": "/cache/config.json"}),
    ):
        assert embedder_assets_ready(MODEL_ID) is False


def test_config_and_single_file_safetensors_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
            }
        ),
    ):
        assert embedder_assets_ready(MODEL_ID) is True


def test_config_and_single_file_pytorch_bin_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "pytorch_model.bin": "/cache/pytorch_model.bin",
            }
        ),
    ):
        assert embedder_assets_ready(MODEL_ID) is True


def test_config_and_sharded_safetensors_index_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors.index.json": "/cache/model.safetensors.index.json",
            }
        ),
    ):
        assert embedder_assets_ready(MODEL_ID) is True


def test_config_and_sharded_pytorch_index_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "pytorch_model.bin.index.json": "/cache/pytorch_model.bin.index.json",
            }
        ),
    ):
        assert embedder_assets_ready(MODEL_ID) is True


def test_confirmed_absent_sharded_index_is_not_mistaken_for_present():
    # A single-file repo whose loader once probed for the sharded-index name
    # and got a real 404: try_to_load_from_cache returns a sentinel object,
    # not None. An `is not None` check would wrongly treat that as cached.
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={"config.json": "/cache/config.json"},
            absent_confirmed={
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
                "model.safetensors",
                "pytorch_model.bin",
            },
        ),
    ):
        assert embedder_assets_ready(MODEL_ID) is False
