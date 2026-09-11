"""Tests for the no-network asset-readiness check (PR #50 review finding 2:
config.json alone reported ``ready`` after a download interrupted before the
weight file arrived).

Issue #52 renamed ``embedder_assets_ready``/``embedder_readiness`` to
``model_assets_ready``/``model_readiness``: the reranker asks the same
question of the same HF cache, and a function the reranker calls should not
claim in its name to be about the embedder.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from noesis.prefetch import model_assets_ready, model_readiness, reranker_readiness

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
    with patch("huggingface_hub.try_to_load_from_cache", _cache_fake(present={})):
        assert model_assets_ready(MODEL_ID) is False


def test_config_without_any_weight_file_is_not_ready():
    # The exact scenario PR #50 review flagged: a download interrupted after
    # metadata but before weights. Must fail against the pre-fix
    # config.json-only check for the reason stated above, not vacuously.
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(present={"config.json": "/cache/config.json"}),
    ):
        assert model_assets_ready(MODEL_ID) is False


def test_config_and_single_file_safetensors_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "tokenizer.json": "/cache/tokenizer.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_config_and_single_file_pytorch_bin_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "pytorch_model.bin": "/cache/pytorch_model.bin",
                "tokenizer.json": "/cache/tokenizer.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_config_and_sharded_safetensors_index_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors.index.json": "/cache/model.safetensors.index.json",
                "tokenizer.json": "/cache/tokenizer.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_config_and_sharded_pytorch_index_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "pytorch_model.bin.index.json": "/cache/pytorch_model.bin.index.json",
                "tokenizer.json": "/cache/tokenizer.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


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
        assert model_assets_ready(MODEL_ID) is False


# --- embedder_readiness: the shared (assets, embedder_ready) computation ----
#
# PR #50 round-5 review: this was pasted verbatim into both routes.py's
# healthz() and jobs.py's index_status(), agreeing only because the text was
# identical and one test compared the two endpoints' output -- not because
# there was one implementation. Extracted here so both call it instead.


async def test_embedder_readiness_reports_ready_with_resolved_device():
    embedder = SimpleNamespace(model_id=MODEL_ID, resolved_device="cpu")
    with patch("noesis.prefetch.model_assets_ready", return_value=True):
        assets, embedder_ready = await model_readiness(embedder)
    assert assets == "ready"
    assert embedder_ready is True


async def test_embedder_readiness_reports_missing_and_na_without_resolved_device():
    # FakeEmbedder (and any embedder that never loaded) has no
    # resolved_device attribute at all -- must read "n/a", not raise or
    # default to a real ready/missing verdict.
    embedder = SimpleNamespace(model_id=MODEL_ID)
    with patch("noesis.prefetch.model_assets_ready", return_value=False):
        assets, embedder_ready = await model_readiness(embedder)
    assert assets == "missing"
    assert embedder_ready == "n/a"


async def test_embedder_readiness_reports_false_once_device_resolved():
    # resolved_device set but falsy (e.g. a loaded-then-failed state some
    # future embedder might report) must come back as the bool False, not
    # the "n/a" sentinel -- "n/a" means "never attempted", not "not ready".
    embedder = SimpleNamespace(model_id=MODEL_ID, resolved_device=False)
    with patch("noesis.prefetch.model_assets_ready", return_value=True):
        _, embedder_ready = await model_readiness(embedder)
    assert embedder_ready is False


def test_a_local_model_directory_does_not_take_the_health_surface_down(tmp_path):
    """``[embedder] model`` is free text and sentence-transformers accepts a
    local directory, which is not a hub repo id.

    ``try_to_load_from_cache`` raises ``HFValidationError`` for one, and that
    propagated out of ``/healthz``, ``GET /projects/{id}/status`` and the
    ``get_index_status`` MCP tool — so pinning a local model took down the
    very surface ADR-78 added to keep the health check honest. Watched
    failing against the unguarded call: ``HFValidationError: Repo id must be
    in the form 'repo_name' or 'namespace/repo_name'``.

    For a directory that exists the answer needs no hub lookup at all: the
    assets ARE that directory. Anything else malformed reports missing, the
    fail-safe direction ADR-79 already chose.
    """
    local_model = tmp_path / "coderank"
    local_model.mkdir()
    # No exception is the property under test. The VERDICT for a directory was
    # ADR-79's "the assets ARE that directory" — an empty one answered True —
    # until issue #52's round-6 review: a half-copied directory then reported
    # `ready` on a surface that had begun promising config + weights +
    # vocabulary. It is now held to the same three requirements as a cached
    # hub repo (see the local-directory tests below), so this bare directory
    # answers False.
    assert model_assets_ready(str(local_model)) is False

    # A path that is not there is "missing", not a crash and not a false ready.
    assert model_assets_ready(str(tmp_path / "absent")) is False


# --- reranker_readiness: the same question, plus the kill switch (issue #52) --
#
# ``AppContext.reranker`` is None whenever ``reranker.enabled=false`` (the
# shipped default), and "off" is a different answer from "on but missing its
# weights". Both /healthz and jobs.index_status need that distinction, so it
# lives here rather than being re-derived at each call site — the exact
# duplication ADR-82 had to unwind for the embedder half.

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


async def test_reranker_readiness_reports_disabled_when_no_reranker_wired():
    # Must not touch the HF cache at all: with no reranker there is no model
    # id to ask about, and "missing" would read as a fetchable problem.
    with patch("noesis.prefetch.model_assets_ready") as probe:
        assets, ready = await reranker_readiness(None)
    assert assets == "disabled"
    assert ready == "disabled"
    probe.assert_not_called()


async def test_reranker_readiness_asks_about_the_reranker_model_not_the_embedder():
    seen: list[str] = []

    def fake(model_id: str) -> bool:
        seen.append(model_id)
        return True

    reranker = SimpleNamespace(model_id=RERANKER_MODEL_ID, resolved_device="cuda")
    with patch("noesis.prefetch.model_assets_ready", fake):
        assets, ready = await reranker_readiness(reranker)
    assert seen == [RERANKER_MODEL_ID]
    assert assets == "ready"
    assert ready is True


async def test_reranker_readiness_is_not_ready_before_the_model_loads():
    # The cold-start case issue #52 exists for: weights cached, model not
    # loaded yet (reranker.preload=false), so the next reranked search still
    # pays the multi-minute construction. Assets ready, reranker NOT ready.
    reranker = SimpleNamespace(model_id=RERANKER_MODEL_ID, resolved_device=None)
    with patch("noesis.prefetch.model_assets_ready", return_value=True):
        assets, ready = await reranker_readiness(reranker)
    assert assets == "ready"
    assert ready is False


async def test_reranker_readiness_reports_na_for_a_reranker_without_a_device():
    # FakeReranker has no resolved_device attribute at all — same "not
    # applicable to this implementation" contract the embedder half uses.
    reranker = SimpleNamespace(model_id="fake-reranker-v1")
    with patch("noesis.prefetch.model_assets_ready", return_value=False):
        assets, ready = await reranker_readiness(reranker)
    assert assets == "missing"
    assert ready == "n/a"


# --- prefetch fetches the models the SERVICE will load (issue #52 review) ----
#
# The plugin's healthcheck answers `reranker_assets: "missing"` with "run
# `uv run python -m noesis.prefetch`". That remedy was wrong for anyone with a
# non-default `[embedder] model` or `[reranker] model`: main() hardcoded the
# two default repo ids and never read config.toml, so the operator downloaded
# a multi-GB model the service never loads and the health check stayed red.


def _patched_main(monkeypatch, argv: list[str], expect_zero: bool = True):
    """Run prefetch.main() with every network-touching step stubbed out;
    returns the model ids it asked for."""
    import noesis.prefetch as prefetch

    called: dict[str, str] = {}
    monkeypatch.setattr(prefetch, "prefetch_grammars", lambda: [])
    monkeypatch.setattr(prefetch, "prefetch_bm25", lambda: None)
    monkeypatch.setattr(
        prefetch,
        "prefetch_model",
        lambda model_id: called.__setitem__("model", model_id),
    )
    monkeypatch.setattr(
        prefetch,
        "prefetch_reranker",
        lambda model_id: called.__setitem__("reranker", model_id),
    )
    monkeypatch.setattr("sys.argv", ["prefetch", *argv])
    code = prefetch.main()
    if expect_zero:
        assert code == 0, f"prefetch.main() returned {code}"
        return called
    return code, called


def test_prefetch_takes_its_model_ids_from_the_resolved_config(monkeypatch):
    import dataclasses

    from noesis.core.config import Settings

    cfg = Settings()
    configured = dataclasses.replace(
        cfg,
        embedder=dataclasses.replace(cfg.embedder, model="acme/custom-embedder"),
        reranker=dataclasses.replace(cfg.reranker, model="acme/custom-reranker"),
    )
    monkeypatch.setattr("noesis.core.config.load_settings", lambda: configured)

    called = _patched_main(monkeypatch, [])
    assert called["model"] == "acme/custom-embedder"
    assert called["reranker"] == "acme/custom-reranker"


def test_explicit_flags_still_beat_the_config(monkeypatch):
    import dataclasses

    from noesis.core.config import Settings

    cfg = Settings()
    configured = dataclasses.replace(
        cfg,
        embedder=dataclasses.replace(cfg.embedder, model="acme/custom-embedder"),
        reranker=dataclasses.replace(cfg.reranker, model="acme/custom-reranker"),
    )
    monkeypatch.setattr("noesis.core.config.load_settings", lambda: configured)

    called = _patched_main(
        monkeypatch, ["--model", "cli/embedder", "--reranker-model", "cli/reranker"]
    )
    assert called["model"] == "cli/embedder"
    assert called["reranker"] == "cli/reranker"


def test_an_unreadable_config_skips_the_models_and_reports_failure(monkeypatch):
    """A config that will not parse means we do not know which models the
    service loads. Downloading the shipped defaults anyway is the exact
    "fetched a model nobody loads" failure this whole change is about — up to
    ~4.5 GB of it — so the config-independent assets (grammars, BM25) still
    come down and the model steps are skipped with a non-zero exit."""

    def explode():
        raise ValueError("config field 'reranker.enabled' must be a boolean")

    monkeypatch.setattr("noesis.core.config.load_settings", explode)

    code, called = _patched_main(monkeypatch, [], expect_zero=False)
    assert code != 0
    assert "model" not in called
    assert "reranker" not in called


def test_explicit_flags_still_work_with_an_unreadable_config(monkeypatch):
    # The operator has said which model they want; the broken file has no say.
    def explode():
        raise ValueError("bad config")

    monkeypatch.setattr("noesis.core.config.load_settings", explode)

    code, called = _patched_main(
        monkeypatch, ["--model", "cli/embedder", "--skip-reranker"], expect_zero=False
    )
    assert called["model"] == "cli/embedder"
    assert "reranker" not in called
    # Everything the operator asked for happened — the id came from the flag
    # and the reranker was explicitly skipped — so this run IS clean. Only an
    # id that stayed unknown makes the exit non-zero.
    assert code == 0

    # ...and asking for a model whose id only the broken config knows does not.
    code, called = _patched_main(monkeypatch, ["--skip-reranker"], expect_zero=False)
    assert "model" not in called
    assert code != 0


async def test_a_context_that_does_not_model_a_reranker_is_unknown_not_disabled():
    """``"disabled"`` is a claim: the kill switch is off. A hand-built or
    adapter context with no ``reranker`` attribute at all has not told us
    anything, and collapsing that into ``"disabled"`` would make the very
    distinction the sentinel exists for (issue #52 review) untrustworthy —
    the health surfaces already have ``"unknown"`` for "cannot tell"."""
    from noesis.prefetch import UNKNOWN_RERANKER

    ctx_without_reranker = SimpleNamespace(embedder=SimpleNamespace(model_id=MODEL_ID))
    with patch("noesis.prefetch.model_assets_ready") as probe:
        assets, ready = await reranker_readiness(
            getattr(ctx_without_reranker, "reranker", UNKNOWN_RERANKER)
        )
    assert (assets, ready) == ("unknown", "unknown")
    probe.assert_not_called()


# --- a model without its tokenizer is not "ready" (issue #52 review round 3) --
#
# Watched, not reasoned about: with only config.json + model.safetensors left in
# a real bge-reranker-v2-m3 snapshot (the other files removed, HF_HUB_OFFLINE=1),
# `CrossEncoder(...)` did NOT fail — it constructed an XLMRobertaTokenizer with
# vocab_size 5, tokenized "hello world" to [0, 3, 3, 2] (every token <unk>) and
# still returned a score of 0.849. Silent nonsense ranking is worse than the
# stall this field exists to warn about, and `model_assets_ready` called that
# cache "ready".


def test_config_and_weights_without_a_tokenizer_is_not_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is False


def test_a_sentencepiece_only_tokenizer_counts(monkeypatch):
    # bge-reranker-v2-m3's family ships sentencepiece rather than a vocab.txt;
    # requiring one specific filename would report a false "missing" for it.
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "sentencepiece.bpe.model": "/cache/sentencepiece.bpe.model",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_a_wordpiece_vocab_counts_too():
    # CodeRankEmbed ships vocab.txt alongside tokenizer.json.
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "vocab.txt": "/cache/vocab.txt",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_tokenizer_config_alone_is_not_a_tokenizer():
    """``tokenizer_config.json`` is metadata — it names the tokenizer class and
    its special tokens, and holds no vocabulary. A cache with the weights and
    the small JSONs but no vocab file is exactly the state ADR-90 measured
    loading a ``vocab_size`` 5 tokenizer that scored ``<unk>`` pairs, so it
    must not satisfy the tokenizer requirement (issue #52 review round 4)."""
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "tokenizer_config.json": "/cache/tokenizer_config.json",
                "special_tokens_map.json": "/cache/special_tokens_map.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is False


def test_bpe_merges_without_a_vocab_is_not_ready():
    """``merges.txt`` holds BPE merge RULES; the vocabulary it merges over
    lives in ``vocab.json``. A cache with the weights and merges but no vocab
    is the same false-``ready`` round 4 removed for ``tokenizer_config.json``
    — and listing it was a self-contradiction with the docstring's own "every
    entry carries a vocabulary" (issue #52 review round 5)."""
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "merges.txt": "/cache/merges.txt",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is False


def test_a_byte_level_bpe_vocab_without_its_merges_is_not_ready():
    """Byte-level BPE is the one family whose vocabulary is TWO files:
    ``GPT2Tokenizer.vocab_files_names`` is
    ``{'vocab_file': 'vocab.json', 'merges_file': 'merges.txt'}`` (checked
    against the installed transformers, not assumed). Either alone is an
    incomplete tokenizer, so neither alone may read ``ready`` (issue #52
    review round 6)."""
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "vocab.json": "/cache/vocab.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is False


def test_the_byte_level_bpe_pair_together_is_ready():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "vocab.json": "/cache/vocab.json",
                "merges.txt": "/cache/merges.txt",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_a_llama_family_sentencepiece_vocab_counts():
    """``LlamaTokenizer.vocab_files_names`` names ``tokenizer.model``. Leaving
    it out is a false MISSING, which sounds like the safe direction until you
    follow it through: the plugin healthcheck then exits 1 on every run for a
    correctly cached model, and its remedy — "run prefetch" — can never clear
    it (issue #52 review round 6)."""
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "tokenizer.model": "/cache/tokenizer.model",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


# --- a local model directory is held to the same contract (round 6) ----------
#
# `[embedder] model` / `[reranker] model` are free text and
# sentence-transformers accepts a directory. ADR-79 answered that case with
# "the assets ARE that directory" — which was true enough when the only
# question was whether a hub repo had been fetched, and false the moment the
# function started promising config + weights + vocabulary. An empty or
# half-copied directory reported `ready`, and the plugin healthcheck printed
# [ OK ] for it (issue #52 review round 6).


def test_an_empty_local_model_directory_is_not_ready(tmp_path):
    empty = tmp_path / "half-copied-model"
    empty.mkdir()
    assert model_assets_ready(str(empty)) is False


def test_a_local_directory_missing_its_vocabulary_is_not_ready(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    assert model_assets_ready(str(d)) is False


def test_a_local_directory_missing_its_config_is_not_ready(tmp_path):
    """The third requirement, which round 6 first left out of the directory
    path: ``config.json`` was checked on the hub-cache branch only, so a
    directory copy interrupted before it reported ``ready`` and the first
    ``search_code`` then failed inside ``SentenceTransformer(...)`` (issue #52
    review round 7)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}")
    assert model_assets_ready(str(d)) is False


def test_a_complete_local_model_directory_is_ready(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}")
    assert model_assets_ready(str(d)) is True


def test_a_path_that_is_not_a_directory_is_not_ready(tmp_path):
    assert model_assets_ready(str(tmp_path / "nope")) is False


# --- the config is read only when an id is actually needed (round 6) ---------


def test_help_does_not_read_the_config(monkeypatch, capsys):
    """`--help` printed the config-parse error above the usage text, because
    the ids were resolved before argparse ran. Nothing that only prints help
    should touch config.toml."""
    import noesis.prefetch as prefetch

    def explode():
        raise ValueError("bad config")

    monkeypatch.setattr("noesis.core.config.load_settings", explode)
    monkeypatch.setattr("sys.argv", ["prefetch", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        prefetch.main()
    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert "bad config" not in captured.out + captured.err


def test_skipping_the_models_does_not_read_the_config(monkeypatch, capsys):
    # --skip-model skips the reranker too, so no id is needed and a broken
    # config is none of this run's business: clean exit, nothing printed.
    def explode():
        raise ValueError("bad config")

    monkeypatch.setattr("noesis.core.config.load_settings", explode)
    code, called = _patched_main(monkeypatch, ["--skip-model"], expect_zero=False)
    assert code == 0
    assert called == {}
    assert "bad config" not in capsys.readouterr().err


def test_a_relative_directory_shaped_like_a_repo_id_is_read_from_disk(
    tmp_path, monkeypatch
):
    """``models/bge-reranker`` is BOTH a valid hub repo id and a relative
    directory, so the ``HFValidationError`` fallback never fired for it and it
    answered ``missing`` forever while the model loaded fine from disk —
    turning the plugin healthcheck into a permanent ``exit 1`` whose remedy
    (run prefetch) can never clear it (issue #52 review round 8).

    Both halves were measured, not reasoned about:
    ``validate_repo_id("models/bge-reranker")`` does not raise, and with an
    empty HF cache and ``HF_HUB_OFFLINE=1`` a ``CrossEncoder`` pointed at such
    a directory loaded and scored 0.965 while this function said ``missing``.
    A local directory also WINS over a cached hub repo of the same name (an
    incomplete ``./BAAI/bge-reranker-v2-m3/`` beside a fully cached one made
    the load fail on the local copy), so asking the filesystem first is what
    the loader itself does.
    """
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "bge-local"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}")

    assert model_assets_ready("models/bge-local") is True


def test_an_incomplete_relative_directory_still_reports_missing(tmp_path, monkeypatch):
    # The directory answer is the same contract, not a free pass.
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "bge-local"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")

    assert model_assets_ready("models/bge-local") is False


def test_a_hub_repo_id_with_no_such_directory_still_asks_the_cache(
    tmp_path, monkeypatch
):
    # Nothing on disk called "nomic-ai/CodeRankEmbed", so the cache decides.
    monkeypatch.chdir(tmp_path)
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        _cache_fake(
            present={
                "config.json": "/cache/config.json",
                "model.safetensors": "/cache/model.safetensors",
                "tokenizer.json": "/cache/tokenizer.json",
            }
        ),
    ):
        assert model_assets_ready(MODEL_ID) is True


def test_an_unusable_path_pin_reports_missing_instead_of_raising():
    """Round 8 moved the directory check ahead of the hub lookup and left it
    outside the guard, so two pins escaped as exceptions rather than answers:
    ``~nosuchuser/model`` (``RuntimeError: Could not determine home
    directory``) and a path segment past NAME_MAX (``OSError: [Errno 36] File
    name too long``). Both propagate straight out of ``/healthz``,
    ``GET /projects/{id}/status`` and ``get_index_status`` — the exact failure
    ADR-79's guard exists to prevent, and the opposite of this function's
    stated "anything malformed falls through to missing" contract (issue #52
    review round 9)."""
    assert model_assets_ready("~nosuchuser42/model") is False
    assert model_assets_ready("/tmp/" + "x" * 400) is False
