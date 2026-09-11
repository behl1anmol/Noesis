"""Install-time asset prefetch — ``uv run python -m noesis.prefetch``.

Downloads every network-fetched asset the service needs (decision rows 30
and 32, doc §3.8): tree-sitter grammars for all canonical languages (the
1.12.x language pack fetches each grammar over HTTP on first
``get_parser``), the embedding model weights, the M4 reranker weights
(skippable — the reranker is optional, config ``reranker.enabled``), and
the BM25 tokenizer assets fastembed fetches on first sparse encode
(~100 KB of stopwords and stemmer config — same asset class as the model
weights, no code or query leaves the machine). Run once after ``uv sync``;
afterwards the service makes zero outbound calls at runtime. Deliberately
lives outside ``core/`` — this is the only module whose job is to trigger
downloads.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

# fastembed defaults its cache to the system tmp dir, which evaporates on
# reboot and would trigger a re-download at runtime. Pin it somewhere
# persistent unless the operator chose one. app.py resolves the SAME default
# so prefetch and serving share one cache.
FASTEMBED_CACHE_ENV = "FASTEMBED_CACHE_PATH"


def default_fastembed_cache() -> str:
    """Absolute, cwd-independent default cache dir for fastembed's BM25
    assets (M8). The previous ``data/fastembed_cache`` default was resolved
    against the current working directory, so prefetch (run from the repo)
    and the stdio MCP server (spawned with the agent host's cwd) landed on
    DIFFERENT caches — the first sparse search then re-downloaded assets at
    runtime, or failed offline. Anchoring to the user cache dir makes every
    process on the machine resolve one path regardless of cwd."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base).expanduser() / "noesis" / "fastembed")


def model_assets_ready(model_id: str) -> bool:
    """Best-effort, no-network check: is ``model_id`` fully cached in the
    local HF cache (ADR-78 addendum, issue #47 finding 4, PR #50 review)?

    Asked of BOTH model boundaries (issue #52): the embedder's
    ``SentenceTransformer`` and the reranker's ``CrossEncoder`` resolve the
    same HF cache and need the same three kinds of file. Hence the neutral
    name: it was ``embedder_assets_ready`` while the embedder was the only
    caller.

    Requires ``config.json``, at least one weight file, AND at least one
    tokenizer file:

    * weights — a single-file checkpoint (``model.safetensors`` /
      ``pytorch_model.bin``) or a sharded one's index manifest
      (``*.index.json``). ``config.json`` alone reported ``ready`` after a
      download interrupted between metadata and weights (ADR-79).
    * tokenizer — a file carrying an actual VOCABULARY, taken from each
      family's own ``vocab_files_names`` rather than guessed: a fast
      tokenizer's ``tokenizer.json``; sentencepiece's ``sentencepiece.bpe.model``
      (XLM-R), ``spiece.model`` (T5/ALBERT) or ``tokenizer.model`` (Llama);
      wordpiece's ``vocab.txt``; or byte-level BPE's ``vocab.json`` AND
      ``merges.txt``, which is the one family whose vocabulary is two files
      (``GPT2Tokenizer`` declares both) and so is the one entry checked as a
      pair. Deliberately NOT ``tokenizer_config.json`` or
      ``special_tokens_map.json``, which are metadata: they name the tokenizer
      class and its special tokens and hold no vocabulary, so a cache with
      only those is the same broken state as no tokenizer at all (issue #52
      review round 4). Weights without a vocabulary are not
      a working model, and the failure is quieter than a stall: with only
      ``config.json`` + ``model.safetensors`` left in a real
      ``BAAI/bge-reranker-v2-m3`` cache and ``HF_HUB_OFFLINE=1``,
      ``CrossEncoder(...)`` did not raise — it built an ``XLMRobertaTokenizer``
      with ``vocab_size`` 5 that tokenized every word to ``<unk>`` and still
      returned a plausible-looking score (measured, ADR-90). A "ready" that
      means "will silently rank noise" is worse than one that means "will
      stall".

    ANY member of a family satisfies it, never a specific filename: the two
    default models differ (``BAAI/bge-reranker-v2-m3`` ships
    ``sentencepiece.bpe.model``, ``nomic-ai/CodeRankEmbed`` ships
    ``vocab.txt``; both ship ``tokenizer.json``), and requiring one spelling
    would report a false "missing" for the other. Only the pytorch-backend
    filenames are checked, not the whole repo tree: neither ``LocalSTEmbedder``
    nor ``LocalCrossEncoderReranker`` requests ``backend="onnx"``, so an
    onnx/openvino variant shipped alongside pytorch weights in the same repo
    must not cause a false "missing".

    ``try_to_load_from_cache`` returns a sentinel object — not ``None`` — for
    a filename HF has already probed and confirmed absent from the repo
    (e.g. a single-file checkpoint's sharded-index name, probed once by a
    prior load's fallback logic and cached negative; verified against a live
    cache). ``isinstance(result, str)`` treats that sentinel as absent, same
    as a filename that was never probed at all — ``is not None`` would not.

    All lookups happen in the default HF cache location (``HF_HOME`` /
    ``~/.cache/huggingface/hub``), the same resolution ``sentence_transformers``
    itself uses, so this asks exactly what the real load will find."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.errors import HFValidationError

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(model_id, filename), str)

    # ``[embedder] model`` (and ``[reranker] model``) is free text, and
    # sentence-transformers accepts a local directory as readily as a hub repo
    # id. The filesystem is asked FIRST because that is what the loader itself
    # does: with an incomplete ``./BAAI/bge-reranker-v2-m3/`` in the working
    # directory beside a fully cached copy of that hub repo, ``CrossEncoder``
    # failed on the LOCAL one (measured) — the directory wins.
    #
    # Ordering by the exception instead was wrong for a whole class of pins:
    # ``models/bge-reranker`` is a valid hub repo id AND a relative directory,
    # so it never reached the fallback and answered "missing" forever while the
    # model loaded fine from disk — which, now that a missing reranker exits 1
    # from the plugin healthcheck, is a permanent red whose remedy (run
    # prefetch) can never clear it (issue #52 review round 8).
    #
    # Either way the CONTRACT is the same: config, weights and a vocabulary.
    # Only the lookup changes. An unparseable id that is also not a directory
    # falls through to "missing", the fail-safe direction ADR-79 chose: a false
    # "missing" costs a redundant prefetch, a false "ready" is the bug.
    try:
        directory = Path(model_id).expanduser()
        if directory.is_dir():
            return _has_required_files(lambda name: (directory / name).is_file())
    except (OSError, RuntimeError, TypeError):
        # The path itself is unusable, not merely absent: `~nosuchuser/model`
        # raises RuntimeError("Could not determine home directory") from
        # expanduser, and a segment past NAME_MAX raises OSError(ENAMETOOLONG)
        # from is_dir, and a non-string pin (`[embedder] model = 123`, which
        # load_settings does not type-check) raises TypeError from Path()
        # itself — none of which pathlib swallows. Escaping here means
        # /healthz, GET /projects/{id}/status and get_index_status all 500 on a
        # typo'd pin, which is the failure ADR-79's guard exists to prevent
        # (issue #52 review round 9). "missing" is the answer: nothing readable
        # is there, and no hub id can be built from it either.
        return False
    # Not a directory: a hub repo id, or malformed. ``try_to_load_from_cache``
    # raises HFValidationError for the malformed case, which must not escape —
    # it used to propagate out of /healthz, GET /projects/{id}/status and the
    # get_index_status MCP tool, taking down the very surface ADR-78 added to
    # keep the health check honest (ADR-79).
    try:
        return _has_required_files(cached)
    except HFValidationError:
        return False


def _has_required_files(present: Callable[[str], bool]) -> bool:
    """:func:`model_assets_ready`'s whole contract — config, weights and a
    vocabulary — over any "is this file there?" predicate: the HF cache for a
    hub repo id, plain ``is_file()`` for a local model directory. One
    implementation so the two cannot promise different contracts (issue #52
    review rounds 6 and 7: the first cut left the ``config.json`` check
    outside this helper, and the directory path silently skipped it)."""
    if not present("config.json"):
        return False
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    # Each entry is a SELF-SUFFICIENT vocabulary, read off the corresponding
    # tokenizer class's `vocab_files_names` rather than guessed.
    # `tokenizer_config.json` and `special_tokens_map.json` are deliberately
    # absent: they are metadata, and a cache holding them without a vocabulary
    # loads the silently-broken tokenizer ADR-90 measured.
    tokenizer_files = (
        "tokenizer.json",  # fast tokenizers (both default models ship one)
        "sentencepiece.bpe.model",  # XLM-R, e.g. bge-reranker-v2-m3
        "spiece.model",  # T5/ALBERT
        "tokenizer.model",  # Llama/Mistral sentencepiece
        "vocab.txt",  # wordpiece, e.g. CodeRankEmbed
    )
    # Byte-level BPE is the exception: `GPT2Tokenizer.vocab_files_names`
    # declares vocab.json AND merges.txt, so neither counts alone — the merges
    # are rules with nothing to apply them to, and the vocab cannot be merged
    # without them.
    byte_level_bpe = ("vocab.json", "merges.txt")
    if not any(present(f) for f in weight_files):
        return False
    return any(present(f) for f in tokenizer_files) or all(
        present(f) for f in byte_level_bpe
    )


async def model_readiness(model: object) -> tuple[str, bool | str]:
    """Shared ``(assets, ready)`` computation behind ``/healthz`` (ADR rows
    78/79) and ``jobs.index_status`` (ADR row 81) — PR #50 round-5 review.
    Previously pasted verbatim in both call sites: they agreed only because
    the text was identical and one test compared the two endpoints' output,
    not because there was one implementation: a future edit to either copy
    (e.g. a "warming" state, a different ``n/a`` rule) had nothing stopping it
    from landing on one side only. *model* is duck-typed — ``model_id``
    required, ``resolved_device`` optional — so this has no dependency on a
    specific Embedder or Reranker implementation and stays outside ``core/``
    like the rest of this module (module docstring).

    ``resolved_device`` is the load signal, not a device-selection record:
    both boundaries assign it only after the model constructor returns
    (ADR-79 for the embedder; issue #52 for the reranker, which had the same
    premature assignment until this signal started reading it). So ``False``
    here means "assets may be cached, but the model is not loaded yet — the
    next call pays the load", and ``"n/a"`` means the implementation does not
    report a device at all (a test double)."""
    ready = await asyncio.to_thread(model_assets_ready, model.model_id)
    assets = "ready" if ready else "missing"
    resolved_device = getattr(model, "resolved_device", "n/a")
    model_ready = "n/a" if resolved_device == "n/a" else bool(resolved_device)
    return assets, model_ready


class _UnknownReranker:
    """Type of :data:`UNKNOWN_RERANKER` — a distinct type, not a bare
    ``object()``, so the sentinel is legible in a traceback or a repr."""

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return "UNKNOWN_RERANKER"


#: "This context does not model a reranker at all" — the value call sites pass
#: as ``getattr(ctx, "reranker", UNKNOWN_RERANKER)``. Distinct from ``None``,
#: which is a positive statement that the kill switch is off (issue #52
#: review): collapsing the two would let the health surfaces answer
#: ``"disabled"`` for a state they cannot actually see, which is exactly the
#: trust the sentinel was chosen over an absent key to provide.
UNKNOWN_RERANKER = _UnknownReranker()


async def reranker_readiness(reranker: object | None) -> tuple[str, bool | str]:
    """``(reranker_assets, reranker_ready)`` for the health surfaces (issue
    #52). ``None`` is the ``reranker.enabled=false`` kill switch (§3.3, the
    shipped default), which is a different answer from "enabled but its
    weights are missing" — reporting ``"missing"`` for a feature nobody turned
    on would send an operator to fetch 2.3GB they do not need, and omitting
    the fields entirely would leave a caller unable to tell "off" from "this
    server does not report it". So the kill switch gets its own value and no
    HF-cache lookup happens at all.

    :data:`UNKNOWN_RERANKER` is the third case: a duck-typed context (a test
    or an adapter) that carries no ``reranker`` attribute has not told us the
    switch is off, so it reports ``"unknown"`` — the value both surfaces
    already use for "cannot tell".

    Lives here, next to the embedder half, rather than at the two call sites:
    ADR-82 had to unwind exactly that duplication once already."""
    if reranker is UNKNOWN_RERANKER:
        return "unknown", "unknown"
    if reranker is None:
        return "disabled", "disabled"
    return await model_readiness(reranker)


def prefetch_grammars() -> list[str]:
    from tree_sitter_language_pack import get_parser

    from noesis.core.languages import EXT_TO_LANGUAGE

    failed: list[str] = []
    for language in sorted(set(EXT_TO_LANGUAGE.values())):
        try:
            get_parser(language)  # type: ignore[arg-type]
            print(f"grammar ok: {language}")
        except Exception as exc:
            # Missing grammar is degraded (line-chunk fallback), not fatal.
            print(f"grammar FAILED: {language}: {exc}", file=sys.stderr)
            failed.append(language)
    return failed


def prefetch_model(model_id: str) -> None:
    # Through the Embedder boundary — rule 1 forbids importing
    # sentence_transformers anywhere else; one embed forces the weight
    # download inside LocalSTEmbedder's worker.
    import asyncio

    from noesis.core.embedder import LocalSTEmbedder

    embedder = LocalSTEmbedder(model_id=model_id)
    vector = asyncio.run(embedder.embed_query("prefetch"))
    print(f"model ok: {model_id} (dim {len(vector)})")


def prefetch_reranker(model_id: str) -> None:
    # Through the Reranker boundary — rule 1 (as amended by ADR-33) allows
    # sentence_transformers only in the two model-loading modules; a preload
    # forces the weight download inside the reranker's worker.
    import asyncio

    from noesis.core.reranker import LocalCrossEncoderReranker

    reranker = LocalCrossEncoderReranker(model_id=model_id)
    asyncio.run(reranker.preload())
    reranker.close()
    print(f"reranker ok: {model_id}")


def prefetch_bm25() -> None:
    # Through the same client-side inference path the vector store uses at
    # runtime, so exactly the assets qdrant-client will look for get cached.
    from qdrant_client import QdrantClient, models

    from noesis.core.vectorstore import BM25_MODEL_ID

    client = QdrantClient(":memory:")
    client.create_collection(
        "prefetch_probe",
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )
    client.upsert(
        "prefetch_probe",
        points=[
            models.PointStruct(
                id=1,
                vector={"bm25": models.Document(text="prefetch", model=BM25_MODEL_ID)},
            )
        ],
    )
    print(f"bm25 ok: {BM25_MODEL_ID}")


def configured_model_ids() -> tuple[str, str] | None:
    """``(embedder model, reranker model)`` the SERVICE will actually load, or
    ``None`` if the config cannot be read.

    Read from the same config resolution the service uses (``NOESIS_CONFIG``
    → ``./config.toml`` → XDG, ADR-44), because prefetching a model nobody
    loads fixes nothing: with a non-default ``[embedder] model`` or
    ``[reranker] model``, the hardcoded repo ids this replaces downloaded
    multi-GB weights the service never opens, while ``/healthz`` and the
    plugin's ``healthcheck.py`` — whose remedy is literally "run prefetch" —
    kept reporting ``missing`` (issue #52 review). Run prefetch with the same
    ``NOESIS_CONFIG`` the service gets, or from the same directory, for the
    two to agree.

    Reads the model IDS only, deliberately not ``reranker.enabled``: a default
    config (``enabled = false``) still prefetches the reranker's weights, which
    is bandwidth for a model the service will not load. That is tracked as
    issue #58 and was left alone on purpose — unlike a wrong model id it breaks
    nothing (with reranking off the health fields read ``"disabled"`` and never
    complain), ``--skip-reranker`` already exists, and changing what a
    documented install command downloads belongs in its own change (ADR-91).

    A MISSING config is not a failure — ``load_settings`` answers with the
    shipped defaults, which is exactly right for a fresh install. A config
    that exists but will not parse is different: it means the operator
    configured something and we cannot see what, so ``None`` comes back and
    the caller skips the model downloads rather than fetching up to ~4.5 GB of
    defaults the service may never load (ADR-90). The config-independent
    assets — grammars, BM25 — still come down."""
    from noesis.core.config import load_settings

    try:
        cfg = load_settings()
    except Exception as exc:  # noqa: BLE001 — any config error, same answer
        print(f"could not read config: {exc}", file=sys.stderr)
        return None
    return cfg.embedder.model, cfg.reranker.model


def main() -> int:
    os.environ.setdefault(FASTEMBED_CACHE_ENV, default_fastembed_cache())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model", action="store_true", help="grammars only, no model weights"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="embedding model to fetch (default: whatever config.toml names)",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="skip the ~2.3 GB reranker weights (only needed if reranker.enabled)",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="reranker model to fetch (default: whatever config.toml names)",
    )
    args = parser.parse_args()

    # Resolved AFTER parsing, and only when a step actually needs an id: the
    # config was read before argparse ran, so `--help` printed a config-parse
    # error above the usage text, and `--skip-model` complained about a file it
    # was never going to consult (issue #52 review round 6). `None` survives
    # here when the config cannot be read, and the steps below skip.
    wants_embedder = not args.skip_model and args.model is None
    wants_reranker = (
        not (args.skip_model or args.skip_reranker) and args.reranker_model is None
    )
    if wants_embedder or wants_reranker:
        configured = configured_model_ids()
        if configured is not None:
            if args.model is None:
                args.model = configured[0]
            if args.reranker_model is None:
                args.reranker_model = configured[1]

    failed = prefetch_grammars()
    prefetch_bm25()
    # An unresolved model id means the config did not parse and no flag named
    # one. Skipping is the point (see configured_model_ids), but it is a
    # non-zero exit: the prefetch did not do the job it was asked to do, and a
    # CI step or an install script must not read that as success.
    unresolved: list[str] = []
    if not args.skip_model:
        if args.model is None:
            unresolved.append("--model")
        else:
            prefetch_model(args.model)
    if not (args.skip_model or args.skip_reranker):
        if args.reranker_model is None:
            unresolved.append("--reranker-model")
        else:
            prefetch_reranker(args.reranker_model)
    if failed:
        print(f"{len(failed)} grammar(s) failed: {', '.join(failed)}", file=sys.stderr)
    if unresolved:
        print(
            "skipped model weights: the config could not be read, so the model "
            f"ids are unknown — fix config.toml, or pass {' and '.join(unresolved)}",
            file=sys.stderr,
        )
    if failed or unresolved:
        return 1
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
