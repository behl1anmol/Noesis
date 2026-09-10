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
    same HF cache and need the same files — ``BAAI/bge-reranker-v2-m3``, the
    default reranker, ships exactly ``config.json`` + ``model.safetensors``
    (checked against the hub's file list, not assumed). Hence the neutral
    name: it was ``embedder_assets_ready`` while the embedder was the only
    caller.

    Requires ``config.json`` AND at least one weight file: a single-file
    checkpoint (``model.safetensors`` / ``pytorch_model.bin``) or a sharded
    one's index manifest (``*.index.json``). Checking ``config.json`` alone
    reported ``ready`` after a download interrupted between metadata and
    weights — exactly the silent cold-start stall this check exists to
    surface. Only the pytorch-backend filenames are checked, not the whole
    repo tree: neither ``LocalSTEmbedder`` nor ``LocalCrossEncoderReranker``
    requests ``backend="onnx"``, so an onnx/openvino variant shipped alongside
    pytorch weights in the same repo must not cause a false "missing".

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

    # ``[embedder] model`` (and ``[reranker] model``) is free text and
    # sentence-transformers accepts a local directory, which is not a hub repo
    # id — the hub call raises HFValidationError for it. Unguarded, that propagated out of /healthz,
    # GET /projects/{id}/status and the get_index_status MCP tool, so a
    # locally-pinned model took down the very surface ADR-78 added to keep
    # the health check honest. For a real directory the answer is knowable
    # without the hub at all: the assets ARE that directory. Anything else
    # malformed falls through to "missing", the fail-safe direction ADR-79
    # already chose (a false "missing" costs a redundant prefetch; a false
    # "ready" is the bug).
    try:
        if not cached("config.json"):
            return False
    except HFValidationError:
        return Path(model_id).expanduser().is_dir()
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(cached(f) for f in weight_files)


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


async def reranker_readiness(reranker: object | None) -> tuple[str, bool | str]:
    """``(reranker_assets, reranker_ready)`` for the health surfaces (issue
    #52). ``None`` is the ``reranker.enabled=false`` kill switch (§3.3, the
    shipped default), which is a different answer from "enabled but its
    weights are missing" — reporting ``"missing"`` for a feature nobody turned
    on would send an operator to fetch 2.3GB they do not need, and omitting
    the fields entirely would leave a caller unable to tell "off" from "this
    server does not report it". So the kill switch gets its own value and no
    HF-cache lookup happens at all.

    Lives here, next to the embedder half, rather than at the two call sites:
    ADR-82 had to unwind exactly that duplication once already."""
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


def configured_model_ids() -> tuple[str, str]:
    """``(embedder model, reranker model)`` the SERVICE will actually load.

    Read from the same config resolution the service uses (``NOESIS_CONFIG``
    → ``./config.toml`` → XDG, ADR-44), because prefetching a model nobody
    loads fixes nothing: with a non-default ``[embedder] model`` or
    ``[reranker] model``, the hardcoded repo ids this replaces downloaded
    multi-GB weights the service never opens, while ``/healthz`` and the
    plugin's ``healthcheck.py`` — whose remedy is literally "run prefetch" —
    kept reporting ``missing`` (issue #52 review). Run prefetch with the same
    ``NOESIS_CONFIG`` the service gets, or from the same directory, for the
    two to agree.

    A config that cannot be read falls back to the shipped defaults rather
    than aborting: this is the first command a fresh install runs, often
    before any config exists, and grammars plus BM25 assets should not be
    held hostage to an unrelated syntax error in a file the service has not
    tried to load yet."""
    from noesis.core.config import Settings, load_settings

    try:
        cfg = load_settings()
    except Exception as exc:  # noqa: BLE001 — any config error, same fallback
        print(
            f"could not read config ({exc}); using default model ids", file=sys.stderr
        )
        cfg = Settings()
    return cfg.embedder.model, cfg.reranker.model


def main() -> int:
    os.environ.setdefault(FASTEMBED_CACHE_ENV, default_fastembed_cache())
    default_model, default_reranker_model = configured_model_ids()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model", action="store_true", help="grammars only, no model weights"
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"embedding model to fetch (default: {default_model}, from config)",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="skip the ~2.3 GB reranker weights (only needed if reranker.enabled)",
    )
    parser.add_argument(
        "--reranker-model",
        default=default_reranker_model,
        help=f"reranker model to fetch (default: {default_reranker_model}, from config)",
    )
    args = parser.parse_args()

    failed = prefetch_grammars()
    prefetch_bm25()
    if not args.skip_model:
        prefetch_model(args.model)
    if not (args.skip_model or args.skip_reranker):
        prefetch_reranker(args.reranker_model)
    if failed:
        print(f"{len(failed)} grammar(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
