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


def main() -> int:
    os.environ.setdefault(FASTEMBED_CACHE_ENV, default_fastembed_cache())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model", action="store_true", help="grammars only, no model weights"
    )
    parser.add_argument("--model", default="nomic-ai/CodeRankEmbed")
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="skip the ~2.3 GB reranker weights (only needed if reranker.enabled)",
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
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
