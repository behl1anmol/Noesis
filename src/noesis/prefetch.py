"""Install-time asset prefetch — ``uv run python -m noesis.prefetch``.

Downloads every network-fetched asset the service needs (decision rows 30
and 32, doc §3.8): tree-sitter grammars for all canonical languages (the
1.12.x language pack fetches each grammar over HTTP on first
``get_parser``), the embedding model weights, and the BM25 tokenizer
assets fastembed fetches on first sparse encode (~100 KB of stopwords and
stemmer config — same asset class as the model weights, no code or query
leaves the machine). Run once after ``uv sync``; afterwards the service
makes zero outbound calls at runtime. Deliberately lives outside
``core/`` — this is the only module whose job is to trigger downloads.
"""

from __future__ import annotations

import argparse
import os
import sys

# fastembed defaults its cache to the system tmp dir, which evaporates on
# reboot and would trigger a re-download at runtime. Pin it somewhere
# persistent unless the operator chose one. app.py sets the same default
# so prefetch and serving resolve one cache.
FASTEMBED_CACHE_ENV = "FASTEMBED_CACHE_PATH"
FASTEMBED_CACHE_DEFAULT = "data/fastembed_cache"


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
                vector={
                    "bm25": models.Document(text="prefetch", model=BM25_MODEL_ID)
                },
            )
        ],
    )
    print(f"bm25 ok: {BM25_MODEL_ID}")


def main() -> int:
    os.environ.setdefault(FASTEMBED_CACHE_ENV, FASTEMBED_CACHE_DEFAULT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model", action="store_true", help="grammars only, no model weights"
    )
    parser.add_argument("--model", default="nomic-ai/CodeRankEmbed")
    args = parser.parse_args()

    failed = prefetch_grammars()
    prefetch_bm25()
    if not args.skip_model:
        prefetch_model(args.model)
    if failed:
        print(f"{len(failed)} grammar(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
