"""Install-time asset prefetch — ``uv run python -m noesis.prefetch``.

Downloads every network-fetched asset the service needs (decision row 30,
doc §3.8): tree-sitter grammars for all canonical languages (the 1.12.x
language pack fetches each grammar over HTTP on first ``get_parser``) and
the embedding model weights. Run once after ``uv sync``; afterwards the
service makes zero outbound calls at runtime. Deliberately lives outside
``core/`` — this is the only module whose job is to trigger downloads.
"""

from __future__ import annotations

import argparse
import sys


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model", action="store_true", help="grammars only, no model weights"
    )
    parser.add_argument("--model", default="nomic-ai/CodeRankEmbed")
    args = parser.parse_args()

    failed = prefetch_grammars()
    if not args.skip_model:
        prefetch_model(args.model)
    if failed:
        print(f"{len(failed)} grammar(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
