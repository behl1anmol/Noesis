#!/usr/bin/env bash
# CI guardrail greps (CLAUDE.md rules 1-2, expanded doc M2 exit criterion).
# Exit non-zero if any invariant is violated. Run from the repo root.
set -u

fail=0

# Rule 1 (as amended by ADR-33): sentence_transformers may only be imported
# in the two model-loading boundaries, core/embedder.py and core/reranker.py.
hits=$(grep -rn --include='*.py' -E '(^|[^#]*)\b(import|from)\s+sentence_transformers\b' src/ \
  | grep -v -e '^src/noesis/core/embedder\.py:' -e '^src/noesis/core/reranker\.py:' || true)
if [ -n "$hits" ]; then
  echo "FAIL: sentence_transformers imported outside core/{embedder,reranker}.py:"
  echo "$hits"
  fail=1
fi

# Rule 2 / ADR-25: no HTTP client imports anywhere in core/.
hits=$(grep -rn --include='*.py' -E '\b(import|from)\s+(httpx|requests|aiohttp|urllib3|http\.client|urllib\.request)\b' \
  src/noesis/core/ || true)
if [ -n "$hits" ]; then
  echo "FAIL: HTTP client import inside core/ (ADR-25 forbids network egress):"
  echo "$hits"
  fail=1
fi

# Rule 2: never bind 0.0.0.0.
hits=$(grep -rn --include='*.py' '0\.0\.0\.0' src/ || true)
if [ -n "$hits" ]; then
  echo "FAIL: 0.0.0.0 binding found (must be 127.0.0.1):"
  echo "$hits"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "ci_greps: all guardrail greps clean"
fi
exit "$fail"
