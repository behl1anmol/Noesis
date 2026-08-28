#!/usr/bin/env bash
# CI guardrail greps (CLAUDE.md rules 1-2, expanded doc M2 exit criterion).
# Exit non-zero if any invariant is violated. Run from the repo root.
set -u

fail=0

# Rule 1 (as amended by ADR-33): sentence_transformers may only be imported
# in the two model-loading boundaries, core/embedder.py and core/reranker.py.
# (Round-5 finding 27: the old `(^|[^#]*)` prefix was decorative -- grep -E
# matches anywhere in the line regardless, so it never actually excluded a
# `#`-commented line. Dropped rather than "made real": a commented-out import
# doesn't run, but flagging it anyway is over-strict, not under-strict, which
# is the fail-safe direction rule 9 asks for -- not worth the extra regex
# complexity to special-case.)
hits=$(grep -rn --include='*.py' -E '\b(import|from)\s+sentence_transformers\b' src/ \
  | grep -v -e '^src/noesis/core/embedder\.py:' -e '^src/noesis/core/reranker\.py:' || true)
if [ -n "$hits" ]; then
  echo "FAIL: sentence_transformers imported outside core/{embedder,reranker}.py:"
  echo "$hits"
  fail=1
fi

# Rule 2 / ADR-25: no HTTP client imports anywhere in core/.
# (Round-5 finding 10: the module-name alternation only caught the
# `import urllib.request` / `from urllib.request import ...` spellings, not
# `from urllib import request` (with or without `as ...` / other names in the
# same import list). Added a second alternative for that form. Scoped to
# `\brequest\b` specifically -- `urllib.parse`/`urllib.error` are not egress
# and must stay clean -- so `from urllib import parse` is deliberately not
# flagged.)
hits=$(grep -rn --include='*.py' -E '\b(import|from)\s+(httpx|requests|aiohttp|urllib3|http\.client|urllib\.request)\b|\bfrom\s+urllib\s+import\s+.*\brequest\b' \
  src/noesis/core/ || true)
if [ -n "$hits" ]; then
  echo "FAIL: HTTP client import inside core/ (ADR-25 forbids network egress):"
  echo "$hits"
  fail=1
fi

# Rule 2: never bind 0.0.0.0.
# (Round-5 finding 10: `--include='*.py'` missed the six non-Python files
# under src/ that actually reach a browser or a server config --
# api/static/app.js, api/static/style.css, and the four Jinja templates.
# Listing the extensions that exist in src/ today rather than dropping
# --include altogether, to avoid matching the binary favicon/logo PNGs.)
hits=$(grep -rn --include='*.py' --include='*.js' --include='*.css' --include='*.html' '0\.0\.0\.0' src/ || true)
if [ -n "$hits" ]; then
  echo "FAIL: 0.0.0.0 binding found (must be 127.0.0.1):"
  echo "$hits"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "ci_greps: all guardrail greps clean"
fi
exit "$fail"
