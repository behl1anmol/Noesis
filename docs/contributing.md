# Contributing & development

How to work on Noesis itself — environment, tests, guardrails, and the
house rules that keep the design honest.

## Environment

```bash
uv sync --all-groups          # deps incl. dev + docs groups
docker compose up -d          # Qdrant (the only container)
uv run python -m noesis.prefetch   # one-time model/grammar download
```

Python ≥ 3.11 (3.12 targeted), managed by [uv](https://docs.astral.sh/uv/);
`uv.lock` is the source of truth for exact versions.

## Test suites

| Command | What it runs |
|---|---|
| `uv run pytest` | full offline suite (~300 tests) — `FakeEmbedder` + in-memory Qdrant, no Docker, no model downloads |
| `uv run pytest -m integration` | opt-in: loads the real embedding model |
| `uv run pytest tests/eval/ -m golden` | the [evaluation harness](internals/evaluation.md) — self-indexes this repo against the 40-query golden set |
| `bash .claude/scripts/ci_greps.sh` | guardrail greps (local-only invariants) |

## Guardrails enforced in CI

`.github/workflows/ci.yml` runs the greps and the offline suite on every
push and pull request:

- no `sentence_transformers` import outside `core/embedder.py` and
  `core/reranker.py` ([ADR-33](project/decisions.md))
- no HTTP client anywhere in `core/` ([ADR-25](project/decisions.md))
- `127.0.0.1`-only binds — never `0.0.0.0`
- `mcp` pinned `< 2` until the MCP v2 checkpoint decision

## House rules

From [`CLAUDE.md`](https://github.com/behl1anmol/Noesis/blob/main/CLAUDE.md):

1. Every design decision gets a recorded rationale — **no rationale, no
   merge**. The result is the [decision log](project/decisions.md).
2. No new runtime dependency without a decision row.
3. Mistakes are captured as [lessons](project/lessons.md) before moving on;
   active lessons are binding guidance.

## Building this documentation site

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and deployed to GitHub Pages by `.github/workflows/docs.yml` on every push
to `main` that touches docs, and on every published release (ADR-50).

```bash
uv sync --only-group docs
uv run --no-sync mkdocs build --strict   # must pass — CI treats warnings as errors
uv run --no-sync mkdocs serve            # live preview at 127.0.0.1:8000... pick another port
                                         # if the Noesis service is running: -a 127.0.0.1:8001
```

The [Releases](releases.md) page is regenerated in CI by
`scripts/docs/gen_releases.py` from the GitHub Releases API; the committed
file is the zero-release placeholder so offline builds stay green.
API-reference pages are generated from docstrings by mkdocstrings using
static analysis — the docs build never imports the package, so no ML wheels
are needed.

!!! warning "WSL2 note"
    On a Windows-mounted checkout (`/mnt/...`, DrvFs) filenames are
    case-insensitive, which can hide link-case mistakes that then fail on
    Linux CI — treat the CI strict build as truth. `mkdocs serve` file
    watching can also miss events on DrvFs; re-run the build if live reload
    misbehaves.

## Brand assets

The icon/logo/banner SVGs (dark + light) are code-generated:

```bash
python3 assets/scripts/generate_assets.py
```

`/assets` is the source of truth; `docs/assets/brand/` holds committed
copies for the docs build (MkDocs cannot serve files outside `docs/`).
After regenerating, re-copy the SVGs and re-rasterize the PNGs used by the
dashboard.
