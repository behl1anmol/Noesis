# Noesis (code-indexer)

Hybrid code-retrieval service for AI agents. MCP primary, REST secondary.
Local-only: no code, query, or metadata ever leaves the machine (ADR-25).

## Read first
- architecture-docs/code-indexer-expanded-architecture.md  (this plan — authoritative)
- architecture-docs/code-indexer-initial-idea.md            (approved baseline)

## Commands
- `uv run uvicorn noesis.app:app --host 127.0.0.1 --port 8000`
- `uv run pytest` | `uv run pytest tests/eval/ -m golden` (harness)
- `docker compose up -d` (Qdrant)
- `python .claude/scripts/devlog.py latest` (session state)
- `python .claude/scripts/devlog.py checkpoint latest` (most recent checkpoint)

## Hard rules
1.  All embedding calls go through core/embedder.py's Embedder Protocol; all
    rerank calls go through core/reranker.py's Reranker Protocol (ADR-33).
    Never import sentence_transformers outside those two model-loading
    boundaries. CI greps for this.
2. bind 127.0.0.1 only. Never 0.0.0.0. No outbound HTTP anywhere in core/ —
   remote embedding was rejected (ADR-25); do not reintroduce it.
3. No new runtime deps without a decision row (use /adr). astchunk is
   dev/test only.
4. Milestone exit criteria are the definition of done — update via devlog.py.
5. mcp pin: fastmcp resolves it; never float past <2 until M8 decides.
6. Every design decision gets a rationale (decisions table + doc). No
   rationale, no merge.
7. When a mistake is caught (wrong approach, broken invariant, reverted
   work, failed exit criterion), record it with /lesson BEFORE moving on.
   Active lessons in dev/LESSONS.md are binding guidance, second only to
   these hard rules. A lesson may never weaken rules 1-6.
8. Session state is checkpointed automatically before context compaction
   (PreCompact, both manual and automatic) and best-effort on an API/quota
   failure (StopFailure: rate_limit, billing_error, overloaded), into
   devlog.sqlite's checkpoints table. This does NOT cover a plan/usage-limit
   cutoff that happens mid-turn before either of those events fires — treat
   an unclosed prior session with no checkpoint as exactly that: assume
   nothing beyond the plain transcript survived, and ask before reconstructing
   intent. Use /checkpoint to force a save point before a risky step; /resume
   to recall the latest one mid-conversation. Never edit dev/devlog.sqlite
   directly — go through devlog.py.

## Layout
src/noesis/{core,api,mcp,app.py} — api/ and mcp/ are thin over core/.
