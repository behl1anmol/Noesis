"""FastMCP tool server — all six M6 tools as thin wrappers over core/.

Every tool body is the same core call its REST twin makes (routes.py), and
the success payloads are identical dicts — tests assert byte-equality so
the two surfaces cannot drift (§ M6). Failures raise ``ToolError`` carrying
the same detail REST puts in the HTTP error body.

The server is transport-agnostic: ``noesis.app`` mounts ``http_app()`` at
``/mcp`` (streamable HTTP), ``python -m noesis.mcp`` runs stdio. Tools
reach the shared core resources through ``get_ctx`` — a callable, not an
import, so this module never imports ``noesis.app`` (no cycle) and stdio
can supply a context of its own.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from noesis.core import jobs, state
from noesis.core import retriever
from noesis.core import structural as structural_mod


def build_mcp(
    get_ctx: Callable[[], Any], *, lifespan: Any | None = None
) -> FastMCP:
    """Create the noesis MCP server. ``get_ctx()`` must return the live
    AppContext (conn, store, embedder, reranker, structural settings) —
    called per tool invocation so the context can be built after the
    server (lifespan order is the transport's concern). ``lifespan`` is
    the stdio entry point's hook for building that context (the HTTP
    mount builds it in the FastAPI lifespan instead)."""
    mcp = FastMCP("noesis", lifespan=lifespan)

    @mcp.tool
    async def search_code(
        query: str,
        project_id: str,
        top_k: int = Field(default=10, ge=1, le=100),
        language: str | None = None,
        channel: Literal["hybrid", "dense", "sparse"] = "hybrid",
        rerank: bool | None = None,
    ) -> dict[str, Any]:
        """Semantic + lexical (hybrid) search over an indexed project.

        Returns ranked spans: {chunk_id, file_path, start_line, end_line,
        language, symbol_name, score, snippet}. Results are candidates,
        not ground truth — read the live file before acting on a span.
        Use get_chunk(chunk_id) to fetch a hit's full stored text.
        """
        ctx = get_ctx()
        if state.get_project(ctx.conn, project_id) is None:
            raise ToolError("unknown project_id")
        result = await retriever.search_code(
            ctx.store,
            ctx.embedder,
            query,
            project_id,
            top_k=top_k,
            language=language,
            channel=channel,
            reranker=ctx.reranker,
            rerank=rerank,
            candidates=ctx.rerank_candidates,
        )
        return {
            "query": query,
            "channel": channel,
            "reranked": result["reranked"],
            "hits": result["hits"],
        }

    @mcp.tool
    async def structural_search(
        pattern: str,
        language: str,
        project_id: str,
        paths: list[str] | None = None,
        max_results: int | None = Field(default=None, ge=1),
    ) -> dict[str, Any]:
        """AST-pattern search (ast-grep) over the project's live files.

        `pattern` is an ast-grep pattern, e.g. "def $NAME($$$ARGS): $$$BODY"
        (Python) or "console.log($$$A)" (TypeScript). Matches are exact
        syntax-tree matches, not text. `paths` restricts to relative
        path prefixes. Scans the live filesystem — results are current
        even if the index is stale.
        """
        ctx = get_ctx()
        try:
            result = await structural_mod.structural_search(
                ctx.conn,
                project_id,
                pattern,
                language,
                paths=paths,
                max_results=max_results,
                settings=ctx.structural,
            )
        except structural_mod.StructuralSearchError as exc:
            raise ToolError(f"{exc.error_type}: {exc.message}") from exc
        return {"pattern": pattern, "language": language, **result}

    @mcp.tool
    async def list_projects() -> list[dict[str, Any]]:
        """List registered projects with root path, embedding model and
        timestamps. Project ids from here feed every other tool."""
        ctx = get_ctx()
        return [dict(row) for row in state.list_projects(ctx.conn)]

    @mcp.tool
    async def get_index_status(project_id: str) -> dict[str, Any]:
        """Status of the most recent index run for a project: status
        (never_indexed | running | done | failed), file/chunk counts,
        timestamps, error. Poll after reindex until status is done."""
        ctx = get_ctx()
        if state.get_project(ctx.conn, project_id) is None:
            raise ToolError("unknown project_id")
        return jobs.index_status(ctx, project_id)

    @mcp.tool
    async def get_chunk(chunk_id: str) -> dict[str, Any]:
        """Fetch one indexed chunk by id (ids come from search_code hits).

        Returns the exact stored span with full chunk content — the
        indexed snapshot, which may lag the live file.
        """
        ctx = get_ctx()
        chunk = ctx.store.get_chunk(chunk_id)
        if chunk is None:
            raise ToolError("unknown chunk_id")
        return chunk

    @mcp.tool
    async def reindex(project_id: str) -> dict[str, str]:
        """Re-index a registered project (incremental — only changed files
        are re-embedded). Returns immediately with a run_id; poll
        get_index_status for completion."""
        ctx = get_ctx()
        project = state.get_project(ctx.conn, project_id)
        if project is None:
            raise ToolError("unknown project_id")
        try:
            return jobs.launch_index_run(ctx, project["root_path"])
        except ValueError as exc:  # mixed-model guard
            raise ToolError(str(exc)) from exc

    return mcp
