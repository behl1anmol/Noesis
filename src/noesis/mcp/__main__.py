"""stdio MCP entry point — ``python -m noesis.mcp`` (M6).

Serves the same six tools as the HTTP mount, over stdio, for agent hosts
that spawn local servers (e.g. a Claude Code ``command`` server entry).
Core resources are built inside the FastMCP lifespan so they live on the
serving event loop and are torn down on exit. stdio never opens a socket
except the Qdrant client's localhost connection (CLAUDE.md rule 2).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastmcp import FastMCP

from noesis.core.config import load_settings
from noesis.logging_config import configure_logging
from noesis.mcp.server import build_mcp
from noesis.runtime import AppContext, build_runtime_context, close_runtime_context


def main() -> None:
    # Configure logging first — stderr only, never stdout, which carries this
    # process's JSON-RPC stream. propagate=False so a host root handler bound
    # to stdout can't receive these records and corrupt the protocol
    # (noesis.logging_config).
    configure_logging(propagate=False)
    cfg = load_settings()
    ctx_holder: list[AppContext] = []

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        ctx_holder.append(await build_runtime_context(cfg))
        try:
            yield
        finally:
            await close_runtime_context(ctx_holder[0])

    mcp = build_mcp(lambda: ctx_holder[0], lifespan=lifespan)
    mcp.run()  # stdio transport is the FastMCP default


if __name__ == "__main__":
    main()
