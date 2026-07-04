"""MCP adapter — thin FastMCP tools over core/ (M6, §3.1).

Primary agent interface. ``server.build_mcp`` creates the tool server;
``noesis.app`` mounts its streamable-HTTP app at ``/mcp`` with a shared
lifespan, and ``python -m noesis.mcp`` serves the same tools over stdio.
"""

from noesis.mcp.server import build_mcp

__all__ = ["build_mcp"]
