"""A second trivial MCP server, demonstrating that adding a server is a
config-only change (see `agent.toml` and `agent/mcp/registry.py`): one tool,
`count_words`.

Run directly: `uv run python -m mcp_servers.wordcount.server`
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers._runtime import serve

mcp = FastMCP("wordcount")


@mcp.tool()
def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in `text`."""
    return len(text.split())


if __name__ == "__main__":
    serve(mcp)
