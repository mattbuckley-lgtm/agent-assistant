"""Shared entrypoint for `mcp_servers/*/server.py`.

`MCP_TRANSPORT=stdio` (the default) runs as a subprocess over stdio --
spawned in-process by `agent/mcp/client.py`, used by local dev, tests, and
the eval suite (`agent.toml`). `MCP_TRANSPORT=streamable-http` runs as its
own HTTP service -- used by the containerized deployment
(`deploy/agent.container.toml`, `deploy/compose.yaml`), bound to
`MCP_HOST`/`MCP_PORT` (default `0.0.0.0:8000`, i.e. reachable from other
containers).

Adding a new MCP server is then just: write the `FastMCP` instance + tools,
call `serve(mcp)` -- no per-server transport branching.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


def serve(mcp: FastMCP) -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        # FastMCP auto-enables DNS-rebinding protection at __init__ time,
        # allowlisting only 127.0.0.1/localhost Host headers -- which would
        # reject every request once `host` is overridden above to listen on
        # 0.0.0.0 for other containers. That protection guards browser
        # clients hitting a localhost dev server; it doesn't apply to
        # server-to-server traffic on the internal compose network, so
        # disable it here.
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
