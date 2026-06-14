"""MCP client connection management.

Each `MCPServerConnection` owns one transport + `ClientSession` to a single
(process-isolated) MCP server. `agent/mcp/registry.py` builds the
`ToolRegistry` on top of one or more connections.

Two transports are supported: `stdio` spawns the server as a subprocess
(used for local dev/tests/evals -- fast, hermetic, no ports); `streamable_http`
connects to a server running as its own (container-isolated) service. Both
configure the *same* `ToolRegistry`/`ToolSpec`/`call_tool` surface, so
`agent/core` and `agent/mcp/registry.py` never see the difference -- only
`agent.toml` vs `agent.container.toml` changes.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage

from agent.config import MCPServerConfig

_ReadStream = MemoryObjectReceiveStream[SessionMessage | Exception]
_WriteStream = MemoryObjectSendStream[SessionMessage]


class MCPServerConnection:
    """A connected session to one MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.name = config.name
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        if self._config.transport == "stdio":
            read_stream, write_stream = await self._connect_stdio()
        elif self._config.transport == "streamable_http":
            read_stream, write_stream = await self._connect_streamable_http()
        else:
            raise NotImplementedError(
                f"MCP transport '{self._config.transport}' is not yet supported "
                "(only 'stdio' and 'streamable_http' are implemented)"
            )

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._session = session

    async def _connect_stdio(self) -> tuple[_ReadStream, _WriteStream]:
        if self._config.command is None:
            raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")

        params = StdioServerParameters(command=self._config.command, args=self._config.args)
        return await self._exit_stack.enter_async_context(stdio_client(params))

    async def _connect_streamable_http(self) -> tuple[_ReadStream, _WriteStream]:
        if self._config.url is None:
            raise ValueError(f"MCP server '{self.name}': streamable_http transport requires 'url'")

        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamable_http_client(self._config.url)
        )
        return read_stream, write_stream

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool: str, arguments: dict[str, object]) -> types.CallToolResult:
        return await self.session.call_tool(tool, arguments)
