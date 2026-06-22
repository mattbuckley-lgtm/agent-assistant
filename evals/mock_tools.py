"""A `ToolRegistry` backed by canned results (`evals.spec.MockToolResult`)
instead of a real MCP server -- for fully offline, deterministic eval cases
(e.g. a skill that calls `clock`, where the real result is the current time).
"""

from __future__ import annotations

from types import TracebackType

from agent.core.messages import ToolResultBlock, ToolSpec
from evals.spec import MockToolResult


class MockToolRegistry:
    """Implements `agent.core.interfaces.ToolRegistry` with pre-recorded
    results: one `MockToolResult` per (server, tool), returned regardless of
    call arguments."""

    def __init__(self, mocks: list[MockToolResult]) -> None:
        self._mocks = {(m.server, m.tool): m for m in mocks}
        self._tool_to_server = {m.tool: m.server for m in mocks}
        self._tool_specs = [
            ToolSpec(name=m.tool, description=m.description, input_schema=m.input_schema)
            for m in mocks
        ]

    async def __aenter__(self) -> MockToolRegistry:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def list_tool_specs(self) -> list[ToolSpec]:
        return self._tool_specs

    def server_for_tool(self, tool_name: str) -> str:
        try:
            return self._tool_to_server[tool_name]
        except KeyError:
            raise KeyError(f"no mock tool result registered for '{tool_name}'") from None

    async def call_tool(self, server: str, tool: str, args: dict[str, object]) -> ToolResultBlock:
        mock = self._mocks[(server, tool)]
        return ToolResultBlock(
            tool_use_id="",
            content=[{"type": "text", "text": mock.content}],
            is_error=mock.is_error,
        )
