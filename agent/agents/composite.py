"""CompositeToolRegistry: merge N ToolRegistries into one namespace.

The loop sees a single flat namespace; this adapter fans out list/route/call
to whichever sub-registry owns each tool. Duplicate tool names across
sub-registries are rejected at construction — the parent agent's permission
allowlist then gates every call as normal.
"""

from __future__ import annotations

from agent.core.interfaces import ToolRegistry
from agent.core.messages import ToolResultBlock, ToolSpec


class CompositeToolRegistry:
    """Merges an ordered list of ToolRegistries into one flat namespace.

    Construction raises `ValueError` if the same tool name appears in more
    than one sub-registry, so the error surfaces before the loop starts rather
    than at call time.
    """

    def __init__(self, registries: list[ToolRegistry]) -> None:
        self._registries = registries
        self._tool_to_registry: dict[str, ToolRegistry] = {}
        seen: set[str] = set()
        for reg in registries:
            for spec in reg.list_tool_specs():
                if spec.name in seen:
                    raise ValueError(f"Duplicate tool name '{spec.name}' across registries")
                seen.add(spec.name)
                self._tool_to_registry[spec.name] = reg

    def list_tool_specs(self) -> list[ToolSpec]:
        return [spec for reg in self._registries for spec in reg.list_tool_specs()]

    def server_for_tool(self, tool_name: str) -> str:
        try:
            return self._tool_to_registry[tool_name].server_for_tool(tool_name)
        except KeyError:
            raise KeyError(f"no registry provides tool '{tool_name}'") from None

    async def call_tool(self, server: str, tool: str, args: dict[str, object]) -> ToolResultBlock:
        try:
            reg = self._tool_to_registry[tool]
        except KeyError:
            raise KeyError(f"no registry provides tool '{tool}'") from None
        return await reg.call_tool(server, tool, args)
