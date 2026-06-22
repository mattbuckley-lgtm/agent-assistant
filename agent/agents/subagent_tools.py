"""SubAgentToolAdapter: adapts declared sub-agents into the ToolRegistry interface.

The orchestrator's loop sees one flat tool namespace (via CompositeToolRegistry).
This adapter fills the "sub-agent" slice: each declared sub-agent is a ToolSpec
whose `call_tool` runs the full agent loop via the existing `run_agent` entrypoint.

Guard chain (fail-closed, in order):
  1. depth   — ancestry length vs max_subagent_depth
  2. cycle   — tool name already in ancestry or is the parent itself
  3. budget  — tree-wide steps remaining

Guard signals are stable string prefixes in the ToolResultBlock content so
scorers can match them without parsing arbitrary text:
  SUBAGENT_DEPTH_EXCEEDED:<detail>
  SUBAGENT_CYCLE_DETECTED:<detail>
  SUBAGENT_BUDGET_EXHAUSTED:<detail>
"""

from __future__ import annotations

import uuid

from agent.agents.registry import AgentRegistry, Budget
from agent.core.entrypoint import run_agent
from agent.core.interfaces import ToolRegistry, TranscriptSink
from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec
from agent.core.state import Task

_TASK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task to delegate to this sub-agent.",
        }
    },
    "required": ["task"],
}


class SubAgentToolAdapter:
    """Implements ToolRegistry; exposes declared sub-agents as callable tools.

    One instance per parent agent. The parent's CompositeToolRegistry merges
    this with its MCPToolRegistry so the loop sees one flat namespace.

    Each `call_tool` invocation runs the child through the *unchanged*
    `run_agent` entry point on a fresh Task — clean conversation state, no
    bleed between unrelated calls.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        subagent_names: list[str],
        parent_name: str,
        ancestry: tuple[str, ...],
        budget: Budget,
        sink: TranscriptSink,
    ) -> None:
        self._registry = registry
        self._subagent_names = subagent_names
        self._parent_name = parent_name
        self._ancestry = ancestry
        self._budget = budget
        self._sink = sink

    def list_tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name in self._subagent_names:
            try:
                card = self._registry.get_card(name)
            except KeyError:
                continue
            specs.append(
                ToolSpec(
                    name=name,
                    description=card.description,
                    input_schema=_TASK_SCHEMA,
                )
            )
        return specs

    def server_for_tool(self, tool_name: str) -> str:
        return f"subagent:{tool_name}"

    async def call_tool(self, server: str, tool: str, args: dict[str, object]) -> ToolResultBlock:
        # Guard 1: depth
        if len(self._ancestry) >= self._registry.max_subagent_depth:
            return ToolResultBlock(
                tool_use_id="",
                content=(
                    f"SUBAGENT_DEPTH_EXCEEDED: depth {len(self._ancestry) + 1} exceeds"
                    f" max_subagent_depth={self._registry.max_subagent_depth}"
                ),
                is_error=True,
            )

        # Guard 2: cycle
        full_chain = self._ancestry + (self._parent_name,)
        if tool in full_chain:
            return ToolResultBlock(
                tool_use_id="",
                content=(
                    f"SUBAGENT_CYCLE_DETECTED: '{tool}' already in ancestry {list(full_chain)}"
                ),
                is_error=True,
            )

        # Guard 3: budget
        if self._budget.is_exhausted:
            return ToolResultBlock(
                tool_use_id="",
                content=(
                    f"SUBAGENT_BUDGET_EXHAUSTED: 0 steps remaining of {self._budget.max_steps}"
                ),
                is_error=True,
            )

        runtime = await self._registry.get_runtime(tool)
        allocated = self._budget.allocate(runtime.settings.max_steps)
        if allocated == 0:
            return ToolResultBlock(
                tool_use_id="",
                content=(
                    f"SUBAGENT_BUDGET_EXHAUSTED: 0 steps remaining of {self._budget.max_steps}"
                ),
                is_error=True,
            )

        child_ancestry = full_chain
        child_adapter = SubAgentToolAdapter(
            registry=self._registry,
            subagent_names=runtime.card.subagents,
            parent_name=tool,
            ancestry=child_ancestry,
            budget=self._budget,
            sink=self._sink,
        )

        from agent.agents.composite import CompositeToolRegistry  # local to avoid circular

        child_tools: ToolRegistry
        if child_adapter.list_tool_specs():
            child_tools = CompositeToolRegistry([runtime.mcp_tools, child_adapter])
        else:
            child_tools = runtime.mcp_tools

        task_text = str(args.get("task", ""))
        task = Task(
            id=f"{tool}:{uuid.uuid4()}",
            system_prompt=runtime.settings.system_prompt,
            messages=[Message(role="user", content=[TextBlock(text=task_text)])],
        )

        result = await run_agent(
            task,
            model=runtime.model,
            tools=child_tools,
            skills=runtime.skills,
            permissions=runtime.permissions,
            sink=self._sink,
            max_steps=allocated,
        )

        self._budget.consume(allocated)

        text = result.final_text()
        return ToolResultBlock(
            tool_use_id="",
            content=text if text else f"Sub-agent '{tool}' completed with no output.",
            is_error=False,
        )
