"""Agent Card and related config-data models.

`AgentCard` borrows the A2A Agent Card schema shape for its descriptive fields
so cards could later be serialised to /.well-known/agent-card.json, but does
not adopt the A2A protocol, transport, or auth — this is in-process composition
only.

`MockToolResult` lives here (alongside other agent config-data) because it is
needed by `agent/agents/mock_registry.py`, which is inside `agent/` and cannot
import from `evals/`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """A named capability this agent offers, analogous to A2A AgentSkill.

    `examples` are strong few-shot signal for an LLM router: populate them
    deliberately rather than leaving them empty."""

    id: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """Lightweight descriptor for a named agent — the catalog entry."""

    name: str
    description: str
    capabilities: list[Capability] = Field(default_factory=list[Capability])
    subagents: list[str] = Field(default_factory=list[str])


class MockToolResult(BaseModel):
    """A canned tool result, used in place of a real MCP server call.

    `description`/`input_schema` are advertised to the model exactly like a
    real MCP tool's (see `agent.core.messages.ToolSpec`) -- get these right,
    or a model that takes tool schemas literally (e.g. one whose
    function-calling grammar is constrained by `input_schema`) may omit
    arguments the eval's `expected_tool_calls`/scorers expect."""

    server: str
    tool: str
    content: str
    is_error: bool = False
    description: str = ""
    input_schema: dict[str, object] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
