"""The seams: Protocols implemented by every pluggable component.

`run_agent` (agent/core/entrypoint.py) is constructed entirely from these
Protocols. Production wiring, replay/regression, and Inspect-driven evals
all provide different implementations of the same Protocols -- the loop
(agent/core/loop.py) never imports a concrete implementation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, runtime_checkable

from agent.core.events import Decision, TranscriptEvent
from agent.core.messages import Message, ToolResultBlock, ToolSpec
from agent.models.base import StreamEvent
from agent.skills.base import Skill


@runtime_checkable
class Model(Protocol):
    """A swappable LLM backend. Adapters translate normalized
    `Message`/`ToolSpec` to provider requests and provider streams back to
    normalized `StreamEvent`s -- the loop never sees provider specifics."""

    name: str

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Discovers tool specs and dispatches calls to the MCP layer.

    `server_for_tool` lets the loop look up which MCP server backs a given
    tool name *before* asking the permission policy to evaluate it.
    """

    def list_tool_specs(self) -> list[ToolSpec]: ...

    def server_for_tool(self, tool_name: str) -> str: ...

    async def call_tool(
        self, server: str, tool: str, args: dict[str, object]
    ) -> ToolResultBlock: ...


@runtime_checkable
class SkillRegistry(Protocol):
    """Discovers skills for the system-prompt index and loads bodies
    on demand (progressive disclosure)."""

    def list_skills(self) -> list[Skill]: ...

    def get_skill(self, name: str) -> Skill | None: ...


@runtime_checkable
class PermissionPolicy(Protocol):
    """Gates every tool call before it crosses to the (isolated) MCP server."""

    def evaluate(self, server: str, tool: str, args: dict[str, object]) -> Decision: ...


# HITL hook used when a PermissionPolicy.evaluate(...) returns Decision.PROMPT.
# Args: (server, tool, args) -> approved?
HumanApproval = Callable[[str, str, dict[str, object]], Awaitable[bool]]


@runtime_checkable
class TranscriptSink(Protocol):
    """Receives every transcript event. Implementations may fan out (e.g. an
    in-memory sink for evals composed with an OTel-emitting sink)."""

    async def emit(self, event: TranscriptEvent) -> None: ...
