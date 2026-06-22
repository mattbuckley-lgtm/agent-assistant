"""Agent catalog: loads agent TOMLs, builds runtimes lazily, caches per session.

`AgentRegistry` is the single source of truth for what agents exist and how to
build them. It is an async context manager because sub-agent MCP connections
are expensive -- they connect on first demand and are kept alive for the
session rather than torn down/rebuilt per call.

`Budget` is the tree-wide step budget threaded from the orchestrator down
through every `SubAgentToolAdapter.call_tool` invocation, ensuring the whole
tree terminates even if individual agents hit their `max_steps`.
"""

from __future__ import annotations

import tomllib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from agent.agents.card import AgentCard
from agent.agents.mock_registry import MockToolRegistry
from agent.composition import build_model, build_permissions, build_skills
from agent.config import AgentSettings
from agent.core.interfaces import Model, PermissionPolicy, SkillRegistry, ToolRegistry
from agent.mcp.registry import MCPToolRegistry

# Type alias to avoid importing at module top inside method
_ModelOverrides = dict[str, Model]


@dataclass
class Budget:
    """Tree-wide step budget shared across all agents in a single run.

    Initialise once at the root; `SubAgentToolAdapter` threads it down through
    every child call, decrementing as each child run finishes.
    """

    max_steps: int
    steps_remaining: int

    @classmethod
    def new(cls, max_steps: int) -> Budget:
        return cls(max_steps=max_steps, steps_remaining=max_steps)

    @property
    def is_exhausted(self) -> bool:
        return self.steps_remaining <= 0

    def allocate(self, n: int) -> int:
        """Return how many steps to allow: min(requested, remaining), >= 0."""
        return min(n, max(0, self.steps_remaining))

    def consume(self, steps: int) -> None:
        self.steps_remaining = max(0, self.steps_remaining - steps)


@dataclass
class AgentRuntime:
    """The expensive-to-build parts of an agent, cached for the session."""

    card: AgentCard
    settings: AgentSettings
    model: Model
    mcp_tools: ToolRegistry
    skills: SkillRegistry
    permissions: PermissionPolicy


class AgentRegistry:
    """Catalog of named agents loaded from a directory of `*.toml` files.

    Cards are built eagerly (they're cheap). Runtimes (model + MCP connections
    + skills + permissions) are built lazily on the first `get_runtime` call
    and cached -- MCP connections stay alive for the lifetime of the registry
    context.

    `mock_registry_by_agent` lets callers (typically eval solvers) inject a
    pre-built `ToolRegistry` per agent name instead of connecting to real MCP
    servers. `default_to_empty_mocks=True` makes every agent not in the map
    fall back to an empty `MockToolRegistry` -- useful in replay-mode evals
    where no live MCP server should be touched.
    """

    def __init__(
        self,
        agents_dir: Path,
        base_settings: AgentSettings,
        *,
        mock_registry_by_agent: dict[str, ToolRegistry] | None = None,
        mock_model_by_agent: _ModelOverrides | None = None,
        default_to_empty_mocks: bool = False,
    ) -> None:
        self._base_settings = base_settings
        self._mock_by_agent: dict[str, ToolRegistry] = mock_registry_by_agent or {}
        self._mock_model_by_agent: _ModelOverrides = mock_model_by_agent or {}
        self._default_to_empty_mocks = default_to_empty_mocks
        self._agents: dict[str, tuple[AgentCard, AgentSettings]] = {}
        self._cache: dict[str, AgentRuntime] = {}
        self._exit_stack = AsyncExitStack()
        self._scan(agents_dir)

    def _scan(self, agents_dir: Path) -> None:
        for toml_path in sorted(agents_dir.glob("*.toml")):
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            settings = AgentSettings(**data)  # type: ignore[call-arg]
            card = AgentCard(
                name=settings.name or toml_path.stem,
                description=settings.description,
                capabilities=settings.capabilities,
                subagents=settings.subagents,
            )
            self._agents[card.name] = (card, settings)

    async def __aenter__(self) -> AgentRegistry:
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._exit_stack.__aexit__(exc_type, exc, tb)

    def list_cards(self) -> list[AgentCard]:
        return [card for card, _ in self._agents.values()]

    def get_card(self, name: str) -> AgentCard:
        try:
            return self._agents[name][0]
        except KeyError:
            available = ", ".join(sorted(self._agents)) or "(none)"
            raise KeyError(f"Unknown agent '{name}'. Available: {available}") from None

    async def get_runtime(self, name: str) -> AgentRuntime:
        if name not in self._agents:
            available = ", ".join(sorted(self._agents)) or "(none)"
            raise KeyError(f"Unknown agent '{name}'. Available: {available}")

        if name not in self._cache:
            card, settings = self._agents[name]
            model = (
                self._mock_model_by_agent[name]
                if name in self._mock_model_by_agent
                else build_model(settings.resolve_model(None))
            )
            permissions = build_permissions(settings)
            skills = build_skills(settings)

            mcp_tools: ToolRegistry
            if name in self._mock_by_agent:
                mcp_tools = self._mock_by_agent[name]
            elif self._default_to_empty_mocks:
                mcp_tools = MockToolRegistry([])
            else:
                mcp = MCPToolRegistry(settings.mcp_servers)
                mcp_tools = await self._exit_stack.enter_async_context(mcp)

            self._cache[name] = AgentRuntime(
                card=card,
                settings=settings,
                model=model,
                mcp_tools=mcp_tools,
                skills=skills,
                permissions=permissions,
            )

        return self._cache[name]

    @property
    def max_subagent_depth(self) -> int:
        return self._base_settings.max_subagent_depth
