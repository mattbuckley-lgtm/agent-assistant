"""Unit tests for SubAgentToolAdapter — guards, happy path, fresh state."""

from __future__ import annotations

from agent.agents.card import AgentCard
from agent.agents.mock_registry import MockToolRegistry
from agent.agents.registry import Budget
from agent.agents.subagent_tools import SubAgentToolAdapter
from agent.core.messages import ToolResultBlock
from agent.observability.sink import InMemorySink

# ---------------------------------------------------------------------------
# Minimal stub for AgentRegistry used across guard tests
# ---------------------------------------------------------------------------


class _StubRegistry:
    """Minimal AgentRegistry stand-in: only the fields SubAgentToolAdapter needs."""

    def __init__(self, max_depth: int = 3) -> None:
        self._cards = {
            "researcher": AgentCard(name="researcher", description="Researcher agent"),
            "fact_checker": AgentCard(name="fact_checker", description="Fact checker agent"),
        }
        self.max_subagent_depth = max_depth

    def get_card(self, name: str) -> AgentCard:
        return self._cards[name]

    async def get_runtime(self, name: str) -> None:  # type: ignore[override]
        raise AssertionError("get_runtime should not be called in guard tests")


def _adapter(
    *,
    registry: _StubRegistry | None = None,
    subagent_names: list[str] | None = None,
    parent_name: str = "orchestrator",
    ancestry: tuple[str, ...] = (),
    budget: Budget | None = None,
) -> SubAgentToolAdapter:
    return SubAgentToolAdapter(
        registry=registry or _StubRegistry(),  # type: ignore[arg-type]
        subagent_names=subagent_names or ["researcher"],
        parent_name=parent_name,
        ancestry=ancestry,
        budget=budget or Budget.new(60),
        sink=InMemorySink(),
    )


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------


def test_list_tool_specs_returns_one_spec_per_subagent() -> None:
    adapter = _adapter(subagent_names=["researcher", "fact_checker"])
    specs = adapter.list_tool_specs()
    assert len(specs) == 2
    names = {s.name for s in specs}
    assert names == {"researcher", "fact_checker"}


def test_list_tool_specs_skips_unknown_agent() -> None:
    adapter = _adapter(subagent_names=["researcher", "unknown_agent"])
    specs = adapter.list_tool_specs()
    assert len(specs) == 1
    assert specs[0].name == "researcher"


def test_server_for_tool_returns_synthetic_server_id() -> None:
    adapter = _adapter()
    assert adapter.server_for_tool("researcher") == "subagent:researcher"


# ---------------------------------------------------------------------------
# Guard tests (all fail closed — is_error=True, no get_runtime call)
# ---------------------------------------------------------------------------


async def test_depth_guard_fires_when_ancestry_at_max() -> None:
    registry = _StubRegistry(max_depth=3)
    # ancestry length == max_depth → next call would be depth 4
    adapter = _adapter(registry=registry, ancestry=("a", "b", "c"))
    result = await adapter.call_tool("subagent:researcher", "researcher", {"task": "x"})
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    content = result.content
    assert isinstance(content, str)
    assert content.startswith("SUBAGENT_DEPTH_EXCEEDED:")


async def test_cycle_guard_fires_when_tool_in_ancestry() -> None:
    adapter = _adapter(ancestry=("researcher",), parent_name="orchestrator")
    result = await adapter.call_tool("subagent:researcher", "researcher", {"task": "x"})
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    content = result.content
    assert isinstance(content, str)
    assert content.startswith("SUBAGENT_CYCLE_DETECTED:")


async def test_cycle_guard_fires_when_tool_is_parent_itself() -> None:
    adapter = _adapter(ancestry=(), parent_name="researcher")
    result = await adapter.call_tool("subagent:researcher", "researcher", {"task": "x"})
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    content = result.content
    assert isinstance(content, str)
    assert content.startswith("SUBAGENT_CYCLE_DETECTED:")


async def test_budget_guard_fires_when_exhausted() -> None:
    adapter = _adapter(budget=Budget.new(0))
    result = await adapter.call_tool("subagent:researcher", "researcher", {"task": "x"})
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    content = result.content
    assert isinstance(content, str)
    assert content.startswith("SUBAGENT_BUDGET_EXHAUSTED:")


# ---------------------------------------------------------------------------
# Happy path: a successful sub-agent call via replay model
# ---------------------------------------------------------------------------


async def test_successful_call_returns_final_text() -> None:
    """End-to-end: SubAgentToolAdapter calls a sub-agent via run_agent with a
    replay model. Verifies fresh task state and non-error result."""
    from pathlib import Path

    from agent.agents.card import AgentCard
    from agent.agents.registry import AgentRuntime
    from agent.config import AgentSettings
    from agent.mcp.permissions import AllowlistPolicy
    from agent.models.replay import ReplayModel
    from agent.skills.registry import EmptySkillRegistry

    cassette = Path(__file__).parent / "cassettes" / "researcher.json"

    class _FullRegistry(_StubRegistry):
        def __init__(self) -> None:
            super().__init__(max_depth=3)
            self._settings = AgentSettings()  # type: ignore[call-arg]

        async def get_runtime(self, name: str) -> AgentRuntime:  # type: ignore[override]
            return AgentRuntime(
                card=AgentCard(name=name, description="test"),
                settings=self._settings,
                model=ReplayModel(cassette, name="replay"),
                mcp_tools=MockToolRegistry([]),
                skills=EmptySkillRegistry(),
                permissions=AllowlistPolicy([]),
            )

    registry = _FullRegistry()
    adapter = SubAgentToolAdapter(
        registry=registry,  # type: ignore[arg-type]
        subagent_names=["researcher"],
        parent_name="orchestrator",
        ancestry=(),
        budget=Budget.new(60),
        sink=InMemorySink(),
    )

    result = await adapter.call_tool("subagent:researcher", "researcher", {"task": "research X"})
    assert isinstance(result, ToolResultBlock)
    assert not result.is_error
    content = result.content
    assert isinstance(content, str)
    assert len(content) > 0
