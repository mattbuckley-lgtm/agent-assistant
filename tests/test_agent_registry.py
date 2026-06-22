"""Unit tests for AgentRegistry — card scanning, lazy build, cache, mocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.agents.card import AgentCard
from agent.agents.mock_registry import MockToolRegistry
from agent.agents.registry import AgentRegistry, Budget
from agent.config import AgentSettings
from agent.models.replay import ReplayModel

CASSETTE = Path(__file__).parent / "cassettes" / "researcher.json"


@pytest.fixture
def base_settings() -> AgentSettings:
    return AgentSettings()  # type: ignore[call-arg]


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    (tmp_path / "alpha.toml").write_text(
        'name = "alpha"\ndescription = "Alpha agent"\n'
        'system_prompt = "You are alpha."\nsubagents = ["beta"]\n'
    )
    (tmp_path / "beta.toml").write_text(
        'name = "beta"\ndescription = "Beta agent"\nsystem_prompt = "You are beta."\n'
    )
    return tmp_path


async def test_list_cards_returns_all_scanned_agents(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    async with AgentRegistry(agents_dir, base_settings) as reg:
        cards = reg.list_cards()
    names = {c.name for c in cards}
    assert names == {"alpha", "beta"}


async def test_get_card_returns_correct_card(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    async with AgentRegistry(agents_dir, base_settings) as reg:
        card = reg.get_card("alpha")
    assert isinstance(card, AgentCard)
    assert card.name == "alpha"
    assert card.description == "Alpha agent"
    assert card.subagents == ["beta"]


async def test_get_card_unknown_raises_key_error(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    async with AgentRegistry(agents_dir, base_settings) as reg:
        with pytest.raises(KeyError, match="Unknown agent 'nonexistent'"):
            reg.get_card("nonexistent")


async def test_get_runtime_raises_for_unknown_agent(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    async with AgentRegistry(agents_dir, base_settings) as reg:
        with pytest.raises(KeyError, match="Unknown agent 'ghost'"):
            await reg.get_runtime("ghost")


async def test_get_runtime_is_lazy_and_cached(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    """Runtime must be built only once per name, not on each call."""
    mock_model = ReplayModel(CASSETTE, name="replay")
    async with AgentRegistry(
        agents_dir,
        base_settings,
        mock_model_by_agent={"alpha": mock_model, "beta": mock_model},
        default_to_empty_mocks=True,
    ) as reg:
        r1 = await reg.get_runtime("alpha")
        r2 = await reg.get_runtime("alpha")
        assert r1 is r2, "get_runtime must return the cached _AgentRuntime on second call"


async def test_mock_registry_by_agent_overrides_mcp_tools(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    injected = MockToolRegistry([])
    mock_model = ReplayModel(CASSETTE, name="replay")
    async with AgentRegistry(
        agents_dir,
        base_settings,
        mock_registry_by_agent={"alpha": injected},
        mock_model_by_agent={"alpha": mock_model, "beta": mock_model},
        default_to_empty_mocks=True,
    ) as reg:
        runtime = await reg.get_runtime("alpha")
    assert runtime.mcp_tools is injected


async def test_default_to_empty_mocks_gives_mock_registry(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    mock_model = ReplayModel(CASSETTE, name="replay")
    async with AgentRegistry(
        agents_dir,
        base_settings,
        mock_model_by_agent={"alpha": mock_model},
        default_to_empty_mocks=True,
    ) as reg:
        runtime = await reg.get_runtime("alpha")
    assert isinstance(runtime.mcp_tools, MockToolRegistry)
    assert runtime.mcp_tools.list_tool_specs() == []


async def test_max_subagent_depth_from_base_settings(
    agents_dir: Path,
) -> None:
    settings = AgentSettings(max_subagent_depth=7)  # type: ignore[call-arg]
    async with AgentRegistry(agents_dir, settings) as reg:
        assert reg.max_subagent_depth == 7


async def test_context_manager_exits_cleanly(
    agents_dir: Path, base_settings: AgentSettings
) -> None:
    """Entering and exiting the context manager must not raise."""
    async with AgentRegistry(agents_dir, base_settings):
        pass


async def test_budget_new_and_allocate() -> None:
    budget = Budget.new(10)
    assert not budget.is_exhausted
    assert budget.allocate(5) == 5
    budget.consume(5)
    assert budget.allocate(10) == 5  # only 5 remaining
    budget.consume(5)
    assert budget.is_exhausted
    assert budget.allocate(1) == 0
