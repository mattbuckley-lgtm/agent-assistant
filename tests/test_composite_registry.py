"""Unit tests for CompositeToolRegistry."""

import pytest

from agent.agents.card import MockToolResult
from agent.agents.composite import CompositeToolRegistry
from agent.agents.mock_registry import MockToolRegistry
from agent.core.messages import ToolResultBlock


def _echo_mock() -> MockToolRegistry:
    return MockToolRegistry(
        [MockToolResult(server="s1", tool="echo", content="echoed", description="Echo tool")]
    )


def _clock_mock() -> MockToolRegistry:
    return MockToolRegistry(
        [MockToolResult(server="s2", tool="clock", content="12:00", description="Clock tool")]
    )


def test_list_tool_specs_returns_all_tools() -> None:
    composite = CompositeToolRegistry([_echo_mock(), _clock_mock()])
    names = {s.name for s in composite.list_tool_specs()}
    assert names == {"echo", "clock"}


def test_duplicate_tool_name_raises_on_construction() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name 'echo'"):
        CompositeToolRegistry([_echo_mock(), _echo_mock()])


def test_server_for_tool_routes_to_owning_registry() -> None:
    composite = CompositeToolRegistry([_echo_mock(), _clock_mock()])
    assert composite.server_for_tool("echo") == "s1"
    assert composite.server_for_tool("clock") == "s2"


def test_server_for_unknown_tool_raises_key_error() -> None:
    composite = CompositeToolRegistry([_echo_mock()])
    with pytest.raises(KeyError, match="no registry provides tool 'unknown'"):
        composite.server_for_tool("unknown")


async def test_call_tool_routes_to_correct_registry() -> None:
    composite = CompositeToolRegistry([_echo_mock(), _clock_mock()])
    result = await composite.call_tool("s2", "clock", {})
    assert isinstance(result, ToolResultBlock)
    content = result.content
    assert isinstance(content, list)
    assert any("12:00" in str(c) for c in content)


async def test_call_unknown_tool_raises_key_error() -> None:
    composite = CompositeToolRegistry([_echo_mock()])
    with pytest.raises(KeyError):
        await composite.call_tool("s1", "nonexistent", {})


def test_empty_composite_has_no_specs() -> None:
    composite = CompositeToolRegistry([])
    assert composite.list_tool_specs() == []


def test_single_registry_passthrough() -> None:
    composite = CompositeToolRegistry([_echo_mock()])
    assert len(composite.list_tool_specs()) == 1
    assert composite.list_tool_specs()[0].name == "echo"
