"""Phase 1 checkpoint: replay model + MCP echo/clock server + allowlist
policy + agent loop, producing a full transcript."""

import sys
from pathlib import Path

from agent.config import MCPServerConfig
from agent.core.entrypoint import run_agent
from agent.core.events import (
    Decision,
    Error,
    PermissionDecided,
    ToolCallFinished,
    ToolCallRequested,
)
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from agent.skills.registry import EmptySkillRegistry

CASSETTE = Path(__file__).parent / "cassettes" / "echo_clock.json"

ECHO_CLOCK_SERVER = MCPServerConfig(
    name="echo-clock",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.echo_clock.server"],
)


async def test_agent_loop_calls_allowed_mcp_tool_and_answers() -> None:
    model = ReplayModel(CASSETTE)
    permissions = AllowlistPolicy([AllowRule(server="echo-clock", tool="echo")])
    sink = InMemorySink()
    task = Task(
        id="echo-hello",
        system_prompt="You are a helpful agent with access to an echo tool.",
        messages=[Message(role="user", content=[TextBlock(text="Please echo 'hello'.")])],
    )

    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        result = await run_agent(
            task,
            model=model,
            tools=tools,
            skills=EmptySkillRegistry(),
            permissions=permissions,
            sink=sink,
        )

    assert result.stop_reason == "end_turn"
    assert result.final_message is not None
    final_text = result.final_message.content[0]
    assert isinstance(final_text, TextBlock)
    assert "hello" in final_text.text

    # Aggregate usage summed across both model calls.
    assert result.usage.input_tokens == 230
    assert result.usage.output_tokens == 30

    requested = [e for e in sink.events if isinstance(e, ToolCallRequested)]
    assert requested == [
        ToolCallRequested(
            run_id=result.run_id,
            step_index=0,
            ts=requested[0].ts,
            tool_use_id="call_1",
            server="echo-clock",
            tool="echo",
            args={"text": "hello"},
        )
    ]

    decided = [e for e in sink.events if isinstance(e, PermissionDecided)]
    assert len(decided) == 1
    assert decided[0].decision == Decision.ALLOW
    assert decided[0].allowed is True

    finished = [e for e in sink.events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].is_error is False
    assert finished[0].result.content == [
        {"type": "text", "text": "hello", "annotations": None, "meta": None}
    ]

    # entrypoint's recorded transcript matches what was emitted to the sink.
    assert result.transcript == sink.events


async def test_denied_tool_call_never_executes() -> None:
    model = ReplayModel(CASSETTE)
    permissions = AllowlistPolicy([])  # default-deny: no rules at all
    sink = InMemorySink()
    task = Task(
        id="echo-hello-denied",
        messages=[Message(role="user", content=[TextBlock(text="Please echo 'hello'.")])],
    )

    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        result = await run_agent(
            task,
            model=model,
            tools=tools,
            skills=EmptySkillRegistry(),
            permissions=permissions,
            sink=sink,
        )

    decided = [e for e in sink.events if isinstance(e, PermissionDecided)]
    assert len(decided) == 1
    assert decided[0].decision == Decision.DENY
    assert decided[0].allowed is False

    # The MCP server was never actually invoked.
    assert not any(isinstance(e, ToolCallFinished) for e in sink.events)

    # The denial is surfaced to the model as a tool error, which the
    # cassette's second turn echoes back -- proving the run continues safely.
    assert result.stop_reason == "end_turn"


async def test_deny_rule_overrides_allow_rule_for_same_tool() -> None:
    """A DENY rule beats an ALLOW rule for the same tool regardless of order --
    deny-priority semantics, same as AWS IAM / firewall models."""
    model = ReplayModel(CASSETTE)
    permissions = AllowlistPolicy(
        [
            AllowRule(server="echo-clock", tool="echo"),  # ALLOW first
            AllowRule(server="echo-clock", tool="echo", decision=Decision.DENY),  # DENY second
        ]
    )
    sink = InMemorySink()
    task = Task(
        id="echo-hello-deny-wins",
        messages=[Message(role="user", content=[TextBlock(text="Please echo 'hello'.")])],
    )

    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        await run_agent(
            task,
            model=model,
            tools=tools,
            skills=EmptySkillRegistry(),
            permissions=permissions,
            sink=sink,
        )

    decided = [e for e in sink.events if isinstance(e, PermissionDecided)]
    assert len(decided) == 1
    assert decided[0].decision == Decision.DENY
    assert decided[0].allowed is False
    assert not any(isinstance(e, ToolCallFinished) for e in sink.events)


async def test_repeated_identical_tool_call_triggers_loop_detection() -> None:
    cassette = Path(__file__).parent / "cassettes" / "loop_detection.json"
    model = ReplayModel(cassette)
    permissions = AllowlistPolicy([AllowRule(server="echo-clock", tool="echo")])
    sink = InMemorySink()
    task = Task(
        id="echo-loop",
        system_prompt="You are a helpful agent with access to an echo tool.",
        messages=[Message(role="user", content=[TextBlock(text="Please echo 'loop' repeatedly.")])],
    )

    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        result = await run_agent(
            task,
            model=model,
            tools=tools,
            skills=EmptySkillRegistry(),
            permissions=permissions,
            sink=sink,
        )

    assert result.stop_reason == "loop_detected"

    # The first three identical calls execute; the fourth is caught by the
    # guard rail and never reaches the MCP server.
    finished = [e for e in sink.events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 3

    errors = [e for e in sink.events if isinstance(e, Error)]
    assert len(errors) == 1
    assert errors[0].where == "loop_detection"
