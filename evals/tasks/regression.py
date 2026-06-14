"""Phase 4 regression suite: record/replay cassettes that pin down behaviour
across multiple MCP servers and the loop-detection guard rail, driven through
`run_agent` via `evals.bridge.run_agent_solver`.

Run with: `uv run inspect eval evals/tasks/regression.py`
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from agent.config import MCPServerConfig
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.skills.registry import EmptySkillRegistry
from evals.bridge import AgentDeps, run_agent_solver
from evals.scorers import stop_reason_is

CASSETTES = Path(__file__).parent.parent.parent / "tests" / "cassettes"

ECHO_CLOCK_SERVER = MCPServerConfig(
    name="echo-clock",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.echo_clock.server"],
)

WORDCOUNT_SERVER = MCPServerConfig(
    name="wordcount",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.wordcount.server"],
)


@asynccontextmanager
async def _wordcount_deps() -> AsyncGenerator[AgentDeps]:
    model = ReplayModel(CASSETTES / "wordcount.json")
    permissions = AllowlistPolicy([AllowRule(server="wordcount", tool="count_words")])
    async with MCPToolRegistry([WORDCOUNT_SERVER]) as tools:
        yield model, tools, EmptySkillRegistry(), permissions


@asynccontextmanager
async def _loop_detection_deps() -> AsyncGenerator[AgentDeps]:
    model = ReplayModel(CASSETTES / "loop_detection.json")
    permissions = AllowlistPolicy([AllowRule(server="echo-clock", tool="echo")])
    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        yield model, tools, EmptySkillRegistry(), permissions


@task
def regression_wordcount() -> Task:
    """A second MCP server (`wordcount`) is dispatched correctly end-to-end."""
    return Task(
        dataset=[Sample(input="How many words are in 'the quick brown fox jumps'?", target="5")],
        solver=run_agent_solver(
            _wordcount_deps,
            system_prompt="You are a helpful agent with access to a word-counting tool.",
        ),
        scorer=includes(),
    )


@task
def regression_loop_detection() -> Task:
    """A model that repeats the same tool call is stopped by the
    `MAX_REPEATED_TOOL_CALLS` guard rail rather than looping forever."""
    return Task(
        dataset=[Sample(input="Please echo 'loop' repeatedly.", target="")],
        solver=run_agent_solver(
            _loop_detection_deps,
            system_prompt="You are a helpful agent with access to an echo tool.",
        ),
        scorer=stop_reason_is("loop_detected"),
    )
