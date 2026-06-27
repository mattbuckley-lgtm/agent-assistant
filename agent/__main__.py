"""Composition root CLI: `uv run python -m agent "<prompt>"`.

Loads `AgentSettings` (agent.toml + env), wires up a `Model`, MCP tools,
skills, permissions, and an `OtelSink`, runs one task through `run_agent`,
and prints the result.

`--chat` starts an interactive, streaming REPL instead (see `run_chat`).
`--model <key>` selects a backend from the `[models]` registry in
agent.toml (default: `default_model`).
`--agent <name>` loads a named agent from `agents_dir` with sub-agent support.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass

from agent.agents.composite import CompositeToolRegistry
from agent.agents.registry import AgentRegistry, Budget
from agent.agents.subagent_tools import SubAgentToolAdapter
from agent.composition import (
    build_memory_sink,
    build_model,
    build_permissions,
    build_skills,
    memory_context,
)
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.interfaces import (
    MemoryProvider,
    Model,
    PermissionPolicy,
    SkillRegistry,
    ToolRegistry,
    TranscriptSink,
)
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.registry import MCPToolRegistry
from agent.memory.store import McpMemoryStore
from agent.models.base import Usage
from agent.observability.otel import build_tracer_provider
from agent.observability.sink import FanOutSink, OtelSink, StreamingConsoleSink


@dataclass
class _Runtime:
    model: Model
    tools: ToolRegistry
    skills: SkillRegistry
    permissions: PermissionPolicy
    memory_provider: MemoryProvider
    max_steps: int
    system_prompt: str


def _make_sink(
    *parts: TranscriptSink | None,
) -> TranscriptSink:
    """Build a FanOutSink from non-None parts, or unwrap if there's only one."""
    sinks = [s for s in parts if s is not None]
    return FanOutSink(sinks) if len(sinks) > 1 else sinks[0]


def _print_result(usage: Usage, stop_reason: str, text: str) -> None:
    print(f"stop_reason: {stop_reason}")
    print(f"usage: {usage.model_dump()}")
    print("---")
    print(text)


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    parser.add_argument("prompt", nargs="?", help="user message to send to the agent")
    parser.add_argument("--task-id", default="cli-task")
    parser.add_argument(
        "--model",
        default=None,
        help="model registry key from agent.toml [models] (default: default_model)",
    )
    parser.add_argument("--chat", action="store_true", help="interactive streaming chat REPL")
    parser.add_argument(
        "--agent",
        default=None,
        metavar="NAME",
        help="run a named agent from the agents_dir (e.g. orchestrator) with sub-agent support",
    )
    args = parser.parse_args(argv)

    if not args.chat and args.prompt is None:
        parser.error("prompt is required unless --chat is given")

    settings = AgentSettings()  # type: ignore[call-arg]  # models come from agent.toml/env

    tracer_provider = build_tracer_provider(settings.otel)
    tracer = tracer_provider.get_tracer(settings.otel.service_name)

    async with memory_context(settings) as (memory_provider, mem_store):
        if args.agent is not None:
            await _run_named_agent(args, settings, tracer, memory_provider, mem_store)
        else:
            try:
                model_config = settings.resolve_model(args.model)
            except ValueError as exc:
                parser.error(str(exc))

            rt = _Runtime(
                model=build_model(model_config),
                tools=None,  # type: ignore[arg-type]  # filled in below
                skills=build_skills(settings),
                permissions=build_permissions(settings),
                memory_provider=memory_provider,
                max_steps=settings.max_steps,
                system_prompt=settings.system_prompt,
            )
            async with MCPToolRegistry(settings.mcp_servers) as tools:
                rt.tools = tools
                otel = OtelSink(tracer)
                if args.chat:
                    base = _make_sink(StreamingConsoleSink(), otel)
                    await run_chat(rt, base, mem_store, settings)
                else:
                    assert args.prompt is not None
                    await _run_oneshot(args, rt, otel, mem_store, settings)

    tracer_provider.shutdown()
    return 0


async def _run_named_agent(
    args: argparse.Namespace,
    settings: AgentSettings,
    tracer: object,
    memory_provider: MemoryProvider,
    mem_store: McpMemoryStore | None,
) -> None:
    agents_dir = settings.agents_dir
    if agents_dir is None:
        raise SystemExit("agents_dir not set in agent.toml -- cannot use --agent")

    agent_name: str = args.agent
    # The streaming sink is shared with the sub-agent adapter so sub-agent output
    # reaches the console. MemorySink is NOT in this sink — it's added only to the
    # top-level run_agent call so sub-agent RunFinished events don't trigger writes.
    streaming_sink = _make_sink(StreamingConsoleSink(), OtelSink(tracer))  # type: ignore[arg-type]

    async with AgentRegistry(agents_dir, settings) as registry:
        runtime = await registry.get_runtime(agent_name)

        try:
            model_config = runtime.settings.resolve_model(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        budget = Budget.new(settings.max_steps * settings.max_subagent_depth)
        adapter = SubAgentToolAdapter(
            registry=registry,
            subagent_names=runtime.card.subagents,
            parent_name=agent_name,
            ancestry=(),
            budget=budget,
            sink=streaming_sink,
        )

        async with MCPToolRegistry(runtime.settings.mcp_servers) as mcp_tools:
            tools: ToolRegistry
            if adapter.list_tool_specs():
                tools = CompositeToolRegistry([mcp_tools, adapter])
            else:
                tools = mcp_tools

            rt = _Runtime(
                model=build_model(model_config),
                tools=tools,
                skills=runtime.skills,
                permissions=runtime.permissions,
                memory_provider=memory_provider,
                max_steps=runtime.settings.max_steps,
                system_prompt=runtime.settings.system_prompt,
            )

            if args.chat:
                await run_chat(rt, streaming_sink, mem_store, settings)
            else:
                assert args.prompt is not None
                await _run_oneshot(args, rt, streaming_sink, mem_store, settings)


async def _run_oneshot(
    args: argparse.Namespace,
    rt: _Runtime,
    base_sink: TranscriptSink,
    mem_store: McpMemoryStore | None,
    settings: AgentSettings,
) -> None:
    task = Task(
        id=args.task_id,
        system_prompt=rt.system_prompt,
        messages=[Message(role="user", content=[TextBlock(text=args.prompt)])],
    )
    mem_sink = build_memory_sink(settings, mem_store) if mem_store is not None else None
    result = await run_agent(
        task,
        model=rt.model,
        tools=rt.tools,
        skills=rt.skills,
        permissions=rt.permissions,
        sink=_make_sink(base_sink, mem_sink),
        max_steps=rt.max_steps,
        memory_provider=rt.memory_provider,
    )
    _print_result(result.usage, result.stop_reason, result.final_text())


async def _stdin_approval(server: str, tool: str, args: dict[str, object]) -> bool:
    """Approval hook for `Decision.PROMPT`: reads y/n from stdin."""
    import json as _json

    prompt = f"[approval required] Allow {server}.{tool}({_json.dumps(args)})? [y/N] "
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in {"y", "yes"}


async def run_chat(
    rt: _Runtime,
    base_sink: TranscriptSink,
    mem_store: McpMemoryStore | None,
    settings: AgentSettings,
) -> None:
    """Interactive REPL: each line is a user turn, streamed against a
    growing conversation. A new MemorySink is created per turn so each
    turn produces its own episodic record."""
    accumulated: list[Message] = []

    print(f"Chatting with '{rt.model.name}'. Type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            line = await asyncio.to_thread(input, "\nyou> ")
        except EOFError:
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in {"exit", "quit"}:
            break

        accumulated.append(Message(role="user", content=[TextBlock(text=line)]))

        print("agent> ", end="", flush=True)
        task = Task(
            id=str(uuid.uuid4()),
            system_prompt=rt.system_prompt,
            messages=list(accumulated),
        )
        mem_sink = build_memory_sink(settings, mem_store) if mem_store is not None else None
        result = await run_agent(
            task,
            model=rt.model,
            tools=rt.tools,
            skills=rt.skills,
            permissions=rt.permissions,
            sink=_make_sink(base_sink, mem_sink),
            approval=_stdin_approval,
            max_steps=rt.max_steps,
            memory_provider=rt.memory_provider,
        )
        print()
        if result.stop_reason not in {"end_turn", "stop"}:
            print(f"[stop_reason: {result.stop_reason}]")
        accumulated = list(result.messages)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
