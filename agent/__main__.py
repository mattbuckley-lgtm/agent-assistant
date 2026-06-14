"""Composition root CLI: `uv run python -m agent "<prompt>"`.

Loads `AgentSettings` (agent.toml + env), wires up a `Model`, MCP tools,
skills, permissions, and an `OtelSink`, runs one task through `run_agent`,
and prints the result.

`--chat` starts an interactive, streaming REPL instead (see `run_chat`).
`--model <key>` selects a backend from the `[models]` registry in
agent.toml (default: `default_model`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from opentelemetry.trace import Tracer

from agent.composition import build_model, build_permissions, build_skills
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.interfaces import Model, PermissionPolicy, SkillRegistry, ToolRegistry
from agent.core.loop import DEFAULT_SYSTEM_PROMPT, compose_system_prompt, run_steps
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.registry import MCPToolRegistry
from agent.observability.otel import build_tracer_provider
from agent.observability.sink import FanOutSink, OtelSink, StreamingConsoleSink


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
    args = parser.parse_args(argv)

    if not args.chat and args.prompt is None:
        parser.error("prompt is required unless --chat is given")

    settings = AgentSettings()  # type: ignore[call-arg]  # models come from agent.toml/env

    try:
        model_config = settings.resolve_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    model = build_model(model_config)
    permissions = build_permissions(settings)
    skills = build_skills(settings)

    tracer_provider = build_tracer_provider(settings.otel)
    tracer = tracer_provider.get_tracer(settings.otel.service_name)

    async with MCPToolRegistry(settings.mcp_servers) as tools:
        if args.chat:
            await run_chat(
                model=model,
                tools=tools,
                skills=skills,
                permissions=permissions,
                tracer=tracer,
                max_steps=settings.max_steps,
            )
        else:
            assert args.prompt is not None
            task = Task(
                id=args.task_id,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                messages=[Message(role="user", content=[TextBlock(text=args.prompt)])],
            )
            result = await run_agent(
                task,
                model=model,
                tools=tools,
                skills=skills,
                permissions=permissions,
                sink=OtelSink(tracer),
                max_steps=settings.max_steps,
            )

            print(f"stop_reason: {result.stop_reason}")
            print(f"usage: {result.usage.model_dump()}")
            print("---")
            print(result.final_text())

    tracer_provider.shutdown()

    return 0


async def run_chat(
    *,
    model: Model,
    tools: ToolRegistry,
    skills: SkillRegistry,
    permissions: PermissionPolicy,
    tracer: Tracer,
    max_steps: int,
) -> None:
    """Interactive REPL: each line is a user turn, streamed against a
    growing conversation. Model output prints token-by-token via
    `StreamingConsoleSink` so the terminal doesn't sit idle while the model
    generates."""
    sink = FanOutSink([StreamingConsoleSink(), OtelSink(tracer)])
    system = compose_system_prompt(DEFAULT_SYSTEM_PROMPT, skills)
    messages: list[Message] = []

    print(f"Chatting with '{model.name}'. Type 'exit' or Ctrl-D to quit.")
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

        messages.append(Message(role="user", content=[TextBlock(text=line)]))

        print("agent> ", end="", flush=True)
        _, stop_reason, _ = await run_steps(
            str(uuid.uuid4()),
            messages,
            model=model,
            tools=tools,
            skills=skills,
            permissions=permissions,
            sink=sink,
            system=system,
            max_steps=max_steps,
        )
        print()
        if stop_reason not in {"end_turn", "stop"}:
            print(f"[stop_reason: {stop_reason}]")


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
