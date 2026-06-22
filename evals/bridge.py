"""Bridges Inspect AI samples to `run_agent` for case-based evals (see
`evals.spec.EvalCase` and `evals.suite`).

Each sample carries an `EvalCase` as `Sample.metadata`, and may override the
tool registry (`MockToolRegistry`) and permission policy. Everything else --
skills, MCP servers, the default permission policy -- comes from the same
`agent.toml` used in production (via `agent.composition`), so evals exercise
the same wiring as production. This keeps a single code path between
production and evals: both call `run_agent`.

The assistant is either a scripted `ReplayModel` (the default `"replay"`,
deterministically replaying the case's cassette) or a real model from
`agent.toml`'s `[models]` registry, selected via `run_eval_case(model=...)`
the same way `python -m agent --model <key>` does.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agent.composition import build_model, build_permissions, build_permissions_from_rules, build_skills
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.interfaces import Model, PermissionPolicy, ToolRegistry
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from evals.mock_tools import MockToolRegistry
from evals.spec import EvalCase

CASSETTES_DIR = Path(__file__).parent.parent / "tests" / "cassettes"


@solver
def run_eval_case(model: str = "replay") -> Solver:
    """Run each sample's `EvalCase` through `run_agent`, reporting the final
    answer as `state.output` plus the full transcript and stop reason (via
    `state.store`) so `evals.scorers` can grade both the response and how the
    agent got there.

    For multi-turn cases (`EvalCase.turns`), runs one `run_agent` call per
    turn with accumulated message history, storing per-turn transcripts,
    stop reasons, and final texts under `turn_*` keys in `state.store`.

    `model` is `"replay"` (default) for deterministic cassette playback, or
    a `agent.toml` `[models]` registry key to run against a real model."""

    settings = AgentSettings()  # type: ignore[call-arg]  # agent.toml + env
    skills = build_skills(settings)
    default_permissions = build_permissions(settings)
    real_model: Model | None = (
        None if model == "replay" else build_model(settings.resolve_model(model))
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = state.metadata_as(EvalCase)
        sample_id = str(state.sample_id)
        permissions: PermissionPolicy = (
            build_permissions_from_rules(case.permissions)
            if case.permissions is not None
            else default_permissions
        )

        tools_impl: MockToolRegistry | MCPToolRegistry = (
            MockToolRegistry(case.mock_tools)
            if case.mock_tools
            else MCPToolRegistry(settings.mcp_servers)
        )
        async with tools_impl as tools:

            if case.turns:
                accumulated: list[Message] = []
                turn_transcripts: list[list[dict[str, object]]] = []
                turn_stop_reasons: list[str] = []
                turn_final_texts: list[str] = []

                for turn in case.turns:
                    active_model = real_model or ReplayModel(
                        CASSETTES_DIR / turn.cassette, name="replay"
                    )
                    accumulated.append(
                        Message(role="user", content=[TextBlock(text=turn.user_message)])
                    )
                    agent_task = Task(
                        id=sample_id,
                        system_prompt=case.system_prompt or settings.system_prompt,
                        messages=list(accumulated),
                    )
                    turn_result = await run_agent(
                        agent_task,
                        model=active_model,
                        tools=tools,
                        skills=skills,
                        permissions=permissions,
                        sink=InMemorySink(),
                    )
                    turn_transcripts.append(
                        [e.model_dump(mode="json") for e in turn_result.transcript]
                    )
                    turn_stop_reasons.append(turn_result.stop_reason)
                    turn_final_texts.append(turn_result.final_text())
                    accumulated = list(turn_result.messages)

                final_text = turn_final_texts[-1] if turn_final_texts else ""
                active_model_name = real_model.name if real_model else "replay"
                state.output = ModelOutput.from_content(model=active_model_name, content=final_text)
                # Populate the full conversation into state.messages so the
                # eval-view shows every clarifying exchange, not just the final
                # answer. The first user message is already in state.messages
                # from the sample input (Inspect AI adds it on construction).
                for i, (turn, text) in enumerate(zip(case.turns, turn_final_texts, strict=False)):
                    if i > 0:
                        state.messages.append(ChatMessageUser(content=turn.user_message))
                    state.messages.append(ChatMessageAssistant(content=text))
                state.store.set("turn_transcripts", turn_transcripts)
                state.store.set("turn_stop_reasons", turn_stop_reasons)
                state.store.set("turn_final_texts", turn_final_texts)
                # Combined transcript lets skills_used and other aggregate scorers
                # work across the whole conversation without changes.
                state.store.set("transcript", [e for ts in turn_transcripts for e in ts])
                state.store.set("stop_reason", turn_stop_reasons[-1] if turn_stop_reasons else "")

            else:
                active_model = real_model or ReplayModel(
                    CASSETTES_DIR / case.cassette, name="replay"
                )
                agent_task = Task(
                    id=sample_id,
                    system_prompt=case.system_prompt or settings.system_prompt,
                    messages=[Message(role="user", content=[TextBlock(text=state.input_text)])],
                )
                result = await run_agent(
                    agent_task,
                    model=active_model,
                    tools=tools,
                    skills=skills,
                    permissions=permissions,
                    sink=InMemorySink(),
                )
                final_text = result.final_text()
                state.output = ModelOutput.from_content(model=active_model.name, content=final_text)
                state.messages.append(ChatMessageAssistant(content=final_text))
                state.store.set(
                    "transcript", [e.model_dump(mode="json") for e in result.transcript]
                )
                state.store.set("stop_reason", result.stop_reason)

        return state

    return solve
