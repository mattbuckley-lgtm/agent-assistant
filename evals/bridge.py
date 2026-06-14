"""Bridges Inspect AI tasks to `run_agent`.

The resulting `Solver` ignores Inspect's own model/`generate` entirely --
it runs the sample's input through our agent loop (with whichever `Model`/
`ToolRegistry`/etc. the eval task wires up) and reports the agent's final
answer as `state.output` so Inspect's scorers can grade it. This keeps a
single code path between production and evals: both call `run_agent`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agent.core.entrypoint import run_agent
from agent.core.interfaces import Model, PermissionPolicy, SkillRegistry, ToolRegistry
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.observability.sink import InMemorySink

AgentDeps = tuple[Model, ToolRegistry, SkillRegistry, PermissionPolicy]
DepsFactory = Callable[[], AbstractAsyncContextManager[AgentDeps]]


@solver
def run_agent_solver(build: DepsFactory, *, system_prompt: str = "") -> Solver:
    """Run the sample's input through `run_agent`; report the final answer
    as `state.output`. `build` is called once per sample and must yield a
    fresh `(model, tools, skills, permissions)` tuple."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with build() as (model, tools, skills, permissions):
            agent_task = Task(
                id=str(state.sample_id),
                system_prompt=system_prompt,
                messages=[Message(role="user", content=[TextBlock(text=state.input_text)])],
            )
            result = await run_agent(
                agent_task,
                model=model,
                tools=tools,
                skills=skills,
                permissions=permissions,
                sink=InMemorySink(),
            )

        final_text = result.final_text()
        state.output = ModelOutput.from_content(model=model.name, content=final_text)
        state.messages.append(ChatMessageAssistant(content=final_text))
        state.store.set("transcript", [e.model_dump(mode="json") for e in result.transcript])
        state.store.set("stop_reason", result.stop_reason)
        return state

    return solve
