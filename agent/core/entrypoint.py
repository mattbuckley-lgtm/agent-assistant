"""The injected entrypoint. This is what makes the runtime evaluable: prod
wiring, replay/regression, and Inspect-driven evals all call this same
function with different Protocol implementations."""

from __future__ import annotations

import uuid

from agent.core.events import Provenance, RunFinished, RunStarted, UserTurnReceived
from agent.core.interfaces import (
    HumanApproval,
    MemoryProvider,
    Model,
    PermissionPolicy,
    SkillRegistry,
    ToolRegistry,
    TranscriptSink,
)
from agent.core.loop import compose_system_prompt, run_steps
from agent.core.memory import MemoryRecord
from agent.core.messages import TextBlock
from agent.core.state import RunResult, Task
from agent.observability.sink import FanOutSink, InMemorySink

_FALLBACK_SYSTEM_PROMPT = "You are a helpful agent."


async def run_agent(
    task: Task,
    *,
    model: Model,
    tools: ToolRegistry,
    skills: SkillRegistry,
    permissions: PermissionPolicy,
    sink: TranscriptSink,
    approval: HumanApproval | None = None,
    max_steps: int = 20,
    memory_provider: MemoryProvider | None = None,
) -> RunResult:
    run_id = str(uuid.uuid4())

    # Fan out to the caller's sink while also recording the full transcript
    # for RunResult, regardless of what sink(s) the caller wired up.
    recorder = InMemorySink()
    fanout = FanOutSink([recorder, sink])

    scope = task.task_id or run_id
    memories: list[MemoryRecord] = []
    if memory_provider is not None:
        try:
            memories = await memory_provider.recall(task, scope=scope)
        except Exception:  # noqa: BLE001 — backend failure: proceed without memories
            pass

    system = compose_system_prompt(task.system_prompt or _FALLBACK_SYSTEM_PROMPT, skills, memories)
    messages = list(task.messages)

    await fanout.emit(RunStarted(run_id=run_id, task_name=task.id, task_id=task.task_id))

    user_text = ""
    for msg in reversed(task.messages):
        if msg.role == "user":
            for block in msg.content:
                if isinstance(block, TextBlock):
                    user_text = block.text
                    break
            break
    await fanout.emit(
        UserTurnReceived(
            run_id=run_id,
            content=user_text,
            provenance=Provenance.USER_STATED,
        )
    )

    final_message, stop_reason, usage = await run_steps(
        run_id,
        messages,
        model=model,
        tools=tools,
        skills=skills,
        permissions=permissions,
        sink=fanout,
        system=system,
        approval=approval,
        max_steps=max_steps,
    )

    await fanout.emit(RunFinished(run_id=run_id, stop_reason=stop_reason, usage=usage))

    return RunResult(
        run_id=run_id,
        task_id=task.id,
        final_message=final_message,
        stop_reason=stop_reason,
        usage=usage,
        transcript=recorder.events,
        messages=messages,
    )
