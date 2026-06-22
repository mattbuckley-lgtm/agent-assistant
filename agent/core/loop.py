"""The agent loop: composes the system prompt and runs the step loop.

See `agent/core/entrypoint.py` for the injected `run_agent` callable that
wraps this with run-level bookkeeping (run id, RunStarted/RunFinished).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter

from agent.core.events import (
    Decision,
    Error,
    ModelCallFinished,
    ModelCallStarted,
    ModelTextDelta,
    PermissionDecided,
    SkillInvoked,
    StepFinished,
    StepStarted,
    ToolCallFinished,
    ToolCallRequested,
    ToolCallStarted,
)
from agent.core.interfaces import (
    HumanApproval,
    Model,
    PermissionPolicy,
    SkillRegistry,
    ToolRegistry,
    TranscriptSink,
)
from agent.core.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent.models.base import (
    StreamDone,
    StreamUsage,
    TextDelta,
    ToolCallComplete,
    Usage,
)

# Guard rail: hard stop if the same (tool, args) call repeats this many times.
MAX_REPEATED_TOOL_CALLS = 3

# Guard rail: at most this many tool calls are executed per step; any
# remaining tool-use blocks from the model are returned as errors.
MAX_TOOL_CALLS_PER_STEP = 8


def compose_system_prompt(base: str, skills: SkillRegistry) -> str:
    """Base instructions + skill index (names/descriptions only)."""
    skill_list = skills.list_skills()
    if not skill_list:
        return base
    index_lines = [f"- {s.name}: {s.description} (use when: {s.when_to_use})" for s in skill_list]
    header = "\n\nAvailable skills (call the matching tool to load full instructions):\n"
    return base + header + "\n".join(index_lines)


def _skill_tool_specs(skills: SkillRegistry) -> list[ToolSpec]:
    """Synthesize a ToolSpec per skill so the model can 'call' a skill by
    name to load its full body. Handled in-loop -- never reaches MCP/permissions."""
    return [
        ToolSpec(
            name=skill.name,
            description=f"Load the full '{skill.name}' skill instructions: {skill.description}",
            input_schema={"type": "object", "properties": {}},
        )
        for skill in skills.list_skills()
    ]


def _digest(messages: list[Message]) -> str:
    payload = json.dumps([m.model_dump(mode="json") for m in messages], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _call_key(tool: str, args: dict[str, object]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True)}"


def _merge_usage(a: Usage, b: Usage) -> Usage:
    cost = None
    if a.cost_usd is not None or b.cost_usd is not None:
        cost = (a.cost_usd or 0.0) + (b.cost_usd or 0.0)
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        cost_usd=cost,
        estimated=a.estimated or b.estimated,
    )


async def run_steps(
    run_id: str,
    messages: list[Message],
    *,
    model: Model,
    tools: ToolRegistry,
    skills: SkillRegistry,
    permissions: PermissionPolicy,
    sink: TranscriptSink,
    system: str,
    approval: HumanApproval | None = None,
    max_steps: int = 20,
) -> tuple[Message | None, str, Usage]:
    """Run the bounded step loop in-place against `messages`.

    Returns (final_message, stop_reason, aggregate_usage). `final_message` is
    the last assistant message (the answer) when the model stops requesting
    tools, or the last assistant message produced before a guard rail tripped.
    """
    tool_specs = tools.list_tool_specs() + _skill_tool_specs(skills)
    aggregate_usage = Usage()
    final_message: Message | None = None
    stop_reason = "max_steps"
    call_counts: Counter[str] = Counter()

    for step_index in range(max_steps):
        await sink.emit(StepStarted(run_id=run_id, step_index=step_index))

        text_parts: list[str] = []
        tool_use_blocks: list[ToolUseBlock] = []
        call_usage = Usage()
        model_stop_reason = "end_turn"

        await sink.emit(
            ModelCallStarted(
                run_id=run_id,
                step_index=step_index,
                model=model.name,
                messages_digest=_digest(messages),
            )
        )
        start = time.monotonic()
        async for event in model.generate(messages, tool_specs, system=system):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                await sink.emit(
                    ModelTextDelta(run_id=run_id, step_index=step_index, text=event.text)
                )
            elif isinstance(event, ToolCallComplete):
                tool_use_blocks.append(event.block)
            elif isinstance(event, StreamUsage):
                call_usage = _merge_usage(call_usage, event.usage)
            elif isinstance(event, StreamDone):
                model_stop_reason = event.stop_reason
            else:
                pass  # ToolCallDelta is streaming-only; the loop waits for ToolCallComplete
        latency_ms = (time.monotonic() - start) * 1000

        await sink.emit(
            ModelCallFinished(
                run_id=run_id,
                step_index=step_index,
                usage=call_usage,
                stop_reason=model_stop_reason,
                latency_ms=latency_ms,
            )
        )
        aggregate_usage = _merge_usage(aggregate_usage, call_usage)

        content: list[ContentBlock] = []
        if text_parts:
            content.append(TextBlock(text="".join(text_parts)))
        content.extend(tool_use_blocks)
        assistant_message = Message(role="assistant", content=content)
        messages.append(assistant_message)
        final_message = assistant_message

        if not tool_use_blocks:
            stop_reason = model_stop_reason
            await sink.emit(StepFinished(run_id=run_id, step_index=step_index))
            break

        tool_results, loop_detected = await _execute_tool_calls(
            run_id,
            step_index,
            tool_use_blocks,
            tools=tools,
            skills=skills,
            permissions=permissions,
            sink=sink,
            approval=approval,
            call_counts=call_counts,
        )
        messages.append(Message(role="tool", content=tool_results))
        await sink.emit(StepFinished(run_id=run_id, step_index=step_index))

        if loop_detected:
            stop_reason = "loop_detected"
            break

    return final_message, stop_reason, aggregate_usage


async def _execute_tool_calls(
    run_id: str,
    step_index: int,
    tool_use_blocks: list[ToolUseBlock],
    *,
    tools: ToolRegistry,
    skills: SkillRegistry,
    permissions: PermissionPolicy,
    sink: TranscriptSink,
    approval: HumanApproval | None,
    call_counts: Counter[str],
) -> tuple[list[ContentBlock], bool]:
    results: list[ContentBlock] = []
    loop_detected = False

    for block in tool_use_blocks[:MAX_TOOL_CALLS_PER_STEP]:
        skill = skills.get_skill(block.name)
        if skill is not None:
            await sink.emit(
                SkillInvoked(
                    run_id=run_id,
                    step_index=step_index,
                    tool_use_id=block.id,
                    skill=skill.name,
                )
            )
            results.append(ToolResultBlock(tool_use_id=block.id, content=skill.load_body()))
            continue

        server = tools.server_for_tool(block.name)

        key = _call_key(block.name, block.input)
        call_counts[key] += 1
        if call_counts[key] > MAX_REPEATED_TOOL_CALLS:
            count = call_counts[key]
            await sink.emit(
                Error(
                    run_id=run_id,
                    step_index=step_index,
                    where="loop_detection",
                    message=f"tool '{block.name}' repeated with identical args {count} times",
                )
            )
            results.append(
                ToolResultBlock(
                    tool_use_id=block.id,
                    content="loop detected: repeated identical tool call, aborting run",
                    is_error=True,
                )
            )
            loop_detected = True
            continue

        await sink.emit(
            ToolCallRequested(
                run_id=run_id,
                step_index=step_index,
                tool_use_id=block.id,
                server=server,
                tool=block.name,
                args=block.input,
            )
        )

        decision = permissions.evaluate(server, block.name, block.input)
        allowed = decision is Decision.ALLOW
        reason = f"policy decision: {decision.value}"
        if decision is Decision.PROMPT:
            if approval is None:
                allowed = False
                reason = "prompted but no approval hook configured"
            else:
                approved = await approval(server, block.name, block.input)
                allowed = approved
                reason = "human approved" if approved else "human denied"

        await sink.emit(
            PermissionDecided(
                run_id=run_id,
                step_index=step_index,
                tool_use_id=block.id,
                server=server,
                tool=block.name,
                decision=decision,
                allowed=allowed,
                reason=reason,
            )
        )

        if not allowed:
            results.append(
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=f"Permission denied for {server}.{block.name}: {reason}",
                    is_error=True,
                )
            )
            continue

        await sink.emit(
            ToolCallStarted(
                run_id=run_id,
                step_index=step_index,
                tool_use_id=block.id,
                server=server,
                tool=block.name,
            )
        )
        tool_start = time.monotonic()
        try:
            result = await tools.call_tool(server, block.name, block.input)
            result.tool_use_id = block.id
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            result = ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
        tool_latency_ms = (time.monotonic() - tool_start) * 1000

        await sink.emit(
            ToolCallFinished(
                run_id=run_id,
                step_index=step_index,
                tool_use_id=block.id,
                result=result,
                is_error=result.is_error,
                latency_ms=tool_latency_ms,
            )
        )
        results.append(result)

    return results, loop_detected
