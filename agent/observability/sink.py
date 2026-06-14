"""TranscriptSink implementations: in-memory, fan-out, and OTel-emitting.

One trace per run; spans nest run -> step -> (model call | tool call) via
explicit OTel context propagation (`set_span_in_context`). Permission
decisions and errors are recorded as span events on the enclosing step span
since they don't have their own start/finish pair in the transcript.
"""

from __future__ import annotations

import json

from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context

from agent.core.events import (
    Error,
    ModelCallFinished,
    ModelCallStarted,
    ModelTextDelta,
    PermissionDecided,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    ToolCallFinished,
    ToolCallRequested,
    ToolCallStarted,
    TranscriptEvent,
)
from agent.core.interfaces import TranscriptSink
from agent.models._convert import tool_result_to_text
from agent.models.base import Usage
from agent.observability.semconv import (
    AGENT_MCP_SERVER,
    AGENT_PERMISSION_DECISION,
    AGENT_PERMISSION_REASON,
    AGENT_RUN_ID,
    AGENT_STEP_INDEX,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASON,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_CACHE_READ_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_TOKENS,
    GEN_AI_USAGE_COST_USD,
    GEN_AI_USAGE_ESTIMATED,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)


class InMemorySink:
    """Collects every event in order. Used by evals and tests."""

    def __init__(self) -> None:
        self.events: list[TranscriptEvent] = []

    async def emit(self, event: TranscriptEvent) -> None:
        self.events.append(event)


class FanOutSink:
    """Forwards each event to every wrapped sink, in order."""

    def __init__(self, sinks: list[TranscriptSink]) -> None:
        self._sinks = sinks

    async def emit(self, event: TranscriptEvent) -> None:
        for s in self._sinks:
            await s.emit(event)


_TOOL_RESULT_PREVIEW_LEN = 200


class StreamingConsoleSink:
    """Prints model output token-by-token as it streams in, plus brief
    tool-call/result indicators. For interactive use (`make chat`)."""

    async def emit(self, event: TranscriptEvent) -> None:
        if isinstance(event, ModelTextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallRequested):
            print(f"\n[tool call: {event.tool}({json.dumps(event.args)})]", flush=True)
        elif isinstance(event, ToolCallFinished):
            text = tool_result_to_text(event.result.content)
            if len(text) > _TOOL_RESULT_PREVIEW_LEN:
                text = text[:_TOOL_RESULT_PREVIEW_LEN] + "..."
            status = "error" if event.is_error else "ok"
            print(f"[tool result ({status}): {text}]", flush=True)
        else:
            pass


def _tag(event: TranscriptEvent, span: Span) -> None:
    """Back-fill the event's `trace_id`/`span_id` for cross-referencing the
    transcript with the OTel trace. Safe because `FanOutSink` passes every
    sink the same event object (mutation is visible to all sinks)."""
    ctx = span.get_span_context()
    event.trace_id = format(ctx.trace_id, "032x")
    event.span_id = format(ctx.span_id, "016x")


class OtelSink:
    """Projects the transcript onto one OTel trace per run: spans nest
    run -> step -> (model call | tool call)."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._run_spans: dict[str, Span] = {}
        self._step_spans: dict[tuple[str, int], Span] = {}
        self._model_spans: dict[tuple[str, int], Span] = {}
        self._tool_spans: dict[tuple[str, str], Span] = {}

    async def emit(self, event: TranscriptEvent) -> None:
        if isinstance(event, RunStarted):
            self._start_run(event)
        elif isinstance(event, RunFinished):
            self._finish_run(event)
        elif isinstance(event, StepStarted):
            self._start_step(event)
        elif isinstance(event, StepFinished):
            self._finish_step(event)
        elif isinstance(event, ModelCallStarted):
            self._start_model_call(event)
        elif isinstance(event, ModelCallFinished):
            self._finish_model_call(event)
        elif isinstance(event, ToolCallStarted):
            self._start_tool_call(event)
        elif isinstance(event, ToolCallFinished):
            self._finish_tool_call(event)
        elif isinstance(event, PermissionDecided):
            self._record_permission(event)
        elif isinstance(event, Error):
            self._record_error(event)
        else:
            # ToolCallRequested has no span of its own (covered by the
            # tool-call span); ModelTextDelta is too high-frequency for spans
            # and is only consumed by streaming sinks.
            pass

    def _enclosing_span(self, run_id: str, step_index: int | None) -> Span | None:
        if step_index is not None:
            span = self._step_spans.get((run_id, step_index))
            if span is not None:
                return span
        return self._run_spans.get(run_id)

    def _start_run(self, event: RunStarted) -> None:
        span = self._tracer.start_span(
            f"agent_run {event.task_name}",
            attributes={AGENT_RUN_ID: event.run_id},
        )
        self._run_spans[event.run_id] = span
        _tag(event, span)

    def _finish_run(self, event: RunFinished) -> None:
        span = self._run_spans.pop(event.run_id, None)
        if span is None:
            return
        _tag(event, span)
        _set_usage_attributes(span, event.usage)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASON, [event.stop_reason])
        span.end()

    def _start_step(self, event: StepStarted) -> None:
        assert event.step_index is not None
        parent = self._run_spans.get(event.run_id)
        context = set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            f"step {event.step_index}",
            context=context,
            attributes={AGENT_RUN_ID: event.run_id, AGENT_STEP_INDEX: event.step_index},
        )
        self._step_spans[(event.run_id, event.step_index)] = span
        _tag(event, span)

    def _finish_step(self, event: StepFinished) -> None:
        assert event.step_index is not None
        span = self._step_spans.pop((event.run_id, event.step_index), None)
        if span is None:
            return
        _tag(event, span)
        span.end()

    def _start_model_call(self, event: ModelCallStarted) -> None:
        assert event.step_index is not None
        parent = self._enclosing_span(event.run_id, event.step_index)
        context = set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            f"chat {event.model}",
            context=context,
            attributes={
                GEN_AI_SYSTEM: "agent-runtime",
                GEN_AI_REQUEST_MODEL: event.model,
                AGENT_RUN_ID: event.run_id,
                AGENT_STEP_INDEX: event.step_index,
            },
        )
        self._model_spans[(event.run_id, event.step_index)] = span
        _tag(event, span)

    def _finish_model_call(self, event: ModelCallFinished) -> None:
        assert event.step_index is not None
        span = self._model_spans.pop((event.run_id, event.step_index), None)
        if span is None:
            return
        _tag(event, span)
        _set_usage_attributes(span, event.usage)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASON, [event.stop_reason])
        span.end()

    def _start_tool_call(self, event: ToolCallStarted) -> None:
        assert event.step_index is not None
        parent = self._enclosing_span(event.run_id, event.step_index)
        context = set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            f"execute_tool {event.tool}",
            context=context,
            attributes={
                GEN_AI_TOOL_NAME: event.tool,
                GEN_AI_TOOL_CALL_ID: event.tool_use_id,
                AGENT_MCP_SERVER: event.server,
                AGENT_RUN_ID: event.run_id,
                AGENT_STEP_INDEX: event.step_index,
            },
        )
        self._tool_spans[(event.run_id, event.tool_use_id)] = span
        _tag(event, span)

    def _finish_tool_call(self, event: ToolCallFinished) -> None:
        span = self._tool_spans.pop((event.run_id, event.tool_use_id), None)
        if span is None:
            return
        _tag(event, span)
        if event.is_error:
            span.set_status(Status(StatusCode.ERROR))
        span.end()

    def _record_permission(self, event: PermissionDecided) -> None:
        span = self._enclosing_span(event.run_id, event.step_index)
        if span is None:
            return
        _tag(event, span)
        span.add_event(
            "permission_decided",
            attributes={
                AGENT_PERMISSION_DECISION: event.decision.value,
                AGENT_PERMISSION_REASON: event.reason,
                GEN_AI_TOOL_NAME: event.tool,
                AGENT_MCP_SERVER: event.server,
            },
        )

    def _record_error(self, event: Error) -> None:
        span = self._enclosing_span(event.run_id, event.step_index)
        if span is None:
            return
        _tag(event, span)
        span.add_event("error", attributes={"where": event.where, "message": event.message})


def _set_usage_attributes(span: Span, usage: Usage) -> None:
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
    span.set_attribute(GEN_AI_USAGE_CACHE_READ_TOKENS, usage.cache_read_tokens)
    span.set_attribute(GEN_AI_USAGE_CACHE_WRITE_TOKENS, usage.cache_write_tokens)
    span.set_attribute(GEN_AI_USAGE_ESTIMATED, usage.estimated)
    if usage.cost_usd is not None:
        span.set_attribute(GEN_AI_USAGE_COST_USD, usage.cost_usd)
