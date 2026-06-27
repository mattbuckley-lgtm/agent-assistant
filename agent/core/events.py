"""The transcript: a single ordered, typed event stream per run.

This is the source of truth. OTel spans, Langfuse traces, and eval scoring
are all projections of this stream (see `agent/observability/sink.py`).

Events are emitted at exactly three seams: step boundaries (the loop),
model calls (inside model adapters), and tool calls (inside the permission
interceptor).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from agent.core.messages import ToolResultBlock
from agent.models.base import Usage


class Provenance(StrEnum):
    """Trust axis of a memory record.

    Rendered as labels in the system-prompt memory section and used as a
    read gate: AGENT_REASONING is trusted internal inference, USER_STATED is
    self-reported (lower trust), TOOL_OUTPUT is unverified external data.
    """

    AGENT_REASONING = "agent_reasoning"
    USER_STATED = "user_stated"
    TOOL_OUTPUT = "tool_output"


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    PROMPT = "prompt"


def _now() -> datetime:
    return datetime.now(UTC)


class EventBase(BaseModel):
    run_id: str
    step_index: int | None = None
    ts: datetime = Field(default_factory=_now)
    trace_id: str | None = None
    span_id: str | None = None


class RunStarted(EventBase):
    type: Literal["run_started"] = "run_started"
    task_name: str
    task_id: str | None = None


class RunFinished(EventBase):
    type: Literal["run_finished"] = "run_finished"
    stop_reason: str
    usage: Usage


class StepStarted(EventBase):
    type: Literal["step_started"] = "step_started"


class StepFinished(EventBase):
    type: Literal["step_finished"] = "step_finished"


class ModelCallStarted(EventBase):
    type: Literal["model_call_started"] = "model_call_started"
    model: str
    messages_digest: str


class ModelCallFinished(EventBase):
    type: Literal["model_call_finished"] = "model_call_finished"
    usage: Usage
    stop_reason: str
    latency_ms: float


class ModelTextDelta(EventBase):
    type: Literal["model_text_delta"] = "model_text_delta"
    text: str
    provenance: Provenance = Provenance.AGENT_REASONING


class ToolCallRequested(EventBase):
    type: Literal["tool_call_requested"] = "tool_call_requested"
    tool_use_id: str
    server: str
    tool: str
    args: dict[str, object]


class PermissionDecided(EventBase):
    type: Literal["permission_decided"] = "permission_decided"
    tool_use_id: str
    server: str
    tool: str
    decision: Decision
    allowed: bool
    reason: str


class SkillInvoked(EventBase):
    type: Literal["skill_invoked"] = "skill_invoked"
    tool_use_id: str
    skill: str


class ToolCallStarted(EventBase):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_use_id: str
    server: str
    tool: str


class UserTurnReceived(EventBase):
    type: Literal["user_turn_received"] = "user_turn_received"
    content: str
    provenance: Provenance = Provenance.USER_STATED


class ToolCallFinished(EventBase):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    tool_use_id: str
    result: ToolResultBlock
    is_error: bool
    latency_ms: float
    provenance: Provenance = Provenance.TOOL_OUTPUT
    source: str = ""


class Error(EventBase):
    type: Literal["error"] = "error"
    where: str
    message: str


TranscriptEvent = Annotated[
    RunStarted
    | RunFinished
    | StepStarted
    | StepFinished
    | ModelCallStarted
    | ModelCallFinished
    | ModelTextDelta
    | UserTurnReceived
    | ToolCallRequested
    | PermissionDecided
    | SkillInvoked
    | ToolCallStarted
    | ToolCallFinished
    | Error,
    Field(discriminator="type"),
]
