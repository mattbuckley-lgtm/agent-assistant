"""Normalized streaming types shared by all model adapters.

The agent loop (`agent/core/loop.py`) consumes only `StreamEvent`s from
`Model.generate(...)`. Every adapter (Anthropic, OpenAI-compatible, replay,
prompted-tools, Inspect-driven) must translate its provider's stream into
these events.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from agent.core.messages import ToolUseBlock


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    estimated: bool = False
    """True when token counts are estimated locally rather than reported
    by the provider (e.g. local llama.cpp/vLLM endpoints)."""


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallDelta(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str
    name: str | None = None
    args_delta: str


class ToolCallComplete(BaseModel):
    type: Literal["tool_call_complete"] = "tool_call_complete"
    block: ToolUseBlock


class StreamUsage(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage


class StreamDone(BaseModel):
    type: Literal["done"] = "done"
    stop_reason: str


StreamEvent = Annotated[
    TextDelta | ToolCallDelta | ToolCallComplete | StreamUsage | StreamDone,
    Field(discriminator="type"),
]
