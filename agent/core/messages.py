"""Provider-agnostic message and tool types.

These are the only message/content shapes the agent loop and model adapters
exchange. Provider-specific request/response shapes are translated to and
from these types entirely inside each adapter (see `agent/models/`).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, object]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, object]]
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentBlock]


class ToolSpec(BaseModel):
    """Provider-agnostic tool description, translated to each provider's
    function/tool-calling format by the relevant model adapter."""

    name: str
    description: str
    input_schema: dict[str, object]
