"""Run-level state: the task definition and the final result of a run."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.core.events import TranscriptEvent
from agent.core.messages import Message
from agent.models.base import Usage


class Task(BaseModel):
    """A unit of work for the agent loop: an id and an initial conversation."""

    id: str
    system_prompt: str | None = None
    messages: list[Message]


class RunResult(BaseModel):
    """The outcome of a single `run_agent(...)` call."""

    run_id: str
    task_id: str
    final_message: Message | None = None
    stop_reason: str
    usage: Usage
    transcript: list[TranscriptEvent] = Field(default_factory=list[TranscriptEvent])
