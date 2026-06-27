"""Distiller: converts a run transcript into a compact EpisodicRecord.

The default HeuristicDistiller is deterministic (no model call): it extracts
the task goal from UserTurnReceived, the agent's conclusion from the final
block of ModelTextDelta, and a provenance-tagged list of tool calls from
ToolCallFinished events. Overall provenance is the high-water-mark-of-untrust
across all tool calls in the transcript.

LlmDistiller is config-selected (optional) and makes one local model call for
a richer natural-language summary; it is out of scope for Phase 3 and raises
NotImplementedError.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from agent.core.events import (
    ModelTextDelta,
    Provenance,
    ToolCallFinished,
    TranscriptEvent,
    UserTurnReceived,
)
from agent.memory.records import EpisodicRecord

_PROVENANCE_LABELS: dict[Provenance, str] = {
    Provenance.AGENT_REASONING: "agent",
    Provenance.USER_STATED: "user",
    Provenance.TOOL_OUTPUT: "tool_output",
}

_MAX_TOOL_RESULT_CHARS = 200


def _effective_provenance(transcript: list[TranscriptEvent]) -> Provenance:
    """High-water-mark-of-untrust: TOOL_OUTPUT beats USER_STATED beats AGENT_REASONING."""
    provenances = [e.provenance for e in transcript if isinstance(e, ToolCallFinished)]
    if Provenance.TOOL_OUTPUT in provenances:
        return Provenance.TOOL_OUTPUT
    if Provenance.USER_STATED in provenances:
        return Provenance.USER_STATED
    return Provenance.AGENT_REASONING


def _extract_goal(transcript: list[TranscriptEvent]) -> str:
    for e in transcript:
        if isinstance(e, UserTurnReceived):
            return e.content
    return ""


def _extract_conclusion(transcript: list[TranscriptEvent]) -> str:
    """Text from ModelTextDelta events after the last ToolCallFinished."""
    last_tool_idx = -1
    for i, e in enumerate(transcript):
        if isinstance(e, ToolCallFinished):
            last_tool_idx = i
    return "".join(
        e.text for e in transcript[last_tool_idx + 1 :] if isinstance(e, ModelTextDelta)
    ).strip()


def _extract_tool_lines(transcript: list[TranscriptEvent]) -> list[str]:
    lines: list[str] = []
    for e in transcript:
        if not isinstance(e, ToolCallFinished):
            continue
        label = _PROVENANCE_LABELS.get(e.provenance, e.provenance.value)
        raw = e.result.content
        if isinstance(raw, list):
            text = next(
                (str(item.get("text", "")) for item in raw if item.get("type") == "text"),
                str(raw),
            )
        else:
            text = str(raw)
        if len(text) > _MAX_TOOL_RESULT_CHARS:
            text = text[:_MAX_TOOL_RESULT_CHARS] + "…"
        lines.append(f"- [{label}] `{e.source}` → {text!r}")
    return lines


@runtime_checkable
class Distiller(Protocol):
    async def distil(
        self,
        transcript: list[TranscriptEvent],
        *,
        run_id: str,
        task_id: str,
    ) -> EpisodicRecord: ...


class HeuristicDistiller:
    """Deterministic distiller: no model call, always available as the default."""

    async def distil(
        self,
        transcript: list[TranscriptEvent],
        *,
        run_id: str,
        task_id: str,
    ) -> EpisodicRecord:
        goal = _extract_goal(transcript)
        conclusion = _extract_conclusion(transcript)
        tool_lines = _extract_tool_lines(transcript)

        parts: list[str] = [f"## Goal\n\n{goal}"]
        if conclusion:
            parts.append(f"## Conclusion\n\n{conclusion}")
        if tool_lines:
            parts.append("## Tool calls\n\n" + "\n".join(tool_lines))

        return EpisodicRecord(
            id=str(uuid.uuid4()),
            task_id=task_id,
            run_id=run_id,
            timestamp=datetime.now(UTC),
            summary="\n\n".join(parts),
            provenance=_effective_provenance(transcript),
        )


class LlmDistiller:
    """Optional model-backed distiller. Raises NotImplementedError until Phase 3+."""

    async def distil(
        self,
        transcript: list[TranscriptEvent],
        *,
        run_id: str,
        task_id: str,
    ) -> EpisodicRecord:
        raise NotImplementedError("LlmDistiller is not implemented in Phase 3")
