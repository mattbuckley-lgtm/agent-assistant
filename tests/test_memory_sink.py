"""Tests for the Phase 3 episodic formation: distiller, sink, and formatting."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from agent.core.events import (
    ModelTextDelta,
    Provenance,
    RunFinished,
    RunStarted,
    ToolCallFinished,
    TranscriptEvent,
    UserTurnReceived,
)
from agent.core.messages import ToolResultBlock
from agent.memory.distiller import (
    HeuristicDistiller,
    _effective_provenance,  # pyright: ignore[reportPrivateUsage]
    _extract_conclusion,  # pyright: ignore[reportPrivateUsage]
    _extract_goal,  # pyright: ignore[reportPrivateUsage]
    _extract_tool_lines,  # pyright: ignore[reportPrivateUsage]
)
from agent.memory.records import EpisodicRecord
from agent.memory.sink import (
    MemorySink,
    _format_episodic_record,  # pyright: ignore[reportPrivateUsage]
)
from agent.models.base import Usage

_RUN_ID = "run-abc"
_TASK_ID = "task-xyz"


def _run_started(**kw: object) -> RunStarted:
    return RunStarted(run_id=_RUN_ID, task_name="test", **kw)  # type: ignore[arg-type]


def _run_finished() -> RunFinished:
    return RunFinished(
        run_id=_RUN_ID,
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _user_turn(text: str) -> UserTurnReceived:
    return UserTurnReceived(run_id=_RUN_ID, content=text)


def _text_delta(text: str) -> ModelTextDelta:
    return ModelTextDelta(run_id=_RUN_ID, text=text)


def _tool_finished(
    tool: str = "my_tool",
    content: str = "ok",
    provenance: Provenance = Provenance.TOOL_OUTPUT,
    source: str = "",
) -> ToolCallFinished:
    return ToolCallFinished(
        run_id=_RUN_ID,
        tool_use_id="id-1",
        result=ToolResultBlock(tool_use_id="id-1", content=content, is_error=False),
        is_error=False,
        latency_ms=0.0,
        provenance=provenance,
        source=source or f"tool:{tool}",
    )


# ---------------------------------------------------------------------------
# _effective_provenance
# ---------------------------------------------------------------------------


def test_effective_provenance_empty_returns_agent_reasoning() -> None:
    events: list[TranscriptEvent] = []
    assert _effective_provenance(events) == Provenance.AGENT_REASONING


def test_effective_provenance_no_tool_calls_returns_agent_reasoning() -> None:
    events: list[TranscriptEvent] = [_user_turn("hi"), _text_delta("hello")]
    assert _effective_provenance(events) == Provenance.AGENT_REASONING


def test_effective_provenance_tool_output_wins() -> None:
    events: list[TranscriptEvent] = [_tool_finished(provenance=Provenance.TOOL_OUTPUT)]
    assert _effective_provenance(events) == Provenance.TOOL_OUTPUT


def test_effective_provenance_high_water_mark() -> None:
    events: list[TranscriptEvent] = [
        _tool_finished(provenance=Provenance.AGENT_REASONING),
        _tool_finished(provenance=Provenance.TOOL_OUTPUT),
    ]
    assert _effective_provenance(events) == Provenance.TOOL_OUTPUT


# ---------------------------------------------------------------------------
# _extract_goal
# ---------------------------------------------------------------------------


def test_extract_goal_finds_user_turn() -> None:
    events: list[TranscriptEvent] = [_run_started(), _user_turn("Find the answer")]
    assert _extract_goal(events) == "Find the answer"


def test_extract_goal_returns_empty_when_no_user_turn() -> None:
    events: list[TranscriptEvent] = [_text_delta("hello")]
    assert _extract_goal(events) == ""


# ---------------------------------------------------------------------------
# _extract_conclusion
# ---------------------------------------------------------------------------


def test_extract_conclusion_text_after_last_tool_call() -> None:
    events: list[TranscriptEvent] = [
        _tool_finished(),
        _text_delta("First answer. "),
        _tool_finished(),
        _text_delta("Final answer."),
    ]
    assert _extract_conclusion(events) == "Final answer."


def test_extract_conclusion_all_text_when_no_tool_calls() -> None:
    events: list[TranscriptEvent] = [_text_delta("Hello "), _text_delta("world")]
    assert _extract_conclusion(events) == "Hello world"


def test_extract_conclusion_empty_when_only_tool_calls() -> None:
    events: list[TranscriptEvent] = [_tool_finished()]
    assert _extract_conclusion(events) == ""


# ---------------------------------------------------------------------------
# _extract_tool_lines
# ---------------------------------------------------------------------------


def test_extract_tool_lines_labels_provenance() -> None:
    events: list[TranscriptEvent] = [_tool_finished(tool="echo", content="hi", source="tool:echo")]
    lines = _extract_tool_lines(events)
    assert len(lines) == 1
    assert "[tool_output]" in lines[0]
    assert "`tool:echo`" in lines[0]


def test_extract_tool_lines_truncates_long_result() -> None:
    events: list[TranscriptEvent] = [_tool_finished(content="x" * 300)]
    lines = _extract_tool_lines(events)
    assert "…" in lines[0]


def test_extract_tool_lines_list_content() -> None:
    event = ToolCallFinished(
        run_id=_RUN_ID,
        tool_use_id="id-1",
        result=ToolResultBlock(
            tool_use_id="id-1",
            content=[{"type": "text", "text": "list result"}],
            is_error=False,
        ),
        is_error=False,
        latency_ms=0.0,
        source="tool:x",
    )
    events: list[TranscriptEvent] = [event]
    lines = _extract_tool_lines(events)
    assert "list result" in lines[0]


# ---------------------------------------------------------------------------
# HeuristicDistiller
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heuristic_distiller_goal_in_summary() -> None:
    events: list[TranscriptEvent] = [_user_turn("What is the capital of France?")]
    distiller = HeuristicDistiller()
    record = await distiller.distil(events, run_id=_RUN_ID, task_id=_TASK_ID)
    assert "What is the capital of France?" in record.summary
    assert record.run_id == _RUN_ID
    assert record.task_id == _TASK_ID


@pytest.mark.asyncio
async def test_heuristic_distiller_includes_conclusion() -> None:
    events: list[TranscriptEvent] = [
        _tool_finished(),
        _text_delta("Paris is the capital."),
    ]
    distiller = HeuristicDistiller()
    record = await distiller.distil(events, run_id=_RUN_ID, task_id=_TASK_ID)
    assert "Paris is the capital." in record.summary


@pytest.mark.asyncio
async def test_heuristic_distiller_includes_tool_calls_section() -> None:
    events: list[TranscriptEvent] = [_tool_finished(tool="search", source="tool:search")]
    distiller = HeuristicDistiller()
    record = await distiller.distil(events, run_id=_RUN_ID, task_id=_TASK_ID)
    assert "## Tool calls" in record.summary


@pytest.mark.asyncio
async def test_heuristic_distiller_provenance_from_tools() -> None:
    events: list[TranscriptEvent] = [_tool_finished(provenance=Provenance.TOOL_OUTPUT)]
    distiller = HeuristicDistiller()
    record = await distiller.distil(events, run_id=_RUN_ID, task_id=_TASK_ID)
    assert record.provenance == Provenance.TOOL_OUTPUT


@pytest.mark.asyncio
async def test_heuristic_distiller_no_tool_calls_no_section() -> None:
    events: list[TranscriptEvent] = [_user_turn("hello"), _text_delta("world")]
    distiller = HeuristicDistiller()
    record = await distiller.distil(events, run_id=_RUN_ID, task_id=_TASK_ID)
    assert "## Tool calls" not in record.summary


# ---------------------------------------------------------------------------
# _format_episodic_record
# ---------------------------------------------------------------------------


def test_format_episodic_record_has_frontmatter() -> None:
    record = EpisodicRecord(
        id="rec-1",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        summary="## Goal\n\nHello",
        provenance=Provenance.TOOL_OUTPUT,
    )
    text = _format_episodic_record(record)
    assert text.startswith("---\n")
    assert "id: rec-1" in text
    assert "task_id: task-xyz" in text
    assert "provenance: tool_output" in text
    assert "- episodic" in text
    assert "## Goal" in text


def test_format_episodic_record_iso_timestamp() -> None:
    ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    record = EpisodicRecord(
        id="rec-2",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        timestamp=ts,
        summary="body",
        provenance=Provenance.AGENT_REASONING,
    )
    text = _format_episodic_record(record)
    assert "2025-06-15" in text


# ---------------------------------------------------------------------------
# MemorySink
# ---------------------------------------------------------------------------


class _CapturingStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []
        self.fail: bool = False

    async def write(self, path: str, content: str) -> None:
        if self.fail:
            raise RuntimeError("write failed")
        self.writes.append((path, content))


@pytest.mark.asyncio
async def test_memory_sink_writes_on_run_finished() -> None:
    store = _CapturingStore()
    sink = MemorySink(store, HeuristicDistiller(), episodic_path_prefix="Test/Prefix")

    await sink.emit(_run_started(task_id=_TASK_ID))
    await sink.emit(_user_turn("Do something"))
    await sink.emit(_run_finished())

    assert len(store.writes) == 1
    path, content = store.writes[0]
    assert path.startswith("Test/Prefix/")
    assert path.endswith(".md")
    assert "Do something" in content


@pytest.mark.asyncio
async def test_memory_sink_no_write_before_run_finished() -> None:
    store = _CapturingStore()
    sink = MemorySink(store, HeuristicDistiller())

    await sink.emit(_run_started())
    await sink.emit(_user_turn("hello"))

    assert store.writes == []


@pytest.mark.asyncio
async def test_memory_sink_swallows_write_failure() -> None:
    store = _CapturingStore()
    store.fail = True
    sink = MemorySink(store, HeuristicDistiller())

    await sink.emit(_run_started())
    await sink.emit(_run_finished())
    # no exception raised


@pytest.mark.asyncio
async def test_memory_sink_uses_run_id_when_no_task_id() -> None:
    store = _CapturingStore()
    sink = MemorySink(store, HeuristicDistiller())

    await sink.emit(_run_started())  # no task_id
    await sink.emit(_run_finished())

    _, content = store.writes[0]
    assert f"run_id: {_RUN_ID}" in content


@pytest.mark.asyncio
async def test_memory_sink_path_uses_record_id() -> None:
    store = _CapturingStore()
    sink = MemorySink(store, HeuristicDistiller(), episodic_path_prefix="Claude/Memory/Episodic")

    await sink.emit(_run_started())
    await sink.emit(_run_finished())

    path, _ = store.writes[0]
    assert re.match(r"Claude/Memory/Episodic/[0-9a-f-]+\.md", path)
