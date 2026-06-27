"""Unit tests for Phase 1 memory providers: Empty, Rag, and Fixed.

Also covers: _parse_record / _str_from in store.py, EpisodicRecord /
SemanticFact data models in records.py, and run_agent memory-provider
integration (entrypoint.py lines 49-52).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent.core.entrypoint import run_agent
from agent.core.events import Provenance
from agent.core.memory import MemoryKind, MemoryRecord
from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy
from agent.memory.provider import EmptyMemoryProvider, FixedMemoryProvider, RagMemoryProvider
from agent.memory.records import EpisodicRecord, SemanticFact
from agent.memory.store import (
    _parse_rag_result,  # pyright: ignore[reportPrivateUsage]
    _provenance_from_str,  # pyright: ignore[reportPrivateUsage]
    _str_from,  # pyright: ignore[reportPrivateUsage]
)
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from agent.skills.registry import EmptySkillRegistry

MEMORY_CASSETTE = Path(__file__).parent / "cassettes" / "memory_recall.json"


class _NullTools:
    def list_tool_specs(self) -> list[ToolSpec]:
        return []

    def server_for_tool(self, tool_name: str) -> str:
        return ""

    async def call_tool(
        self, server: str, tool: str, args: dict[str, object]
    ) -> tuple[ToolResultBlock, Provenance]:
        raise NotImplementedError


def _task(text: str = "test query", task_id: str | None = None) -> Task:
    return Task(
        id="test",
        task_id=task_id,
        messages=[Message(role="user", content=[TextBlock(text=text)])],
    )


def _record(
    id: str = "r1",
    kind: MemoryKind = MemoryKind.SEMANTIC,
    content: str = "A fact.",
    provenance: Provenance = Provenance.AGENT_REASONING,
    score: float = 0.9,
    task_id: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=id, kind=kind, content=content, provenance=provenance, score=score, task_id=task_id
    )


class FakeMemoryStore:
    """Returns records filtered by folder path prefix."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self._records = records
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        folder: str,
        question: str,
        *,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        self.calls.append({"folder": folder, "question": question})
        is_episodic = "episodic" in folder.lower()
        kind = MemoryKind.EPISODIC if is_episodic else MemoryKind.SEMANTIC
        results = [r for r in self._records if r.kind == kind]
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


class BrokenStore:
    async def search(
        self,
        folder: str,
        question: str,
        *,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        raise RuntimeError("backend failure")


async def test_empty_provider_returns_empty() -> None:
    result = await EmptyMemoryProvider().recall(_task(), scope="test-scope")
    assert result == []


async def test_fixed_provider_returns_given_records() -> None:
    records = [_record("a"), _record("b")]
    result = await FixedMemoryProvider(records).recall(_task(), scope="any")
    assert result == records


async def test_rag_provider_queries_both_folders() -> None:
    store = FakeMemoryStore(
        [_record("s1", MemoryKind.SEMANTIC), _record("e1", MemoryKind.EPISODIC)]
    )
    await RagMemoryProvider(store).recall(_task(), scope="t1")
    folders = [str(c["folder"]) for c in store.calls]
    assert any("Episodic" in f for f in folders)
    assert any("Semantic" in f for f in folders)


async def test_rag_provider_merges_and_sorts_by_score() -> None:
    records = [
        _record("e1", MemoryKind.EPISODIC, score=0.8),
        _record("s1", MemoryKind.SEMANTIC, score=0.95),
        _record("e2", MemoryKind.EPISODIC, score=0.5),
    ]
    store = FakeMemoryStore(records)
    result = await RagMemoryProvider(store).recall(_task(), scope="t1")
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].score == 0.95


async def test_rag_provider_degrades_on_backend_failure() -> None:
    result = await RagMemoryProvider(BrokenStore()).recall(_task(), scope="any")
    assert result == []


async def test_rag_provider_trims_to_token_budget() -> None:
    long_content = "x" * 400  # 400 chars → 100 tokens (len // 4)
    records = [
        _record(f"r{i}", MemoryKind.SEMANTIC, content=long_content, score=float(10 - i))
        for i in range(5)
    ]
    store = FakeMemoryStore(records)
    result = await RagMemoryProvider(store, token_budget=250).recall(_task(), scope="any")
    assert len(result) == 2  # 2 × 100 = 200 ≤ 250; 3 × 100 = 300 > 250


async def test_rag_provider_uses_task_message_as_question() -> None:
    store = FakeMemoryStore([])
    await RagMemoryProvider(store).recall(_task("What is the capital of France?"), scope="x")
    assert any("What is the capital" in str(c["question"]) for c in store.calls)


# ---------------------------------------------------------------------------
# _str_from and _parse_rag_result (agent/memory/store.py)
# ---------------------------------------------------------------------------


def test_str_from_returns_default_when_key_missing() -> None:
    assert _str_from({}, "missing", "default") == "default"


def test_str_from_converts_value_to_str() -> None:
    assert _str_from({"k": 42}, "k") == "42"


def test_str_from_returns_empty_default_when_omitted() -> None:
    assert _str_from({}, "x") == ""


def _rag_item(
    *,
    rank: int = 1,
    source: str = "Claude/Memory/Episodic/abc-123.md",
    snippet: str = "A fact.",
    title: str = "Test",
    provenance: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "rank": rank,
        "source": source,
        "snippet": snippet,
        "title": title,
        "entry_date": "2025-01-01",
    }
    if provenance is not None:
        item["provenance"] = provenance
    return item


def test_parse_rag_result_extracts_id_from_source() -> None:
    r = _parse_rag_result(_rag_item(source="Claude/Memory/Episodic/my-uuid.md"), rank=1, total=1)
    assert r.id == "my-uuid"


def test_parse_rag_result_content_is_snippet() -> None:
    r = _parse_rag_result(_rag_item(snippet="The content here."), rank=1, total=1)
    assert r.content == "The content here."


def test_parse_rag_result_kind_episodic_from_path() -> None:
    r = _parse_rag_result(_rag_item(source="Claude/Memory/Episodic/x.md"), rank=1, total=1)
    assert r.kind == MemoryKind.EPISODIC


def test_parse_rag_result_kind_semantic_from_path() -> None:
    r = _parse_rag_result(_rag_item(source="Claude/Memory/Semantic/x.md"), rank=1, total=1)
    assert r.kind == MemoryKind.SEMANTIC


def test_parse_rag_result_provenance_absent_defaults_to_tool_output() -> None:
    r = _parse_rag_result(_rag_item(), rank=1, total=1)
    assert r.provenance == Provenance.TOOL_OUTPUT


def test_parse_rag_result_provenance_agent_reasoning() -> None:
    r = _parse_rag_result(_rag_item(provenance="agent_reasoning"), rank=1, total=1)
    assert r.provenance == Provenance.AGENT_REASONING


def test_parse_rag_result_provenance_user_stated() -> None:
    r = _parse_rag_result(_rag_item(provenance="user_stated"), rank=1, total=1)
    assert r.provenance == Provenance.USER_STATED


def test_parse_rag_result_provenance_unknown_defaults_to_tool_output() -> None:
    r = _parse_rag_result(_rag_item(provenance="something_else"), rank=1, total=1)
    assert r.provenance == Provenance.TOOL_OUTPUT


def test_provenance_from_str_none_is_tool_output() -> None:
    assert _provenance_from_str(None) == Provenance.TOOL_OUTPUT


def test_parse_rag_result_score_rank1_is_1() -> None:
    r = _parse_rag_result(_rag_item(rank=1), rank=1, total=3)
    assert r.score == 1.0


def test_parse_rag_result_score_decreases_with_rank() -> None:
    r1 = _parse_rag_result(_rag_item(rank=1), rank=1, total=3)
    r2 = _parse_rag_result(_rag_item(rank=2), rank=2, total=3)
    assert r1.score > r2.score


# ---------------------------------------------------------------------------
# EpisodicRecord and SemanticFact data models (agent/memory/records.py)
# ---------------------------------------------------------------------------


def test_episodic_record_is_constructible() -> None:
    now = datetime.now(UTC)
    ep = EpisodicRecord(
        id="ep-1",
        task_id="t1",
        run_id="r1",
        timestamp=now,
        summary="A brief summary.",
        provenance=Provenance.AGENT_REASONING,
        entities=["Python"],
        tags=["language"],
    )
    assert ep.id == "ep-1"
    assert ep.entities == ["Python"]


def test_semantic_fact_is_constructible() -> None:
    now = datetime.now(UTC)
    fact = SemanticFact(
        id="sf-1",
        content="The project uses Python 3.12.",
        provenance=Provenance.USER_STATED,
        source_episode_ids=["ep-1"],
        timestamp=now,
    )
    assert fact.content == "The project uses Python 3.12."


# ---------------------------------------------------------------------------
# run_agent integration with memory_provider (agent/core/entrypoint.py)
# ---------------------------------------------------------------------------


async def _run_with_provider(provider: object) -> None:
    model = ReplayModel(MEMORY_CASSETTE)
    task = Task(
        id="mem-test",
        messages=[Message(role="user", content=[TextBlock(text="What language?")])],
    )
    result = await run_agent(
        task,
        model=model,
        tools=_NullTools(),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([]),
        sink=InMemorySink(),
        memory_provider=provider,  # type: ignore[arg-type]
    )
    assert result.stop_reason == "end_turn"


async def test_run_agent_with_fixed_memory_provider_succeeds() -> None:
    records = [_record("m1", content="Python 3.12 is the language.")]
    await _run_with_provider(FixedMemoryProvider(records))


async def test_run_agent_memory_provider_failure_degrades_gracefully() -> None:
    class _BrokenProvider:
        async def recall(self, task: Task, *, scope: str) -> list[MemoryRecord]:
            raise RuntimeError("provider failure")

    await _run_with_provider(_BrokenProvider())
