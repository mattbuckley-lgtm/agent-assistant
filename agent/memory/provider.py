"""MemoryProvider implementations.

EmptyMemoryProvider (the default): returns [] for every recall call, so
all existing code paths remain unchanged when memory is not configured.

RagMemoryProvider: queries the episodic collection (scoped to task_id) and
the semantic collection (global), merges results by score, and trims to a
token budget. Backend failures are swallowed and degrade to [].

FixedMemoryProvider: returns a fixed list regardless of task or scope.
Used by evals to seed known facts without a live markdown-rag server.
"""

from __future__ import annotations

from agent.core.memory import MemoryRecord
from agent.core.messages import TextBlock
from agent.core.state import Task
from agent.memory.store import MemoryStore


def _task_to_query(task: Task) -> str:
    """First 500 chars of the first user message, falling back to task.id."""
    for msg in task.messages:
        if msg.role == "user":
            for block in msg.content:
                if isinstance(block, TextBlock):
                    return block.text[:500]
    return task.id


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


class EmptyMemoryProvider:
    """No-op provider (the default). Always returns [] so nothing regresses."""

    async def recall(self, task: Task, *, scope: str) -> list[MemoryRecord]:
        return []


class RagMemoryProvider:
    """Queries episodic + semantic folders and returns top-k within budget."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        episodic_folder: str = "Claude/Memory/Episodic",
        semantic_folder: str = "Claude/Memory/Semantic",
        top_k: int = 10,
        token_budget: int = 2000,
    ) -> None:
        self._store = store
        self._episodic_folder = episodic_folder
        self._semantic_folder = semantic_folder
        self._top_k = top_k
        self._token_budget = token_budget

    async def recall(self, task: Task, *, scope: str) -> list[MemoryRecord]:
        question = _task_to_query(task)
        try:
            episodic = await self._store.search(
                self._episodic_folder,
                question,
                top_k=self._top_k,
            )
            semantic = await self._store.search(
                self._semantic_folder,
                question,
                top_k=self._top_k,
            )
        except Exception:  # noqa: BLE001 — backend failure: degrade to no memories
            return []

        merged = sorted(episodic + semantic, key=lambda r: r.score, reverse=True)
        return self._trim(merged)

    def _trim(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        result: list[MemoryRecord] = []
        tokens_used = 0
        for r in records:
            t = _token_estimate(r.content)
            if tokens_used + t > self._token_budget:
                break
            result.append(r)
            tokens_used += t
            if len(result) >= self._top_k:
                break
        return result


class FixedMemoryProvider:
    """Returns a fixed list of records regardless of task or scope. Used in evals."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self._records = records

    async def recall(self, task: Task, *, scope: str) -> list[MemoryRecord]:
        return list(self._records)
