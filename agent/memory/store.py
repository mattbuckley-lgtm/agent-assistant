"""MemoryStore: Protocol + MCP-backed implementation.

MemoryStore is the read-side interface: retrieve records from a named
collection by semantic similarity to a query string.

McpMemoryStore wraps the markdown-rag MCP server, reusing the same
MCPServerConnection pattern as MCPToolRegistry. The caller manages the
lifecycle via async context manager.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Protocol, cast, runtime_checkable

from agent.config import MCPServerConfig
from agent.core.events import Provenance
from agent.core.memory import MemoryKind, MemoryRecord
from agent.mcp.client import MCPServerConnection


@runtime_checkable
class MemoryStore(Protocol):
    async def search(
        self,
        folder: str,
        question: str,
        *,
        top_k: int = 10,
    ) -> list[MemoryRecord]: ...


@runtime_checkable
class MemoryWriteStore(Protocol):
    """Write one markdown note to the vault via obsidian-mcp-guard."""

    async def write(self, path: str, content: str) -> None: ...


def _str_from(mapping: dict[str, object], key: str, default: str = "") -> str:
    val = mapping.get(key)
    return str(val) if val is not None else default


def _kind_from_source(source: str) -> MemoryKind:
    lower = source.lower()
    if "/episodic/" in lower or lower.endswith("/episodic"):
        return MemoryKind.EPISODIC
    return MemoryKind.SEMANTIC


def _score_from_rank(rank: object, total: int) -> float:
    """Linear score: rank 1 of N → 1.0, rank N → 1/N (minimum 0.0)."""
    r = int(rank) if isinstance(rank, (int, float)) else 1
    return max(0.0, 1.0 - (r - 1) / max(total, 1))


def _parse_rag_result(raw: dict[str, object], *, rank: int, total: int) -> MemoryRecord:
    """Parse one item from markdown-rag's /retrieve/dated response.

    Each item is flat: {rank, source, snippet, entry_date, entry_date_ts, entities, title}.
    There is no distance, no nested metadata, no provenance field — those are injected here.
    """
    source = _str_from(raw, "source")
    content = _str_from(raw, "snippet")
    # Use source stem as id (e.g. "abc-uuid" from "Claude/Memory/Episodic/abc-uuid.md")
    stem = source.rsplit("/", 1)[-1]
    record_id = stem[:-3] if stem.endswith(".md") else stem or source

    return MemoryRecord(
        id=record_id,
        kind=_kind_from_source(source),
        content=content,
        provenance=Provenance.TOOL_OUTPUT,
        source=source,
        task_id=None,
        score=_score_from_rank(raw.get("rank", rank), total),
    )


class McpMemoryStore:
    """Read/write memory via MCP servers.

    Read path:  markdown-rag  (`search` tool, queried by `search()`).
    Write path: obsidian-mcp-guard (`create_note` tool, called by `write()`).
    The write connection is optional; `write()` is a no-op when not configured.

    Usage::

        async with McpMemoryStore(read_config, write_config=write_cfg) as store:
            records = await store.search("episodic", "query", top_k=5)
            await store.write("Claude/Memory/Episodic/abc.md", content)
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        write_config: MCPServerConfig | None = None,
        search_tool: str = "search",
        write_tool: str = "create_note",
    ) -> None:
        self._connection = MCPServerConnection(config)
        self._write_connection = MCPServerConnection(write_config) if write_config else None
        self._search_tool = search_tool
        self._write_tool = write_tool

    async def __aenter__(self) -> McpMemoryStore:
        await self._connection.connect()
        if self._write_connection is not None:
            await self._write_connection.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._write_connection is not None:
            await self._write_connection.close()
        await self._connection.close()

    async def search(
        self,
        folder: str,
        question: str,
        *,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        result = await self._connection.call_tool(
            self._search_tool,
            {"question": question, "top_k": top_k, "folder": folder},
        )
        if result.isError:
            return []

        raw_items: list[dict[str, object]] = []
        for item in result.content:
            item_dict = item.model_dump(mode="json")
            if item_dict.get("type") != "text":
                continue
            text = item_dict.get("text", "")
            if not isinstance(text, str):
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                items: list[object] = cast(list[object], data)
                for elem in items:
                    if isinstance(elem, dict):
                        raw_items.append(cast(dict[str, object], elem))
            elif isinstance(data, dict):
                raw_items.append(cast(dict[str, object], data))

        total = len(raw_items)
        records: list[MemoryRecord] = []
        for i, raw in enumerate(raw_items):
            try:
                records.append(_parse_rag_result(raw, rank=i + 1, total=total))
            except Exception:  # noqa: BLE001
                pass
        return records

    async def write(self, path: str, content: str) -> None:
        """Write a markdown note to the vault. No-op when write_config is absent."""
        if self._write_connection is None:
            return
        result = await self._write_connection.call_tool(
            self._write_tool,
            {"source": path, "content": content},
        )
        if result.isError:
            raise RuntimeError(f"memory write failed for '{path}': {result.content}")
