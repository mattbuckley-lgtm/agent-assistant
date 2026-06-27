"""Unit tests for McpMemoryStore search/write logic.

MCPServerConnection.connect/close require a running server so we swap the
connection with a lightweight fake. The connection lifecycle methods
(__aenter__/__aexit__) are covered here by mocking connect/close directly.
"""

from __future__ import annotations

import json

import pytest

from agent.config import MCPServerConfig
from agent.core.events import Provenance
from agent.core.memory import MemoryKind
from agent.memory.store import McpMemoryStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_DUMMY_CONFIG = MCPServerConfig(name="dummy", transport="stdio", command="false")


class _FakeContent:
    """Mimics mcp.types.TextContent / ImageContent .model_dump()."""

    def __init__(self, type_: str = "text", text: str = "") -> None:
        self._data: dict[str, object] = {"type": type_, "text": text}

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return self._data


class _FakeResult:
    def __init__(
        self,
        content: list[_FakeContent] | None = None,
        *,
        is_error: bool = False,
    ) -> None:
        self.content: list[_FakeContent] = content or []
        self.isError = is_error


class _FakeConnection:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._next_result: _FakeResult = _FakeResult()

    def set_result(self, result: _FakeResult) -> None:
        self._next_result = result

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def call_tool(self, tool: str, arguments: dict[str, object]) -> _FakeResult:
        self.calls.append((tool, arguments))
        return self._next_result


def _make_store(
    *, with_write: bool = True
) -> tuple[McpMemoryStore, _FakeConnection, _FakeConnection | None]:
    store = McpMemoryStore(
        _DUMMY_CONFIG,
        write_config=_DUMMY_CONFIG if with_write else None,
        search_tool="search_memory",
    )
    read_conn = _FakeConnection()
    write_conn = _FakeConnection() if with_write else None
    store._connection = read_conn  # type: ignore[assignment]
    store._write_connection = write_conn  # type: ignore[assignment]
    return store, read_conn, write_conn


def _text_item(payload: object) -> _FakeContent:
    return _FakeContent(type_="text", text=json.dumps(payload))


def _raw_record(
    *,
    rank: int = 1,
    source: str = "Claude/Memory/Episodic/rec-1.md",
    snippet: str = "hello world",
    entry_date: str = "2025-01-01",
    entities: str = "",
    title: str = "Test record",
) -> dict[str, object]:
    return {
        "rank": rank,
        "source": source,
        "snippet": snippet,
        "entry_date": entry_date,
        "entry_date_ts": 1735689600,
        "entities": entities,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_connects_read_and_write() -> None:
    store, read_conn, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    returned = await store.__aenter__()
    assert returned is store
    assert read_conn.connected
    assert write_conn.connected


@pytest.mark.asyncio
async def test_aenter_skips_write_when_absent() -> None:
    store, read_conn, _ = _make_store(with_write=False)
    await store.__aenter__()
    assert read_conn.connected


@pytest.mark.asyncio
async def test_aexit_closes_read_and_write() -> None:
    store, read_conn, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    await store.__aexit__(None, None, None)
    assert read_conn.closed
    assert write_conn.closed


@pytest.mark.asyncio
async def test_aexit_skips_write_close_when_absent() -> None:
    store, read_conn, _ = _make_store(with_write=False)
    await store.__aexit__(None, None, None)
    assert read_conn.closed


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_passes_correct_args() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(_FakeResult([_text_item([_raw_record()])]))

    await store.search("Claude/Memory/Episodic", "my question", top_k=5)

    assert len(read_conn.calls) == 1
    tool, args = read_conn.calls[0]
    assert tool == "search_memory"
    assert args["folder"] == "Claude/Memory/Episodic"
    assert args["question"] == "my question"
    assert args["top_k"] == 5


@pytest.mark.asyncio
async def test_search_returns_empty_on_error() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(_FakeResult(is_error=True))
    records = await store.search("Claude/Memory/Episodic", "q")
    assert records == []


@pytest.mark.asyncio
async def test_search_parses_list_response() -> None:
    store, read_conn, _ = _make_store()
    raw = [
        _raw_record(rank=1, source="Claude/Memory/Episodic/a.md", snippet="doc a"),
        _raw_record(rank=2, source="Claude/Memory/Episodic/b.md", snippet="doc b"),
    ]
    read_conn.set_result(_FakeResult([_text_item(raw)]))

    records = await store.search("Claude/Memory/Episodic", "q")
    assert len(records) == 2
    assert records[0].id == "a"
    assert records[0].content == "doc a"
    assert records[1].id == "b"


@pytest.mark.asyncio
async def test_search_parses_single_dict_response() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(
        _FakeResult([_text_item(_raw_record(source="Claude/Memory/Episodic/solo.md"))])
    )

    records = await store.search("Claude/Memory/Episodic", "q")
    assert len(records) == 1
    assert records[0].id == "solo"


@pytest.mark.asyncio
async def test_search_score_from_rank() -> None:
    store, read_conn, _ = _make_store()
    raw = [
        _raw_record(rank=1, source="Claude/Memory/Episodic/a.md"),
        _raw_record(rank=2, source="Claude/Memory/Episodic/b.md"),
    ]
    read_conn.set_result(_FakeResult([_text_item(raw)]))

    records = await store.search("Claude/Memory/Episodic", "q")
    assert records[0].score > records[1].score
    assert records[0].score == 1.0


@pytest.mark.asyncio
async def test_search_kind_episodic_from_path() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(
        _FakeResult([_text_item(_raw_record(source="Claude/Memory/Episodic/x.md"))])
    )
    records = await store.search("Claude/Memory/Episodic", "q")
    assert records[0].kind == MemoryKind.EPISODIC


@pytest.mark.asyncio
async def test_search_kind_semantic_from_path() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(
        _FakeResult([_text_item(_raw_record(source="Claude/Memory/Semantic/x.md"))])
    )
    records = await store.search("Claude/Memory/Semantic", "q")
    assert records[0].kind == MemoryKind.SEMANTIC


@pytest.mark.asyncio
async def test_search_provenance_absent_defaults_to_tool_output() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(_FakeResult([_text_item(_raw_record())]))
    records = await store.search("Claude/Memory/Episodic", "q")
    assert records[0].provenance == Provenance.TOOL_OUTPUT


@pytest.mark.asyncio
async def test_search_provenance_preserved_from_frontmatter() -> None:
    store, read_conn, _ = _make_store()
    rec = {**_raw_record(), "provenance": "agent_reasoning"}
    read_conn.set_result(_FakeResult([_text_item(rec)]))
    records = await store.search("Claude/Memory/Episodic", "q")
    assert records[0].provenance == Provenance.AGENT_REASONING


@pytest.mark.asyncio
async def test_search_source_and_content_passed_through() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(
        _FakeResult(
            [_text_item(_raw_record(source="Claude/Memory/Episodic/z.md", snippet="the content"))]
        )
    )
    records = await store.search("Claude/Memory/Episodic", "q")
    assert records[0].source == "Claude/Memory/Episodic/z.md"
    assert records[0].content == "the content"


@pytest.mark.asyncio
async def test_search_skips_non_text_content() -> None:
    store, read_conn, _ = _make_store()
    image_item = _FakeContent(type_="image", text="")
    read_conn.set_result(_FakeResult([image_item, _text_item(_raw_record())]))

    records = await store.search("Claude/Memory/Episodic", "q")
    assert len(records) == 1


@pytest.mark.asyncio
async def test_search_skips_malformed_json() -> None:
    store, read_conn, _ = _make_store()
    bad = _FakeContent(type_="text", text="not json {{{")
    read_conn.set_result(_FakeResult([bad]))

    records = await store.search("Claude/Memory/Episodic", "q")
    assert records == []


@pytest.mark.asyncio
async def test_search_skips_non_dict_list_elements() -> None:
    store, read_conn, _ = _make_store()
    read_conn.set_result(
        _FakeResult(
            [_text_item([_raw_record(source="Claude/Memory/Episodic/good.md"), "not a dict"])]
        )
    )
    records = await store.search("Claude/Memory/Episodic", "q")
    assert len(records) == 1
    assert records[0].id == "good"


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_calls_write_tool_with_path_and_content() -> None:
    store, _, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    write_conn.set_result(_FakeResult())

    await store.write("Claude/Memory/Episodic/abc.md", "# content")

    assert len(write_conn.calls) == 1
    tool, args = write_conn.calls[0]
    assert tool == "create_note"
    assert args["source"] == "Claude/Memory/Episodic/abc.md"
    assert args["content"] == "# content"


@pytest.mark.asyncio
async def test_write_noop_when_no_write_connection() -> None:
    store, read_conn, _ = _make_store(with_write=False)
    await store.write("some/path.md", "content")
    assert read_conn.calls == []


@pytest.mark.asyncio
async def test_write_raises_on_mcp_error() -> None:
    store, _, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    write_conn.set_result(_FakeResult(is_error=True))

    with pytest.raises(RuntimeError, match="memory write failed"):
        await store.write("bad/path.md", "content")


@pytest.mark.asyncio
async def test_write_raises_on_application_error_in_response() -> None:
    """obsidian-mcp-guard returns {"error": "..."} as body content, not as isError=True."""
    store, _, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    write_conn.set_result(_FakeResult([_text_item({"error": "host_vault_path_not_configured"})]))

    with pytest.raises(RuntimeError, match="memory write failed"):
        await store.write("Claude/Memory/Episodic/abc.md", "content")


@pytest.mark.asyncio
async def test_write_succeeds_when_response_is_plain_text() -> None:
    """Non-JSON text content in response should not cause an error."""
    store, _, write_conn = _make_store(with_write=True)
    assert write_conn is not None
    write_conn.set_result(_FakeResult([_FakeContent(type_="text", text="OK")]))

    await store.write("Claude/Memory/Episodic/abc.md", "content")
