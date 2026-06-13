"""Phase 0 checkpoint: the skeleton imports and the core types compose."""

from agent.core.entrypoint import run_agent
from agent.core.events import Decision, RunStarted
from agent.core.interfaces import PermissionPolicy
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy
from agent.observability.sink import InMemorySink


def test_run_agent_is_callable() -> None:
    assert callable(run_agent)


def test_task_and_message_construct() -> None:
    task = Task(id="t1", messages=[Message(role="user", content=[TextBlock(text="hi")])])
    assert task.messages[0].content[0].type == "text"


def test_allowlist_policy_default_deny() -> None:
    policy: PermissionPolicy = AllowlistPolicy()
    assert policy.evaluate("fs", "read", {}) == Decision.DENY


async def test_in_memory_sink_records_events() -> None:
    sink = InMemorySink()
    event = RunStarted(run_id="r1", task_name="t1")
    await sink.emit(event)
    assert sink.events == [event]
