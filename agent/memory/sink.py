"""MemorySink: a TranscriptSink that writes one episodic record on RunFinished.

Implements the same TranscriptSink Protocol as OtelSink. Add it to the
FanOutSink alongside OtelSink when memory.enabled is True; it is never
the primary sink. Fire-and-forget: write failures are swallowed so they
never abort the run (same contract as OtelSink's OTLP export).
"""

from __future__ import annotations

import logging

from agent.core.events import RunFinished, RunStarted, TranscriptEvent
from agent.memory.distiller import Distiller
from agent.memory.records import EpisodicRecord
from agent.memory.store import MemoryWriteStore

_log = logging.getLogger(__name__)

_FRONTMATTER_TEMPLATE = """\
---
id: {id}
task_id: {task_id}
run_id: {run_id}
timestamp: {timestamp}
provenance: {provenance}
tags:
  - episodic
entities: []
---

{body}
"""


def _format_episodic_record(record: EpisodicRecord) -> str:
    return _FRONTMATTER_TEMPLATE.format(
        id=record.id,
        task_id=record.task_id,
        run_id=record.run_id,
        timestamp=record.timestamp.isoformat(),
        provenance=record.provenance.value,
        body=record.summary,
    )


class MemorySink:
    """Accumulates transcript events and writes one EpisodicRecord on RunFinished.

    Constructed once per run by the composition layer and added to the
    FanOutSink. The write store lifecycle (MCP connection) is managed by the
    caller via async context manager on McpMemoryStore.
    """

    def __init__(
        self,
        write_store: MemoryWriteStore,
        distiller: Distiller,
        *,
        episodic_path_prefix: str = "Claude/Memory/Episodic",
    ) -> None:
        self._store = write_store
        self._distiller = distiller
        self._prefix = episodic_path_prefix
        self._events: list[TranscriptEvent] = []
        self._task_id: str | None = None

    async def emit(self, event: TranscriptEvent) -> None:
        self._events.append(event)
        if isinstance(event, RunStarted):
            self._task_id = event.task_id
        elif isinstance(event, RunFinished):
            await self._flush(event)

    async def _flush(self, event: RunFinished) -> None:
        task_id = self._task_id or event.run_id
        try:
            record = await self._distiller.distil(
                self._events,
                run_id=event.run_id,
                task_id=task_id,
            )
            content = _format_episodic_record(record)
            path = f"{self._prefix}/{record.id}.md"
            await self._store.write(path, content)
        except Exception:  # noqa: BLE001 — fire-and-forget; write never aborts the run
            _log.exception("MemorySink: failed to write episodic record for run %s", event.run_id)
