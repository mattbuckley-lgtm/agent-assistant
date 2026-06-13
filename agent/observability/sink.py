"""TranscriptSink implementations.

`InMemorySink` and `FanOutSink` are plain data structures usable everywhere
(evals, tests, prod). The OTel-emitting sink lives in `agent/observability/otel.py`
since it depends on tracer/exporter setup.
"""

from __future__ import annotations

from agent.core.events import TranscriptEvent
from agent.core.interfaces import TranscriptSink


class InMemorySink:
    """Collects every event in order. Used by evals and tests."""

    def __init__(self) -> None:
        self.events: list[TranscriptEvent] = []

    async def emit(self, event: TranscriptEvent) -> None:
        self.events.append(event)


class FanOutSink:
    """Forwards each event to every wrapped sink, in order."""

    def __init__(self, sinks: list[TranscriptSink]) -> None:
        self._sinks = sinks

    async def emit(self, event: TranscriptEvent) -> None:
        for s in self._sinks:
            await s.emit(event)
