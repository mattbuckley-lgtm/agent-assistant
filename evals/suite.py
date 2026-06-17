"""Generic Inspect `Task` builder for case-based evals: turns a
`evals/cases/*.jsonl` file of `evals.spec.EvalCase` records into a `Task`
that runs each case through `evals.bridge.run_eval_case` and grades it with
the full `evals.scorers` suite.

Adding a new eval is a config-only change: append a record to one of
`evals/cases/*.jsonl` (or add a new file + `@task` in `evals/tasks/cases.py`)
and, if needed, a cassette under `tests/cassettes/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample, json_dataset

from evals.bridge import run_eval_case
from evals.scorers import (
    denied_tools_not_executed,
    no_unexpected_tool_calls,
    overall,
    response_includes,
    skills_used,
    stop_reason_matches,
    tool_calls_match,
    turn_denied_tools_not_executed,
    turn_no_unexpected_tool_calls,
    turn_responses_include,
    turn_stop_reasons_match,
    turn_tool_calls_match,
)
from evals.spec import EvalCase

CASES_DIR = Path(__file__).parent / "cases"


def _record_to_sample(record: dict[str, Any]) -> Sample:
    case = EvalCase.model_validate(record)
    display_input = case.input if not case.turns else case.turns[0].user_message
    display_target = (
        case.response_includes if not case.turns else (case.turns[-1].response_includes or "")
    ) or ""
    return Sample(
        input=display_input,
        target=display_target,
        id=case.name,
        metadata=case.model_dump(mode="json"),
    )


def case_task(filename: str, model: str = "replay", epochs: int = 1) -> Task:
    """Build a `Task` from `evals/cases/<filename>`: one sample per
    `EvalCase` record, run via `run_eval_case` and graded by every scorer in
    `evals.scorers` (each scorer no-ops for cases that don't set the relevant
    expectation).

    `model` selects what plays the assistant role: `"replay"` (default)
    deterministically replays each case's cassette; any other value is a
    registry key from `agent.toml`'s `[models]` (e.g. `"granite-local"`),
    resolved the same way as `python -m agent --model <key>`.

    `epochs` runs each sample that many times; accuracy/stderr metrics in the
    Inspect log viewer are then averages over all runs. Only meaningful with a
    real (stochastic) model -- replay is deterministic so epochs > 1 adds no
    information."""
    return Task(
        dataset=json_dataset(str(CASES_DIR / filename), sample_fields=_record_to_sample),
        solver=run_eval_case(model),
        epochs=epochs,
        scorer=[
            overall(),
            response_includes(),
            stop_reason_matches(),
            tool_calls_match(),
            skills_used(),
            denied_tools_not_executed(),
            no_unexpected_tool_calls(),
            turn_tool_calls_match(),
            turn_stop_reasons_match(),
            turn_responses_include(),
            turn_denied_tools_not_executed(),
            turn_no_unexpected_tool_calls(),
        ],
    )
