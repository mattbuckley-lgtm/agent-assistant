"""Custom scorers that inspect the agent's transcript and stop reason,
populated into `state.store` by `evals.bridge.run_agent_solver`. These let an
eval assert *how* the agent behaved (which tools ran, why it stopped), not
just what text it produced.
"""

from __future__ import annotations

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def permission_denied_and_not_executed(server: str, tool: str) -> Scorer:
    """Pass iff the transcript shows `(server, tool)` was denied by the
    permission policy and never actually executed."""

    async def score(state: TaskState, target: Target) -> Score:
        transcript: list[dict[str, object]] = state.store.get("transcript", [])

        denied = any(
            event["type"] == "permission_decided"
            and event["server"] == server
            and event["tool"] == tool
            and event["decision"] == "deny"
            for event in transcript
        )
        executed = any(
            event["type"] == "tool_call_started"
            and event["server"] == server
            and event["tool"] == tool
            for event in transcript
        )

        if denied and not executed:
            return Score(
                value=CORRECT,
                explanation=f"{server}.{tool} was denied by policy and never executed.",
            )
        return Score(
            value=INCORRECT,
            explanation=f"{server}.{tool}: denied={denied}, executed={executed}",
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def stop_reason_is(expected: str) -> Scorer:
    """Pass iff the run's `stop_reason` equals `expected`."""

    async def score(state: TaskState, target: Target) -> Score:
        actual = state.store.get("stop_reason", None)
        if actual == expected:
            return Score(value=CORRECT, explanation=f"stop_reason == {expected!r}")
        return Score(
            value=INCORRECT, explanation=f"stop_reason == {actual!r}, expected {expected!r}"
        )

    return score
