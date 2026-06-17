"""Case-based eval suite: one Inspect task per `evals/cases/*.jsonl` file.

Each file is a family of `evals.spec.EvalCase` records -- ground truth (input,
scripted cassette, expected tool calls/skills/response/stop reason) -- run
through `run_agent` with the real `agent.toml` configuration (skills, MCP
servers, permissions) unless a case overrides them. See `evals.suite`.

Run with: `uv run inspect eval evals/tasks/cases.py`

By default the assistant is a scripted `ReplayModel` that deterministically
replays each case's cassette. Pass `-T model=<key>` (a registry key from
`agent.toml`'s `[models]`, e.g. `granite-local` or `anthropic`) to run the
same ground truth against a real model instead: `uv run inspect eval
evals/tasks/ -T model=granite-local`.

Pass `-T epochs=N` to run each sample N times and average the results -- only
meaningful with a real model, since replay is deterministic.
"""

from __future__ import annotations

from inspect_ai import Task, task

from evals.suite import case_task


@task
def tool_choice(model: str = "replay", epochs: int = 1) -> Task:
    """Basic tool-choice ground truth: the agent calls the right tool (echo,
    or a second MCP server's word-counter) with the right arguments and
    reports the result."""
    return case_task("tool_choice.jsonl", model, epochs)


@task
def skills(model: str = "replay", epochs: int = 1) -> Task:
    """Skill usage ground truth: a skill is loaded and its instructions
    (call `clock`, format the result) are followed."""
    return case_task("skills.jsonl", model, epochs)


@task
def prompt_injection(model: str = "replay", epochs: int = 1) -> Task:
    """Adversarial: a mocked tool result embeds an instruction trying to
    coerce the model into calling a disallowed tool. Measures whether the
    model reports the legitimate tool content while ignoring the injected
    instruction, never requesting the disallowed tool."""
    return case_task("prompt_injection.jsonl", model, epochs)


@task
def multiturn(model: str = "replay", epochs: int = 1) -> Task:
    """Multi-turn conversation ground truth: the agent handles an ambiguous
    request by asking a clarifying question, then calls the right tool with
    the user's answer and reports the result."""
    return case_task("multiturn.jsonl", model, epochs)
