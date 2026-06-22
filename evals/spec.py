"""Declarative eval-case schema. Each record in `evals/cases/*.jsonl` is one
`EvalCase`, loaded into an Inspect `Sample` (as `metadata`) by `evals.suite`.

A case is a scripted conversation plus the ground truth the scorers in
`evals.scorers` check it against. Everything not specified by the case --
skills, MCP servers, the default permission policy -- comes from the real
`agent.toml` configuration (see `evals.bridge`), so evals exercise the same
wiring as production.

Single-turn cases set `input`, `cassette`, and top-level assertion fields.
Multi-turn cases set `turns` (a list of `ConversationTurn`), each with its own
user message, cassette, and per-turn assertions; `input` and `cassette` must
be omitted or left empty.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.agents.card import MockToolResult  # noqa: F401 (re-exported for backward compat)
from agent.mcp.permissions import AllowRule


class ExpectedToolCall(BaseModel):
    """A (server, tool) call the agent is expected to make. `args`, if
    given, must be a subset of the actual call's arguments."""

    server: str
    tool: str
    args: dict[str, object] | None = None


class SubAgentMockOverride(BaseModel):
    """Per-sub-agent overrides for eval cases that exercise sub-agent routing.

    `mock_tools` replaces the sub-agent's MCP tool registry with canned
    results so no live server is needed. `cassette` (if set) overrides the
    per-agent TOML's replay cassette for this specific eval case -- useful
    when multiple cases share the same agent TOML but need different model
    behaviour (e.g. the injection-carrying researcher vs the clean one).
    """

    agent_name: str
    mock_tools: list[MockToolResult] = Field(default_factory=list[MockToolResult])
    cassette: str = ""


class ConversationTurn(BaseModel):
    """One turn in a multi-turn conversation: a scripted user message, the
    cassette for the model's response to it, and the per-turn ground truth."""

    model_config = ConfigDict(frozen=True)

    user_message: str
    cassette: str

    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    unexpected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    denied_tools: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    expected_stop_reason: str | None = None
    response_includes: str | None = None


class EvalCase(BaseModel):
    """One eval case: a scripted conversation plus the ground truth the
    scorers in `evals.scorers` check it against. Each ground-truth field is
    opt-in -- an empty/unset field is not checked.

    Single-turn: set `input` + `cassette` + top-level assertion fields.
    Multi-turn: set `turns`; `input` and `cassette` must be omitted."""

    model_config = ConfigDict(frozen=True)

    name: str
    input: str = ""
    cassette: str = ""
    system_prompt: str = ""

    # Overrides of the shared agent.toml configuration.
    mock_tools: list[MockToolResult] = Field(default_factory=list[MockToolResult])
    permissions: list[AllowRule] | None = None

    # Sub-agent eval configuration (used with run_subagent_eval_case).
    subagent_mocks: list[SubAgentMockOverride] = Field(default_factory=list[SubAgentMockOverride])
    initial_ancestry: list[str] = Field(default_factory=list[str])
    initial_budget: int | None = None
    guard_signal: str | None = None

    # Single-turn ground truth (checked when `turns` is empty).
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    expected_skills: list[str] = Field(default_factory=list[str])
    denied_tools: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    unexpected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    expected_stop_reason: str | None = None
    response_includes: str | None = None

    # Multi-turn: ordered rounds of (user message, cassette, per-turn assertions).
    turns: list[ConversationTurn] = Field(default_factory=list[ConversationTurn])

    @model_validator(mode="after")
    def _check_conversation_spec(self) -> EvalCase:
        if not self.turns and not self.cassette:
            raise ValueError("specify 'cassette' for single-turn or 'turns' for multi-turn")
        return self
