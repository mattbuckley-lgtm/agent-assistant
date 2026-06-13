"""Default-deny allowlist permission policy.

Implements `agent.core.interfaces.PermissionPolicy`. Every evaluation is
emitted as a `PermissionDecided` transcript event by the loop -- this module
only decides, it does not emit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.core.events import Decision


class AllowRule(BaseModel):
    """A single (server, tool) grant, optionally constrained by argument
    prefixes (e.g. allow `fs.read` only when `path` starts with `/data/`)."""

    server: str
    tool: str
    decision: Decision = Decision.ALLOW
    arg_prefixes: dict[str, str] = Field(default_factory=dict)

    def matches(self, server: str, tool: str, args: dict[str, object]) -> bool:
        if self.server != server or self.tool != tool:
            return False
        for arg_name, prefix in self.arg_prefixes.items():
            value = args.get(arg_name)
            if not isinstance(value, str) or not value.startswith(prefix):
                return False
        return True


class AllowlistPolicy:
    """Default-deny: a tool call is allowed/prompted only if an `AllowRule`
    matches; everything else is denied."""

    def __init__(self, rules: list[AllowRule] | None = None) -> None:
        self._rules = rules or []

    def evaluate(self, server: str, tool: str, args: dict[str, object]) -> Decision:
        for rule in self._rules:
            if rule.matches(server, tool, args):
                return rule.decision
        return Decision.DENY
