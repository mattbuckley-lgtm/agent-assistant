"""Skill model: a local progressive-disclosure package.

A skill is a directory containing a `SKILL.md` whose frontmatter declares
`name`, `description`, and `when_to_use`. The agent's system prompt is
composed from the *index* (name + description + when_to_use) of every
registered skill; the full body is loaded into context only when a skill is
selected (see `agent/skills/registry.py`).

Skills extend behaviour/knowledge in-process. They never execute capability
themselves -- that always flows through the MCP layer + permission
interceptor (`agent/mcp/`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Skill(BaseModel):
    name: str
    description: str
    when_to_use: str
    path: Path

    def load_body(self) -> str:
        """Read the full SKILL.md body on demand (progressive disclosure)."""
        return self.path.read_text()
