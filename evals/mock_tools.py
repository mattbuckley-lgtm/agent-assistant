"""Thin re-export: MockToolRegistry now lives in `agent.agents.mock_registry`
so `agent/agents/registry.py` can use it without importing from `evals/`.
"""

from agent.agents.mock_registry import MockToolRegistry

__all__ = ["MockToolRegistry"]
