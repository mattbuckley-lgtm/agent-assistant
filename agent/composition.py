"""Composition root: builds concrete `Model`/`ToolRegistry`/etc.
implementations from `AgentSettings`. Nothing in `agent/core` imports this
module -- only `agent/__main__.py` (and, eventually, other entry points like
an HTTP server) do.
"""

from __future__ import annotations

import os

from agent.config import AgentSettings, ModelConfig
from agent.core.interfaces import Model, PermissionPolicy, SkillRegistry
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.models.anthropic import AnthropicModel
from agent.models.openai_compat import OpenAICompatModel
from agent.models.prompted_tools import PromptedToolsModel
from agent.models.replay import ReplayModel
from agent.skills.registry import EmptySkillRegistry, FileSystemSkillRegistry


def build_model(config: ModelConfig) -> Model:
    model = _build_base_model(config)
    if not config.native_tool_calling:
        model = PromptedToolsModel(model)
    return model


def _build_base_model(config: ModelConfig) -> Model:
    if config.provider == "replay":
        if config.cassette_path is None:
            raise ValueError("model.provider='replay' requires model.cassette_path")
        return ReplayModel(config.cassette_path, name=config.name)

    api_key = os.environ[config.api_key_env] if config.api_key_env else None

    if config.provider == "anthropic":
        return AnthropicModel(
            config.name,
            api_key=api_key,
            price_per_input_token_usd=config.price_per_input_token_usd,
            price_per_output_token_usd=config.price_per_output_token_usd,
        )
    if config.provider == "openai_compat":
        return OpenAICompatModel(
            config.name,
            base_url=config.base_url,
            api_key=api_key,
            price_per_input_token_usd=config.price_per_input_token_usd,
            price_per_output_token_usd=config.price_per_output_token_usd,
        )
    raise NotImplementedError(f"model.provider='{config.provider}' is not implemented yet")


def build_permissions(settings: AgentSettings) -> PermissionPolicy:
    return build_permissions_from_rules(settings.permissions)


def build_permissions_from_rules(rules: list[AllowRule] | None = None) -> PermissionPolicy:
    return AllowlistPolicy(rules)


def build_skills(settings: AgentSettings) -> SkillRegistry:
    if settings.skills_dir is None:
        return EmptySkillRegistry()
    return FileSystemSkillRegistry(settings.skills_dir)
