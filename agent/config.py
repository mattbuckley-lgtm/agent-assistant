"""Pydantic settings: the composition root reads this to build real
implementations of the Protocols in `agent/core/interfaces.py`. Nothing in
`agent/core` imports this module -- only `__main__`/wiring code does."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from agent.mcp.permissions import AllowRule

_REPO_ROOT = Path(__file__).resolve().parent.parent

# .env (gitignored; see .env.example) holds machine-local settings such as
# AGENT_DEFAULT_MODEL and API keys (ANTHROPIC_API_KEY, ...). Loaded into the
# environment here -- not just into AgentSettings -- so AGENT_* overrides
# below and os.environ[api_key_env] lookups in agent/composition.py both see
# it, however the process was started (Make, `uv run python -m agent`,
# inspect_ai, ...). Existing environment variables win (override=False).
load_dotenv(_REPO_ROOT / ".env")


def _config_file() -> Path:
    """`agent.toml` by default; `AGENT_CONFIG_FILE` overrides it (e.g. the
    container image points this at `agent.container.toml`, which configures
    MCP servers as sibling services instead of subprocesses)."""
    override = os.environ.get("AGENT_CONFIG_FILE")
    if override is None:
        return _REPO_ROOT / "agent.toml"
    path = Path(override)
    return path if path.is_absolute() else _REPO_ROOT / path


class ModelConfig(BaseModel):
    """Selects which `Model` adapter to construct and how to talk to it."""

    provider: Literal["anthropic", "openai_compat", "replay"]
    name: str
    """Model id passed to the provider (e.g. "claude-sonnet-4-6")."""

    base_url: str | None = None
    """Endpoint for `openai_compat` (local llama.cpp/vLLM/etc)."""

    api_key_env: str | None = None
    """Env var holding the API key, if any."""

    native_tool_calling: bool = True
    """If False, wrap the adapter with `models/prompted_tools.py`."""

    cassette_path: Path | None = None
    """Cassette file for the `replay` provider."""

    price_per_input_token_usd: float | None = None
    price_per_output_token_usd: float | None = None


class MCPServerConfig(BaseModel):
    """One MCP server to connect to. Each runs as an isolated process/service."""

    name: str
    transport: Literal["stdio", "sse", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None


class OtelConfig(BaseModel):
    """OTLP export config. Langfuse (or any OTLP collector) is configured
    purely as a destination -- no Langfuse-specific code in agent/core."""

    enabled: bool = True
    endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str = "agent-runtime"


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        toml_file=_config_file(),
    )

    models: dict[str, ModelConfig]
    default_model: str
    system_prompt: str = "You are a helpful agent."
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list[MCPServerConfig])
    permissions: list[AllowRule] = Field(default_factory=list[AllowRule])
    otel: OtelConfig = Field(default_factory=OtelConfig)
    skills_dir: Path | None = None
    max_steps: int = 20

    @field_validator("skills_dir")
    @classmethod
    def _resolve_skills_dir(cls, value: Path | None) -> Path | None:
        """Resolve relative to the repo root, not the process cwd -- e.g.
        `inspect_ai` chdirs into `evals/tasks/` while loading eval tasks."""
        if value is None or value.is_absolute():
            return value
        return _REPO_ROOT / value

    def resolve_model(self, name: str | None = None) -> ModelConfig:
        """Look up a model by registry key, falling back to `default_model`."""
        key = name or self.default_model
        try:
            return self.models[key]
        except KeyError:
            available = ", ".join(sorted(self.models)) or "(none configured)"
            raise ValueError(f"Unknown model '{key}'. Available: {available}") from None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
