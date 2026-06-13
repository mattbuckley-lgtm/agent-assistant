"""Pydantic settings: the composition root reads this to build real
implementations of the Protocols in `agent/core/interfaces.py`. Nothing in
`agent/core` imports this module -- only `__main__`/wiring code does."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from agent.mcp.permissions import AllowRule


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
        toml_file="agent.toml",
    )

    model: ModelConfig
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list[MCPServerConfig])
    permissions: list[AllowRule] = Field(default_factory=list[AllowRule])
    otel: OtelConfig = Field(default_factory=OtelConfig)
    skills_dir: Path | None = None
    max_steps: int = 20

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
