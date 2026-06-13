"""GenAI semantic-convention attribute keys, used by model adapters and the
OTel-emitting sink so every span carries uniform attributes.

See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Non-standard extensions, namespaced under gen_ai.* for consistency.
GEN_AI_USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read_tokens"
GEN_AI_USAGE_CACHE_WRITE_TOKENS = "gen_ai.usage.cache_write_tokens"
GEN_AI_USAGE_COST_USD = "gen_ai.usage.cost_usd"
GEN_AI_USAGE_ESTIMATED = "gen_ai.usage.estimated"

# Tool-call attributes.
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Agent-runtime-specific attributes (not part of GenAI semconv).
AGENT_RUN_ID = "agent.run_id"
AGENT_STEP_INDEX = "agent.step_index"
AGENT_MCP_SERVER = "agent.mcp.server"
AGENT_PERMISSION_DECISION = "agent.permission.decision"
AGENT_PERMISSION_REASON = "agent.permission.reason"
