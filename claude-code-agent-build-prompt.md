# Build Prompt: Composable, Observable, Evaluable Agent Runtime

## 0. Read this first (how to work)

This is a **learning-and-iteration project** as much as a delivery project. Optimise for a clean, well-factored codebase with sharp module boundaries that I can read, modify, and extend — not for cramming features in fast. Concretely:

- **Build a thin vertical slice end-to-end before going wide.** Get one model adapter, one trivial tool, the permission check, the transcript, the OTel→Langfuse export, and one passing Inspect eval all working *together* first. Only then add the second provider, more tools, and the skills layer.
- **Work in the phases defined in §7.** After each phase, stop at a runnable checkpoint with a short note on what works and how to exercise it. Do not start a later phase until the current one runs.
- **Interfaces before implementations.** Define the Protocols/Pydantic models in §3–§4 first and get me to a compiling skeleton, then fill in bodies.
- **Verify current SDK APIs — do not trust your priors.** The MCP Python SDK, `inspect-ai`, and Langfuse's OTLP ingestion all move fast and may have changed. Before using any of them, check the current docs/source for the real API surface and tell me if it diverges from what's described here. The architecture below is the contract; exact library call signatures are not.
- Ask me before introducing a heavyweight dependency or framework not named here (no LangChain/LlamaIndex/agent frameworks — we are building the loop ourselves to learn it).

## 1. Mission

Build a Python agent runtime that:

1. Runs an LLM-driven tool-use loop where the **underlying model is fully swappable** (local llama.cpp/vLLM via OpenAI-compatible endpoints; cloud via Anthropic API) behind one interface that hides all model-specific prompting and config.
2. Executes **MCP tools** through a client layer, with capability **locked down by a permission interceptor**.
3. Is extensible via a **skills layer** (progressive-disclosure packages).
4. Treats MCP tools and skills as **extension points** — adding either requires no changes to agent core.
5. Emits a **single structured transcript** that is the source of truth for both **observability** (OpenTelemetry → Langfuse) and an **eval suite** (Inspect AI).
6. Runs **locked down in a container**, with MCP servers process-isolated and OS-level hardening.

## 2. Stack & conventions

- **Python 3.12+, asyncio throughout.** The agent loop, model calls, and tool calls are all async.
- **Pydantic v2** for all data models (events, messages, config, tool specs).
- **uv** for dependency/venv management; `pyproject.toml`.
- **OpenTelemetry SDK** + GenAI semantic conventions for instrumentation; **Langfuse as an OTLP exporter destination only** (the agent must not import Langfuse-specific client code in core).
- **MCP**: official Python SDK (`mcp`).
- **Evals**: `inspect-ai`.
- Type-checked (`pyright`/`mypy` strict-ish), `ruff` for lint/format, `pytest` + `pytest-asyncio`.
- Structured logging via the transcript/OTel — not scattered `logging.info` calls in business logic.

## 3. Architecture overview

Six runtime elements behind one dependency-injected entrypoint:

```
                         ┌────────────────────────────────┐
   Task ───────────────► │        agent loop (core)        │
                         │   injected: model, tools,        │
                         │   skills, permissions, sink      │
                         └───┬──────────┬──────────┬────────┘
                             │          │          │
                  ┌──────────▼──┐  ┌────▼─────┐ ┌──▼──────────┐
                  │ Model iface │  │ MCP layer │ │ Skills layer │
                  │ (adapters)  │  │ + perms   │ │ (registry)   │
                  └──────┬──────┘  └────┬──────┘ └──────────────┘
                         │              │
              normalized │   permission │  every model call, tool call,
              tool-calls │   interceptor│  permission decision & step
              + usage    ▼              ▼  ──► Transcript (Pydantic) ──► OTel ──► Langfuse
                                                       │
                                                       └──► Inspect AI (eval)
```

Suggested module map:

```
agent/
  core/
    interfaces.py     # Protocols: Model, Tool, Skill, PermissionPolicy, TranscriptSink
    events.py         # Pydantic transcript event model (single source of truth)
    messages.py       # normalized Message / ContentBlock / ToolSpec types
    loop.py           # the agent loop
    state.py          # run state, RunResult
    entrypoint.py     # run_agent(...) — the injected callable
  models/
    base.py           # normalized stream events, Usage
    anthropic.py      # Anthropic adapter
    openai_compat.py  # local llama.cpp / vLLM / any OpenAI-compatible endpoint
    inspect_model.py  # adapter that delegates to Inspect's provided model (eval driver)
    replay.py         # record/replay cassette adapter
    prompted_tools.py # prompt-driven tool-call shim for models w/o native tool calling
    registry.py
  mcp/
    client.py         # MCP client manager (connect, list tools, call tool)
    permissions.py    # PermissionPolicy impls + interceptor
    registry.py       # tool discovery / extension point
  skills/
    base.py           # Skill model, SKILL.md loader (progressive disclosure)
    registry.py       # skill discovery / extension point
  observability/
    otel.py           # tracer/exporter setup
    semconv.py        # GenAI attribute helpers
    sink.py           # TranscriptSink impls: in-memory, OTel-emitting
  config.py           # Pydantic settings (model selection, server list, policy, otel)
evals/
  bridge.py           # Inspect solver wrapping run_agent with the injected model
  tasks/              # eval tasks/datasets
deploy/
  Dockerfile
  compose.yaml        # agent + langfuse + isolated MCP servers
skills/               # actual skill packages live here
tests/
```

## 4. Core interfaces (define these first)

These are the seams. Get them compiling before implementing anything behind them. Treat the signatures as the intended shape — adjust types as needed but preserve the boundaries.

### 4.1 Normalized messages & tool calls

The hard part of model abstraction is **tool-call normalization, not text**. Anthropic uses `tool_use` content blocks; OpenAI uses function-calling; local models often have weak or no native tool support. The core loop must never see provider specifics.

```python
# messages.py
class TextBlock(BaseModel): type: Literal["text"]; text: str
class ToolUseBlock(BaseModel):
    type: Literal["tool_use"]; id: str; name: str; input: dict
class ToolResultBlock(BaseModel):
    type: Literal["tool_result"]; tool_use_id: str
    content: str | list[dict]; is_error: bool = False

ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentBlock]

class ToolSpec(BaseModel):       # provider-agnostic tool description
    name: str; description: str; input_schema: dict   # JSON Schema
```

### 4.2 Model interface + normalized streaming

```python
# models/base.py
class Usage(BaseModel):
    input_tokens: int = 0; output_tokens: int = 0
    cache_read_tokens: int = 0; cache_write_tokens: int = 0
    cost_usd: float | None = None          # adapter computes/estimates

# normalized stream events — the ONLY thing the loop consumes
class TextDelta(BaseModel): type: Literal["text_delta"]; text: str
class ToolCallDelta(BaseModel): type: Literal["tool_call_delta"]; id: str; name: str | None; args_delta: str
class ToolCallComplete(BaseModel): type: Literal["tool_call_complete"]; block: ToolUseBlock
class StreamUsage(BaseModel): type: Literal["usage"]; usage: Usage
class StreamDone(BaseModel): type: Literal["done"]; stop_reason: str
StreamEvent = TextDelta | ToolCallDelta | ToolCallComplete | StreamUsage | StreamDone

# core/interfaces.py
class Model(Protocol):
    name: str
    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]: ...
```

Adapter responsibilities (this is where providers leak — handle it inside the adapter):
- Translate normalized `Message`/`ToolSpec` → provider request; translate provider stream → normalized `StreamEvent`s.
- Populate `Usage` per provider (Anthropic gives input/output/cache; OpenAI differs; **local endpoints give little/none — estimate from a tokenizer and mark it estimated**).
- For models **without native tool calling**, wrap with `prompted_tools.py`: inject tool specs + a strict output contract into the system prompt, parse tool calls out of the text stream, and surface them as the same `ToolCallComplete` events. The loop must not be able to tell the difference.

### 4.3 Permission policy + interceptor

The permission boundary is the MCP client, gating calls **before** they cross the process boundary to the (isolated) MCP server.

```python
class Decision(StrEnum): ALLOW = "allow"; DENY = "deny"; PROMPT = "prompt"

class PermissionPolicy(Protocol):
    def evaluate(self, server: str, tool: str, args: dict) -> Decision: ...

# HITL hook used when a policy returns PROMPT
HumanApproval = Callable[[str, str, dict], Awaitable[bool]]
```

- Ship an **allowlist policy**: default-deny, explicit `(server, tool)` allow entries, optional per-tool argument constraints (e.g. allow `fs.read` only under a path prefix). Per-tool granularity is the baseline; argument-level constraints are a nice-to-have on the same interface.
- Every evaluation emits a `PermissionDecided` transcript event (allow/deny/prompt + reason). This is observability **and** the hook the prompt-injection evals assert against.

### 4.4 Transcript: the single source of truth

One ordered, typed event stream per run. Everything else (OTel spans, Langfuse traces, eval scoring) is a projection of this. Emit at exactly three seams — **step boundary, model call (inside the adapter), tool call (inside the permission interceptor)**.

```python
# events.py — discriminated union on `type`
# RunStarted / RunFinished
# StepStarted(step) / StepFinished(step)
# ModelCallStarted(model, messages_digest) / ModelCallFinished(usage, stop_reason, latency_ms)
# ToolCallRequested(server, tool, args)
# PermissionDecided(decision, reason)
# ToolCallStarted / ToolCallFinished(result, is_error, latency_ms)
# Error(where, message)
# each event: run_id, step_index, ts, and OTel span/trace ids for correlation

class TranscriptSink(Protocol):
    async def emit(self, event: TranscriptEvent) -> None: ...
```

Provide two sinks: an in-memory list sink (used by evals/tests) and an OTel-emitting sink (spans + GenAI attributes). They can be composed (fan-out).

### 4.5 Skills

Skill = a local **progressive-disclosure** package: a directory with `SKILL.md` whose frontmatter declares `name`, `description`, `when_to_use`, plus optional bundled resources. The agent sees only names + descriptions in its system prompt; the full `SKILL.md` body is loaded into context only when that skill is selected. Skills extend **behaviour/knowledge in-process**; they may *reference* tools but never execute capability themselves — all execution still flows through the MCP layer + permission interceptor, so there is one capability path.

```python
class Skill(BaseModel):
    name: str; description: str; when_to_use: str
    path: Path
    def load_body(self) -> str: ...   # full SKILL.md, loaded on demand
```

### 4.6 Injected entrypoint (this is what makes it evaluable)

The agent must **accept** its dependencies, never self-wire from global config. The composition root in `__main__`/`config.py` builds real implementations; the eval bridge builds Inspect's model + an in-memory sink.

```python
async def run_agent(
    task: Task,
    *,
    model: Model,
    tools: ToolRegistry,
    skills: SkillRegistry,
    permissions: PermissionPolicy,
    sink: TranscriptSink,
    approval: HumanApproval | None = None,
    max_steps: int = 20,
) -> RunResult: ...
```

## 5. The agent loop (core/loop.py)

Standard tool-use loop, fully streamed:

1. Compose system prompt: base instructions + skill index (names/descriptions) + any selected skill bodies.
2. Per step (bounded by `max_steps`):
   - Emit `StepStarted`. Open an OTel span for the step.
   - Call `model.generate(messages, tools)`; consume `StreamEvent`s, accumulating the assistant message (text + `ToolUseBlock`s). Adapter emits `ModelCall*` + usage.
   - For each tool call: `ToolCallRequested` → `permissions.evaluate` → `PermissionDecided`. If `PROMPT`, await `approval`. If allowed, execute via MCP client; emit `ToolCall*`. Append `ToolResultBlock`s as a tool/user message.
   - If no tool calls → that's the final answer; break.
   - Emit `StepFinished`.
3. Emit `RunFinished` with `RunResult` (final message, full transcript, aggregate usage/cost).

Guard rails: max steps, max tool calls per step, and a hard stop if the same `(tool,args)` repeats N times (loop detection).

## 6. Observability

- One **trace per run**; spans nest via OTel context propagation: run → step → (model call | tool call).
- Use **GenAI semantic conventions** (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, etc.) set in `semconv.py` so the model adapters populate them uniformly.
- Export via OTLP. Langfuse is configured purely as an OTLP endpoint (env-driven). **No Langfuse import in `agent/core`** — keep it a destination. Confirm Langfuse's current OTLP ingestion path before wiring.
- Cost: adapters compute `cost_usd` from a small per-model price table (env/config); local models report `cost_usd=0` with estimated tokens flagged.

## 7. Build plan (phases — stop at each checkpoint)

**Phase 0 — Skeleton.** Project scaffolding (uv, pyproject, ruff, pyright, pytest). All interfaces in §4 defined; everything compiles and type-checks; no real behaviour. Checkpoint: `pytest` runs (empty), `pyright` clean.

**Phase 1 — Vertical slice (the important one).**
- `replay.py` model adapter (deterministic, reads a hand-written cassette) — lets us build the whole loop with zero network/model dependency.
- One trivial in-process "echo"/"clock" MCP tool *or* a minimal local MCP server.
- Allowlist permission policy.
- Agent loop end-to-end producing a transcript.
- In-memory + OTel sinks; OTLP export to a local Langfuse via compose.
- One Inspect eval that drives `run_agent` and scores the final answer.
Checkpoint: one command runs the agent on a task, produces a transcript, a trace appears in Langfuse, and `inspect eval` passes. **This is the slice everything else hangs off — get it clean.**

**Phase 2 — Real models behind the same interface.**
- Anthropic adapter (native tool calling, real usage/cache tokens).
- OpenAI-compatible adapter pointed at a local llama.cpp/vLLM endpoint.
- `prompted_tools.py` shim for local models lacking native tool calling.
Checkpoint: the *same* task + tools run unchanged across Anthropic and a local model by config only.

**Phase 3 — Extension points.**
- MCP registry: discover/connect multiple servers from config; tool discovery without touching loop code.
- Skills registry + `SKILL.md` loader with progressive disclosure; ship one example skill.
Checkpoint: adding a new MCP server (config) and a new skill (drop a folder) extends capability with no core changes.

**Phase 4 — Eval suite breadth.**
- Inspect tasks for: task-completion correctness, a record/replay regression suite (cassettes), and **adversarial/prompt-injection** cases asserting the agent never invokes a denied capability even when a tool *result* tries to coerce it (assert against `PermissionDecided`/`ToolCall*` events).
Checkpoint: `inspect eval` green across suites; injection cases demonstrably fail-safe.

**Phase 5 — Containerisation & lockdown** (see §8).

## 8. Containerisation & lockdown

- Multi-stage build → slim/distroless final image; non-root user; `--read-only` root FS with explicit writable tmpfs; `--cap-drop=ALL`; `--security-opt no-new-privileges`; seccomp default on.
- **MCP servers run as separate, isolated services** (own containers in `compose.yaml`), not in the agent process. The agent reaches them over the configured transport; the **permission interceptor is the in-agent gate** and the container/network policy is the outer gate (egress allowlist for servers that need the internet; none for those that don't).
- Secrets via env/secret mounts, never baked into the image.
- `compose.yaml` brings up: agent, Langfuse (+ its store), and the example MCP server(s), wired so the Phase-1 checkpoint runs in-container.

## 9. Eval facilitation — the one rule that makes it work

The **model interface is the eval seam**. It already supports three uses through one boundary: swap providers / inject a mock / replay a cassette. For Inspect, build toward **Inspect-as-driver**:

- `evals/bridge.py` exposes an Inspect `@solver` that constructs `run_agent` with an `inspect_model.py` adapter — i.e. a `Model` implementation that delegates to the model Inspect provides for the eval. Inspect then controls model selection and captures usage natively, while our transcript still records everything.
- Keep `run_agent` pure-injected so the *same* entrypoint is used in prod (real wiring), in regression (replay adapter), and in eval (Inspect-provided model). Verify the current Inspect solver/agent-bridge API before building this.

## 10. Non-goals & constraints

- No agent framework (LangChain/LlamaIndex/etc.). We build the loop.
- No Langfuse coupling in core — OTLP only.
- Don't collapse skills and MCP tools into one mechanism: two registries, one capability/execution path (MCP + permissions).
- Flag, don't guess: wherever the real MCP SDK / Inspect / Langfuse OTLP API differs from this document, tell me and propose the adjustment.

## 11. Acceptance criteria (overall)

- Switching model is a config change only; core loop and tools are untouched.
- Adding an MCP server or a skill requires no edits to `agent/core`.
- Every model call, tool call, and permission decision appears in the transcript, in the OTel trace, and in Langfuse.
- A denied capability is never executed, including when provoked by tool output, and there's an eval proving it.
- `run_agent` runs identically under prod wiring, replay, and Inspect-driven eval.
- The whole Phase-1 slice runs inside the locked-down container via `compose.yaml`.
