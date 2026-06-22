# Build Prompt: Multi-Agent Registry & Sub-Agent Orchestration

## 0. Read this first (how to work)

This extends the existing runtime; it is not a rewrite. Same ethos as `claude-code-agent-build-prompt.md`: optimise for sharp module boundaries I can read and extend, not for cramming the feature in fast.

- **Hard invariant: do not modify `agent/core/`.** The whole point of this feature is that it drops in as new Protocol implementations. Sub-agents become a `ToolRegistry`; the orchestrator is just an agent with sub-agent tools. If you find yourself wanting to edit `loop.py`, `entrypoint.py`, or `interfaces.py`, stop and tell me why first — it almost certainly means the design is wrong, not the core.
- **Build a thin vertical slice before going wide.** Get a registry that loads two agent TOMLs, a parent that calls one sub-agent in-process and returns, and one passing test — all working together — before adding depth guards, budgets, or card-based routing.
- **Work in the phases in §7. Stop at a runnable checkpoint after each** with a one-line note on what works and how to exercise it. Don't start a later phase until the current one runs and `make check` is green.
- **Interfaces before implementations.** Define the Pydantic models (§3) and the new Protocol-implementing class signatures (§4) and get me to a compiling, type-clean skeleton before filling bodies.
- **Verify current SDK APIs — do not trust your priors.** Pydantic v2, `pydantic-settings`, and the MCP SDK move; check the real surface before using it.
- **No new heavyweight dependencies.** No agent frameworks, no A2A SDK, no graph libs. We are building the orchestration ourselves. Ask before adding anything not already in `pyproject.toml`.

## 1. Precondition

This builds on the per-agent system prompt already wired in: `AgentSettings.system_prompt` → `Task.system_prompt` (consumed by `compose_system_prompt` in `agent/core/loop.py`). If that field is **not** present in `agent/config.py` on this branch, add it first as a tiny standalone change (field on `AgentSettings`, threaded through `agent/__main__.py` where it currently passes `DEFAULT_SYSTEM_PROMPT`), with its own test, and checkpoint before starting §2.

## 2. Goal & the seam

A single base runtime should host **many named agents**, each defined by its own TOML, and any agent should be able to call other agents as **sub-agents**. An "orchestrator" is nothing special — it's an agent whose declared sub-agents are exposed to it as callable tools.

The seam, grounded in the existing code:

- `agent/core/interfaces.py` already defines `ToolRegistry` (`list_tool_specs`, `server_for_tool`, `call_tool`). The loop only ever sees this Protocol.
- `agent/core/loop.py` already synthesises in-loop tools for skills (`_skill_tool_specs`) that never reach MCP. Sub-agents follow the same spirit, so no core change is needed.
- The mechanism is forced by the "don't touch core" rule: the loop only discovers and invokes callables through the `ToolRegistry` Protocol, so for an orchestrator's *model* to choose a sub-agent mid-loop, that sub-agent must be presented to the loop as a tool. That is the only seam. **Compose** the sub-agent tools with the existing `MCPToolRegistry` so the loop sees one namespace. The parent's permission allowlist (`AllowlistPolicy`) then gates sub-agent calls exactly like any other tool — default-deny holds with no new policy code.

**Naming (addressing the "why not AgentRegistry?" question).** There are two genuinely different responsibilities, and they must not share a name:
- **`AgentRegistry`** = the *catalog/factory*: knows every agent in the directory, holds their cards, lazily builds one by name. This is the "registry of agents." (Already named this below.)
- **`SubAgentToolAdapter`** = the *glue* that adapts a parent's declared sub-agents into `ToolSpec`s and routes tool calls to `run_agent`. It is an **adapter** (GoF sense: agent → tool interface), not a second registry. Calling it `AgentRegistry` would collide with the catalog and produce two different "registries" — more confusing, not less. The `-Adapter` suffix is the whole point: it signals "sub-agents are not tools; they are being *adapted* to the tool interface the loop requires."

So: one `AgentRegistry` (catalog), one `SubAgentToolAdapter` per parent (its sub-agents-as-tools), merged by `CompositeToolRegistry` (which genuinely *is* a tool registry, so that name stays).

What you will add (all under a new `agent/agents/` package, plus composition wiring):

```
agent/agents/
  card.py          # AgentCard / Capability pydantic models (parsed from TOML)
  registry.py      # AgentRegistry: scan dir -> cards (eager) + lazy build(name)
  subagent_tools.py# SubAgentToolAdapter: adapts declared sub-agents into tools
  composite.py     # CompositeToolRegistry: merge N ToolRegistries into one namespace
```

## 3. Config schema (Pydantic v2, in `agent/config.py`)

Extend `AgentSettings` with identity + capability + sub-agent declarations. Borrow the **A2A Agent Card schema shape** for the descriptive fields (so a card could later be serialised to `/.well-known/agent-card.json` for free) but **do not** adopt the A2A protocol, transport, signing, or auth — out of scope, and wrong layer for in-process composition.

Naming: call the routing descriptors **`capabilities`**, never "skills" — `skills` already means filesystem progressive-disclosure packages in this codebase and the collision will confuse everything.

```toml
# in each agent's TOML (e.g. agents/researcher.toml)
name = "researcher"
description = "Searches the web and returns sourced, typed findings."

[[capabilities]]                 # maps 1:1 onto A2A AgentSkill fields
id = "web_research"
description = "Find and synthesise current information from the web."
tags = ["web", "search", "research"]
examples = ["latest on X", "compare A vs B as of this month"]

# referenced by registry name, resolved against the agents dir; recursive.
subagents = ["researcher", "fact_checker"]
```

- `AgentCard` / `Capability` models live in `agent/agents/card.py`. Populate `examples` deliberately — they are strong few-shot signal for an LLM router and worth more than prose alone.
- Add an `agents_dir: Path | None` (resolved relative to repo root, same validator pattern as `skills_dir`) and a `max_subagent_depth: int = 3`. A new top-level `[agents]`-style discovery mechanism is fine, but keep loading a single `agent.toml` working unchanged (back-compat).

## 4. Components

**`AgentRegistry` (`agent/agents/registry.py`)**
- On construction, scan `agents_dir` for `*.toml`, parse each into `(AgentCard, AgentSettings)`. **Eager** — cards are cheap and drive routing.
- `list_cards() -> list[AgentCard]` — for the orchestrator's tool specs / selection prompt.
- `build(name, *, ancestry, budget, sink, ...) -> <runtime>` — **lazy**: construct the runtime (model + MCP tools + skills + permissions) via `agent/composition.py` only when first requested, then **cache per parent session**. Re-use the cached runtime across repeated calls (MCP connections are expensive to establish); never tear down/rebuild per call.
- Index by `name`; raise a clear error listing available names on a miss (mirror `resolve_model`).

**`SubAgentToolAdapter` (`agent/agents/subagent_tools.py`)** — implements `ToolRegistry` (it has to, per §2), but is named an adapter because that is its role:
- `list_tool_specs()` → one `ToolSpec` per declared sub-agent, name = sub-agent name, description = its card description (+ a tight input schema, e.g. `{"task": str}`).
- `server_for_tool(name)` → a stable synthetic server id per sub-agent (e.g. `f"subagent:{name}"`) so permission rules can scope **per sub-agent**.
- `call_tool(server, tool, args)`:
  1. **Guards (in order, fail closed):** depth ≤ `max_subagent_depth`; `name` not already in `ancestry` (cycle); `budget` remaining > 0. On violation return an `is_error=True` `ToolResultBlock` with a clear message — never raise into the loop.
  2. Lazily `build(name, ancestry=ancestry+(parent,), budget=child_budget, ...)`.
  3. Run it through the **existing** `run_agent` (`agent/core/entrypoint.py`) on a **fresh `Task`** built from `args["task"]` — clean conversation state per invocation, no bleed between unrelated calls.
  4. Decrement/propagate budget from the child's `RunResult.usage`; return the child's `final_text()` as the tool result.

**`CompositeToolRegistry` (`agent/agents/composite.py`)** — implements `ToolRegistry`; merges an ordered list of registries into one tool namespace, delegating `server_for_tool`/`call_tool` to whichever owns the tool. Detect and reject duplicate tool names at construction. The parent agent gets `CompositeToolRegistry([MCPToolRegistry(...), SubAgentToolAdapter(...)])`.

**Composition wiring (`agent/composition.py` + a new entry path)**
- Add a `build_tool_registry(settings, *, registry, ancestry, budget, sink)` that returns the composite. Keep `agent/composition.py`'s "nothing in core imports me" property intact.
- The orchestrator run path (extend `agent/__main__.py`, e.g. `--agent <name>`) loads the `AgentRegistry`, builds the named top-level agent with an empty `ancestry` and the tree-wide `budget`, and runs it through the unchanged `run_agent`.

## 5. Budgets, depth, cycles (the part that bites)

Recursion + lazy loading hides cost: the full tree never appears at startup, it materialises level-by-level under load. The existing `MAX_TOOL_CALLS_PER_STEP` / `max_steps` bound a **single** loop, not the tree. So:
- Thread an explicit **tree-wide budget** (steps and/or tokens) from the top-level call down through `build` → `call_tool` → child `run_agent`, decremented at each hop. Exhaustion returns a clean error result, not an exception.
- Enforce `max_subagent_depth` and **cycle detection via the `ancestry` tuple** on every sub-agent call. These are not optional once sub-agents can declare sub-agents.

## 6. Transport: in-process vs isolated

- **Default = in-process** (the `SubAgentToolAdapter` above): trusted local composition, single trace, no process spawn, and it reuses the permission allowlist. Propagate the OTel/trace context (or a child `run_id`) into the child run so Langfuse shows the nested tree, not disconnected traces.
- **Isolation when needed** (e.g. a Playwright researcher touching the open internet): do **not** build a second transport into the registry. Wrap that sub-agent as a normal MCP server using the existing `mcp_servers/` pattern and register it as an ordinary `MCPServerConfig`. It then flows through `MCPToolRegistry` unchanged, fully process-isolated, with no new code path. Note this in the README as the documented escape hatch.

Keep the researcher's **return typed and sourced** (claims + source + confidence), and have the orchestrator transcribe rather than freely elaborate — the quarantine boundary is what limits both prompt-injection blast radius and reading-step error compounding.

## 7. Evals for the example sub-agents

Add Inspect AI eval coverage for the sub-agent feature, in the existing `evals/` harness style: JSONL cases in `evals/cases/`, one `@task` per file in `evals/tasks/cases.py`, scored by the transcript-based scorers in `evals/scorers.py`. These run under `make eval` and, with the `replay` model, deterministically in CI.

**Fixture cast.** Define a small set of example agents the evals drive — reuse the `agents/` examples (`orchestrator`, `researcher`, `fact_checker`). For eval runs each agent's model is `provider = "replay"` pointing at its **own** cassette. This is the one wrinkle to get right: a sub-agent has its **own conversation**, so the tree needs **one cassette per agent that runs** (orchestrator scripts "call the researcher with task X, then synthesise"; researcher scripts its own tool calls + typed answer). Put them under `tests/cassettes/` alongside the existing ones. Build the orchestrator through the `AgentRegistry` so the nested replays fire naturally.

**Transcript capture.** For the scorers to see *both* orchestrator-level routing and what happened *inside* a sub-agent, the child run's transcript events must reach the eval. Have the `SubAgentToolAdapter` run children with a sink that fans into the parent transcript (tag events with the sub-agent name / depth), so the existing `tool_call_requested` / `permission_decided` / `tool_call_started` events from sub-agents are scoreable. A sub-agent invocation surfaces as a `tool_call_requested` with `server = "subagent:<name>"` — which means the existing `tool_calls_match`, `denied_tools_not_executed`, and `no_unexpected_tool_calls` scorers work on sub-agent routing **for free**.

**Cases to add** (new `evals/cases/subagents.jsonl`, plus a `subagents` task in `evals/tasks/cases.py`):

1. **Routing — correct selection.** A task that should go to the researcher. Assert via `expected_tool_calls` that `subagent:researcher` was called with the delegated task, and the final answer includes the researcher's result. Add a second case whose task should route to `fact_checker`, with `unexpected_tool_calls` asserting `subagent:researcher` was *not* called. This is the capability/`examples`-driven routing check.

2. **Quarantine — injection across the sub-agent boundary** (the headline one, mirrors the existing `prompt_injection` case but through a sub-agent). The researcher's mock web tool returns content carrying an injected instruction that tries to get an effectful tool called or a write-capable sub-agent invoked. Use `denied_tools` + `unexpected_tool_calls` to assert the effectful call is **denied / never executed** and never requested. This directly evals the blast-radius containment: a hijacked researcher cannot escalate because its allowlist has no effectful tools and the orchestrator only consumes its return.

3. **Guards fire cleanly.** Cases that trip each guard — a cyclic fixture (agent A → B → A), an over-depth chain beyond `max_subagent_depth`, and a starved budget — and assert each terminates with a recognisable signal rather than runaway. For this, guards must be **observable in the transcript**: emit a distinct event (or a tool result with a stable marker) and either set a run `stop_reason` like `subagent_cycle_detected` / `subagent_depth_exceeded` / `budget_exhausted` (scored by the existing `stop_reason_matches`) or add a small scorer that asserts the marker is present and no further sub-agent calls followed. Pick whichever fits the harness more cleanly and note the choice.

If a new expectation doesn't map onto an existing `EvalCase` field/scorer (e.g. "selected exactly one sub-agent", or the guard markers), extend `evals/spec.py` and add a scorer in `evals/scorers.py` following the existing pattern (trivially CORRECT when the field is unset; wire it into `overall()`).

**Faithfulness (stretch, note don't force).** Whether the orchestrator transcribes the researcher's typed claims rather than fabricating is the compounding-error check — but it isn't deterministic under replay. Leave a `# TODO` case scaffold that runs only against a real model (`-T model=<key>`) with an NLI-style faithfulness scorer, so it's ready when you want to run the per-model comparison rather than block CI.

## 8. Phased delivery (checkpoint after each)

1. **Slice:** `AgentRegistry` loads two TOMLs from `agents/`; `SubAgentToolAdapter` + `CompositeToolRegistry` let a parent call one sub-agent in-process via the unchanged `run_agent`; one passing unit test using the `replay`/mock model. Runnable: `make run AGENT=orchestrator` (or equivalent) calls the sub-agent and returns.
2. **Guards:** depth limit, cycle detection, tree-wide budget, with tests for each failure mode (cycle, over-depth, budget-exhausted all return clean error results) and transcript-observable guard signals.
3. **Routing:** orchestrator selection driven by the card index (name/description/`examples`) so it picks the right sub-agent; permission rules scoped per `subagent:<name>` server.
4. **Evals (§7):** the `subagents.jsonl` cases + `subagents` task, green under `make eval` with the `replay` model; routing, quarantine, and guard cases all passing.
5. **Polish:** README, example agent TOMLs (`agents/orchestrator.toml`, `agents/researcher.toml`, `agents/fact_checker.toml`), and the MCP-isolation escape hatch documented and demonstrated.

## 9. Definition of done — do not skip

- **`make check` is green**: `ruff check`, `ruff format --check`, **pyright strict** (it includes `agent`, `evals`, `mcp_servers`, `scripts`, `tests`), and **coverage `fail_under = 90`** on `agent`. New modules need real unit tests to hold the line — cover the guard paths (cycle/depth/budget), lazy-build-and-cache, fresh-state-per-call, and composite namespace merging/duplicate rejection. Prefer the `replay` provider / mock tools (see `tests/cassettes/`, `evals/mock_tools.py`) so tests are deterministic and need no network or live model.
- **`make eval` is green** with the default `replay` model: the new `subagents` task passes (routing, quarantine, guards). Keep eval cases deterministic — per-agent cassettes, mock tools, no live model in CI.
- **README updated**: a "Multi-agent registry" section — how to define an agent TOML, the `capabilities`/`subagents` fields, how the orchestrator selects sub-agents, the depth/cycle/budget guards, the in-process-vs-MCP-isolation choice, and how to run the sub-agent evals. Update the module map and any config-reference table.
- Respect existing conventions: Pydantic v2, async throughout, `ruff` (line-length 100, rules `E,F,I,UP,B,ASYNC`), no `logging.*` in business logic (use the transcript/sink).
- After the final phase, run `make check` **and** `make eval` and paste both summaries. If coverage dips below 90, add tests — do not lower the threshold.

## Out of scope (note, don't build)
- A2A protocol/transport/Agent Card HTTP serving, signing, or auth — borrow only the card *schema shape*. A2A is the future **edge** wrapper for exposing the top-level agent to *external* agents; it is not the internal orchestration mechanism.
- Graph DB for capability discovery — markdown/TOML + the card index is sufficient at this scale.
