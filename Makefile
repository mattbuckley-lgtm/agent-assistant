.DEFAULT_GOAL := help

# Container compose command. Defaults to .env's COMPOSE (copy from
# .env.example) -- set this once if you use podman -- or "docker compose" if
# that's unset too. Override per invocation with e.g.
# `make COMPOSE="podman compose" compose-up`.
COMPOSE ?= $(shell sed -n 's/^COMPOSE=//p' .env 2>/dev/null)
COMPOSE := $(if $(COMPOSE),$(COMPOSE),docker compose)
COMPOSE_FILE := deploy/compose.yaml
PROMPT ?= Please echo 'hello'.

# Model registry key from agent.toml [models], used by run/chat/eval/
# compose-eval. Defaults to .env's AGENT_DEFAULT_MODEL (copy from
# .env.example) -- the same setting the agent itself uses via
# agent/config.py -- or "replay" if that's unset too. Override per
# invocation with e.g. `make run MODEL=anthropic`.
MODEL ?= $(shell sed -n 's/^AGENT_DEFAULT_MODEL=//p' .env 2>/dev/null)
MODEL := $(if $(MODEL),$(MODEL),replay)
MODEL_FLAG = --model $(MODEL)

# How many times to run each eval sample (averages accuracy/stderr over runs).
# Only meaningful with a real model -- replay is deterministic.
EPOCHS ?= 1

.PHONY: help
help:
	@echo "Targets:"
	@echo "  install        uv sync (install/update the dev environment)"
	@echo "  lint           ruff check"
	@echo "  format         ruff format (writes changes)"
	@echo "  format-check   ruff format --check"
	@echo "  typecheck      pyright"
	@echo "  test           pytest"
	@echo "  coverage       pytest with coverage report (fails under threshold)"
	@echo "  eval           run the Inspect AI eval suite (evals/tasks/, MODEL=<registry key>,"
	@echo "                 EPOCHS=N to run each sample N times and average results)"
	@echo "  eval-view      serve the Inspect log viewer over logs/ (http://localhost:7575)"
	@echo "  check          lint + format-check + typecheck + coverage (CI gate)"
	@echo "  run            run the agent CLI once (PROMPT=\"...\", MODEL=<registry key>)"
	@echo "  chat           interactive streaming chat (MODEL=<registry key>)"
	@echo "                 run/chat/eval/compose-eval auto-start/stop a matching"
	@echo "                 llama-server for local models (see local_models.toml.example)"
	@echo "  llama-server   serve MODEL's llama-server standalone, in the foreground"
	@echo "  llama-server-stop  stop the background llama-server process"
	@echo "  compose-build  build the agent image for deploy/compose.yaml"
	@echo "  compose-up     start the agent + Langfuse stack"
	@echo "  compose-down   stop the stack and remove volumes"
	@echo "  compose-logs   follow logs for the stack"
	@echo "  compose-eval-up    build + start the containerized MCP servers"
	@echo "  compose-eval       run the eval suite in a hardened container against"
	@echo "                     them (MODEL=<registry key>, default .env's"
	@echo "                     AGENT_DEFAULT_MODEL or \"replay\")"
	@echo "  compose-eval-down  stop the compose-eval containers"
	@echo "  clean          remove caches, __pycache__, and eval logs"

.PHONY: install
install:
	uv sync

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: format-check
format-check:
	uv run ruff format --check .

.PHONY: typecheck
typecheck:
	uv run pyright

.PHONY: test
test:
	uv run pytest -q

.PHONY: coverage
coverage:
	uv run pytest -q --cov=agent --cov-report=term-missing

.PHONY: eval
eval:
	uv run python scripts/local_model.py $(MODEL) -- \
		uv run python -m inspect_ai eval evals/tasks/ -T model=$(MODEL) -T epochs=$(EPOCHS)

.PHONY: eval-view
eval-view:
	uv run inspect view start --log-dir logs

.PHONY: check
check: lint format-check typecheck coverage

.PHONY: run
run:
	uv run python scripts/local_model.py $(MODEL) -- \
		uv run python -m agent $(MODEL_FLAG) "$(PROMPT)"

.PHONY: chat
chat:
	uv run python scripts/local_model.py $(MODEL) -- \
		uv run python -m agent --chat $(MODEL_FLAG)

# Serve MODEL's llama-server in the foreground for standalone debugging
# (requires a local_models.toml entry for MODEL -- see
# local_models.toml.example). `run`/`chat`/`eval`/`compose-eval` don't need
# this: they auto-start/stop a matching llama-server themselves.
.PHONY: llama-server
llama-server:
	uv run python scripts/local_model.py --serve $(MODEL)

.PHONY: llama-server-stop
llama-server-stop:
	pkill -f llama-server

.PHONY: compose-build
compose-build:
	$(COMPOSE) -f $(COMPOSE_FILE) build agent

.PHONY: compose-up
compose-up:
	$(COMPOSE) -f $(COMPOSE_FILE) up --build

.PHONY: compose-down
compose-down:
	$(COMPOSE) -f $(COMPOSE_FILE) down -v

.PHONY: compose-logs
compose-logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

.PHONY: compose-eval-up
compose-eval-up:
	$(COMPOSE) -f $(COMPOSE_FILE) build agent mcp-echo-clock mcp-wordcount
	$(COMPOSE) -f $(COMPOSE_FILE) up -d mcp-echo-clock mcp-wordcount

# Run the eval suite (evals/tasks/) inside a hardened, throwaway agent
# container against the containerized MCP servers over streamable_http (see
# deploy/agent.container.toml, selected via AGENT_CONFIG_FILE). Uses the same
# MODEL as `make eval` (.env's AGENT_DEFAULT_MODEL, or "replay"); for a local
# model, its llama-server is auto-started on the host via
# scripts/local_model.py + local_models.toml (the container reaches it
# through host.containers.internal). Override with e.g.
# `make compose-eval MODEL=anthropic` (needs ANTHROPIC_API_KEY in the
# environment -- .env is not forwarded into the container).
#
# Logs are bind-mounted to ./logs on the host (the agent's root filesystem is
# otherwise read-only, hence INSPECT_LOG_DIR=/app/logs) so `make eval-view`
# can serve results from this run alongside `make eval`'s. Runs as the host
# user/group so the bind mount is writable without loosening ./logs'
# permissions (the image's baked-in `agent` user has no relation to your host
# UID, so `chmod`-based fixes don't help here). Tears down the
# compose-eval-up containers afterwards either way (see compose-eval-down).
.PHONY: compose-eval
compose-eval: compose-eval-up
	mkdir -p logs
	uv run python scripts/local_model.py $(MODEL) -- \
		$(COMPOSE) -f $(COMPOSE_FILE) run --rm --no-deps \
			--user "$(shell id -u):$(shell id -g)" \
			-e AGENT_CONFIG_FILE=/app/deploy/agent.container.toml \
			-e INSPECT_LOG_DIR=/app/logs \
			-v $(CURDIR)/logs:/app/logs \
			--entrypoint python \
			agent -m inspect_ai eval evals/tasks/ -T model=$(MODEL) -T epochs=$(EPOCHS); \
	status=$$?; $(MAKE) compose-eval-down; exit $$status

.PHONY: compose-eval-down
compose-eval-down:
	$(COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov logs
