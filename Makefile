.DEFAULT_GOAL := help

# Override on the command line if you use podman, e.g.:
#   make COMPOSE="podman compose" compose-up
COMPOSE ?= docker compose
COMPOSE_FILE := deploy/compose.yaml
PROMPT ?= Please echo 'hello'.

# Local model served via llama.cpp's `llama-server` (see `make llama-server`
# and the `[models.granite-local]` entry in agent.toml).
OLLAMA_MODEL ?= granite4:tiny-h
LLAMA_PORT ?= 8080

# Model registry key from agent.toml [models], e.g. `make run MODEL=anthropic`.
MODEL ?=
MODEL_FLAG = $(if $(MODEL),--model $(MODEL),)

# Eval suite assistant: "replay" (default) deterministically replays each
# case's cassette; any other agent.toml [models] key (e.g. granite-local)
# runs the same ground truth against that real model.
EVAL_MODEL ?= replay

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
	@echo "  eval           run the Inspect AI eval suite (evals/tasks/, EVAL_MODEL=<registry key>)"
	@echo "  eval-view      serve the Inspect log viewer over logs/ (http://localhost:7575)"
	@echo "  check          lint + format-check + typecheck + coverage (CI gate)"
	@echo "  run            run the agent CLI once (PROMPT=\"...\", MODEL=<registry key>)"
	@echo "  run-local      run the agent CLI once against llama-server (PROMPT=\"...\")"
	@echo "  chat           interactive streaming chat (MODEL=<registry key>)"
	@echo "  chat-local     interactive streaming chat against llama-server"
	@echo "  llama-server   serve an Ollama-pulled model via llama.cpp (OLLAMA_MODEL=...)"
	@echo "  llama-server-stop  stop the background llama-server process"
	@echo "  compose-build  build the agent image for deploy/compose.yaml"
	@echo "  compose-up     start the agent + Langfuse stack"
	@echo "  compose-down   stop the stack and remove volumes"
	@echo "  compose-logs   follow logs for the stack"
	@echo "  compose-eval-up    build + start the containerized MCP servers"
	@echo "  compose-eval       run the eval suite in a hardened container against"
	@echo "                     them (EVAL_MODEL=<registry key>, default granite-local)"
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
	uv run python -m inspect_ai eval evals/tasks/ -T model=$(EVAL_MODEL)

.PHONY: eval-view
eval-view:
	uv run inspect view start --log-dir logs

.PHONY: check
check: lint format-check typecheck coverage

.PHONY: run
run:
	uv run python -m agent $(MODEL_FLAG) "$(PROMPT)"

.PHONY: run-local
run-local:
	uv run python -m agent --model granite-local "$(PROMPT)"

.PHONY: chat
chat:
	uv run python -m agent --chat $(MODEL_FLAG)

.PHONY: chat-local
chat-local:
	uv run python -m agent --chat --model granite-local

.PHONY: llama-server
llama-server:
	@MODEL_PATH=$$(uv run python scripts/ollama_gguf_path.py $(OLLAMA_MODEL)) || exit 1; \
	echo "Serving $(OLLAMA_MODEL) from $$MODEL_PATH"; \
	llama-server -m "$$MODEL_PATH" --port $(LLAMA_PORT) --jinja -c 8192

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
# deploy/agent.container.toml, selected via AGENT_CONFIG_FILE). Defaults to
# the local llama-server model (EVAL_MODEL=granite-local) -- start it first
# with `make llama-server`. Override with e.g.
# `make compose-eval EVAL_MODEL=anthropic` (needs ANTHROPIC_API_KEY).
.PHONY: compose-eval
compose-eval: EVAL_MODEL = granite-local
compose-eval: compose-eval-up
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm --no-deps \
		-e AGENT_CONFIG_FILE=/app/deploy/agent.container.toml \
		-e INSPECT_LOG_DIR=/tmp/logs \
		--entrypoint python \
		agent -m inspect_ai eval evals/tasks/ -T model=$(EVAL_MODEL)

.PHONY: compose-eval-down
compose-eval-down:
	$(COMPOSE) -f $(COMPOSE_FILE) down

.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov logs
