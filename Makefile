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
	@echo "  eval           run the Inspect AI echo-clock eval"
	@echo "  check          lint + format-check + typecheck + coverage (CI gate)"
	@echo "  run            run the agent CLI once (PROMPT=\"...\", MODEL=<registry key>)"
	@echo "  run-local      run the agent CLI once against llama-server (PROMPT=\"...\")"
	@echo "  chat           interactive streaming chat (MODEL=<registry key>)"
	@echo "  chat-local     interactive streaming chat against llama-server"
	@echo "  llama-server   serve an Ollama-pulled model via llama.cpp (OLLAMA_MODEL=...)"
	@echo "  compose-build  build the agent image for deploy/compose.yaml"
	@echo "  compose-up     start the agent + Langfuse stack"
	@echo "  compose-down   stop the stack and remove volumes"
	@echo "  compose-logs   follow logs for the stack"
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
	uv run python -m inspect_ai eval evals/tasks/echo_clock.py

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
	llama-server -m "$$(ollama show $(OLLAMA_MODEL) --modelfile | awk '/^FROM/ {print $$2}')" \
		--port $(LLAMA_PORT) --jinja -c 8192

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

.PHONY: clean
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov logs
