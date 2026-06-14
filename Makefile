.DEFAULT_GOAL := help

# Override on the command line if you use podman, e.g.:
#   make COMPOSE="podman compose" compose-up
COMPOSE ?= docker compose
COMPOSE_FILE := deploy/compose.yaml
PROMPT ?= Please echo 'hello'.

.PHONY: help
help:
	@echo "Targets:"
	@echo "  install        uv sync (install/update the dev environment)"
	@echo "  lint           ruff check"
	@echo "  format         ruff format (writes changes)"
	@echo "  format-check   ruff format --check"
	@echo "  typecheck      pyright"
	@echo "  test           pytest"
	@echo "  eval           run the Inspect AI echo-clock eval"
	@echo "  check          lint + format-check + typecheck + test (CI gate)"
	@echo "  run            run the agent CLI once (PROMPT=\"...\")"
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

.PHONY: eval
eval:
	uv run python -m inspect_ai eval evals/tasks/echo_clock.py

.PHONY: check
check: lint format-check typecheck test

.PHONY: run
run:
	uv run python -m agent "$(PROMPT)"

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
	rm -rf .pytest_cache .ruff_cache logs
