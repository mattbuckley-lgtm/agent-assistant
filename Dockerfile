# Multi-stage build: uv resolves/installs dependencies in a `python:3.12-slim`
# builder; the runtime stage copies only the resulting venv + source and
# never sees uv, pip, or build tooling. Runs as a non-root user with no
# writable paths other than what compose.yaml mounts as tmpfs -- pair with
# `read_only: true`, `cap_drop: [ALL]`, and `security_opt:
# [no-new-privileges:true]` at the container level (see deploy/compose.yaml).
#
# The same image serves the agent (`agent/__main__.py`, the default
# entrypoint) and, with an overridden entrypoint, each isolated MCP server
# under `mcp_servers/` (see deploy/compose.yaml).
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN groupadd --system agent && useradd --system --gid agent --no-create-home agent

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

USER agent

ENTRYPOINT ["python", "-m", "agent"]
