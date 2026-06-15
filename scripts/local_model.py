"""Auto-manage a local llama-server for `agent.toml`'s local model entries.

Reads `local_models.toml` (gitignored -- copy from `local_models.toml.example`
and adjust paths) for per-model-key llama-server settings: `gguf_path` or
`ollama_model`, plus optional `port` (default 8080), `ctx_size` (default
8192), and `extra_args`.

If `<model-key>` has no entry in `local_models.toml` (the common case for
"replay", "anthropic", or no MODEL at all), `<command...>` runs unchanged.
Otherwise: if a llama-server is already serving the right GGUF on the
configured port, it's reused as-is; otherwise one is started, `<command...>`
runs, and -- only if this invocation started it -- it's stopped afterward.

Usage:
  uv run python scripts/local_model.py <model-key> -- <command...>
  uv run python scripts/local_model.py --serve <model-key>

The second form (used by `make llama-server`) starts (or reports reuse of)
`<model-key>`'s llama-server in the foreground, for standalone debugging --
Ctrl-C or `make llama-server-stop` to stop it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

from ollama_gguf_path import OllamaModelNotFound, resolve_gguf_path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "local_models.toml"
READY_TIMEOUT_S = 300
POLL_INTERVAL_S = 2


def _load_entry(model_key: str) -> dict[str, object] | None:
    if not CONFIG_PATH.is_file():
        return None
    config = tomllib.loads(CONFIG_PATH.read_text())
    entry = config.get(model_key)
    if not isinstance(entry, dict):
        return None
    return cast(dict[str, object], entry)


def _get_int(entry: dict[str, object], key: str, default: int) -> int:
    value = entry.get(key, default)
    if isinstance(value, int):
        return value
    raise SystemExit(f"error: local_models.toml '{key}' must be an integer, got {value!r}")


def _get_str_list(entry: dict[str, object], key: str) -> list[str]:
    value = entry.get(key, [])
    if not isinstance(value, list):
        raise SystemExit(f"error: local_models.toml '{key}' must be a list, got {value!r}")
    return [str(item) for item in cast(list[object], value)]


def _gguf_path(entry: dict[str, object]) -> Path:
    if "gguf_path" in entry:
        return Path(str(entry["gguf_path"])).expanduser()
    if "ollama_model" in entry:
        try:
            return resolve_gguf_path(str(entry["ollama_model"]))
        except OllamaModelNotFound as exc:
            raise SystemExit(f"error: {exc}") from exc
    raise SystemExit("error: local_models.toml entry needs 'gguf_path' or 'ollama_model'")


def _served_model_name(port: int) -> str | None:
    """The GGUF filename a llama-server on `port` is currently serving, or
    None if nothing answers `/v1/models` there."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=1) as resp:
            body = json.loads(resp.read())
    except (OSError, urllib.error.URLError, ValueError):
        return None
    data = body.get("data", [])
    if not data:
        return None
    return str(data[0].get("id", ""))


def _llama_server_command(entry: dict[str, object], gguf_path: Path, port: int) -> list[str]:
    ctx_size = _get_int(entry, "ctx_size", 8192)
    extra_args = _get_str_list(entry, "extra_args")
    return [
        "llama-server",
        "-m",
        str(gguf_path),
        "--port",
        str(port),
        "--jinja",
        "-c",
        str(ctx_size),
        *extra_args,
    ]


def _conflict_message(port: int, running: str, gguf_path: Path) -> str:
    return (
        f"error: llama-server on :{port} is serving '{running}', expected "
        f"'{gguf_path.name}'. Stop it first with `make llama-server-stop`."
    )


def _wait_until_ready(port: int, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"error: llama-server exited early (code {proc.returncode})")
        if _served_model_name(port) is not None:
            return
        time.sleep(POLL_INTERVAL_S)
    raise SystemExit(f"error: llama-server on :{port} not ready after {READY_TIMEOUT_S}s")


def _serve(model_key: str) -> int:
    entry = _load_entry(model_key)
    if entry is None:
        print(
            f"error: no [{model_key}] entry in local_models.toml (see local_models.toml.example)",
            file=sys.stderr,
        )
        return 1

    port = _get_int(entry, "port", 8080)
    gguf_path = _gguf_path(entry)
    if not gguf_path.is_file():
        print(f"error: GGUF file not found: {gguf_path}", file=sys.stderr)
        return 1

    running = _served_model_name(port)
    if running == gguf_path.name:
        print(f"local_model: llama-server already serving {gguf_path.name} on :{port}")
        return 0
    if running is not None:
        print(_conflict_message(port, running, gguf_path), file=sys.stderr)
        return 1

    print(f"local_model: serving {gguf_path.name} on :{port} (Ctrl-C to stop)")
    return subprocess.run(_llama_server_command(entry, gguf_path, port)).returncode


def main(argv: list[str]) -> int:
    if argv[:1] == ["--serve"]:
        if len(argv) != 2:
            print("usage: local_model.py --serve <model-key>", file=sys.stderr)
            return 2
        return _serve(argv[1])

    if "--" not in argv:
        print("usage: local_model.py <model-key> -- <command...>", file=sys.stderr)
        return 2
    sep = argv.index("--")
    model_key, command = argv[0], argv[sep + 1 :]
    if not model_key or not command:
        print("usage: local_model.py <model-key> -- <command...>", file=sys.stderr)
        return 2

    entry = _load_entry(model_key)
    if entry is None:
        return subprocess.run(command).returncode

    port = _get_int(entry, "port", 8080)
    gguf_path = _gguf_path(entry)
    if not gguf_path.is_file():
        print(f"error: GGUF file not found: {gguf_path}", file=sys.stderr)
        return 1

    server_proc: subprocess.Popen[bytes] | None = None
    running = _served_model_name(port)
    if running == gguf_path.name:
        print(f"local_model: reusing llama-server on :{port} ({gguf_path.name})")
    elif running is not None:
        print(_conflict_message(port, running, gguf_path), file=sys.stderr)
        return 1
    else:
        print(f"local_model: starting llama-server on :{port} ({gguf_path.name}) ...")
        server_proc = subprocess.Popen(
            _llama_server_command(entry, gguf_path, port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_until_ready(port, server_proc)
        except SystemExit:
            server_proc.terminate()
            server_proc.wait()
            raise
        print(f"local_model: llama-server ready on :{port}")

    try:
        return subprocess.run(command).returncode
    finally:
        if server_proc is not None:
            print(f"local_model: stopping llama-server on :{port}")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
