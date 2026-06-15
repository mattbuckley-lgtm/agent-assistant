"""Resolve the on-disk GGUF path for an Ollama-pulled model.

Reads the model's manifest directly from `~/.ollama/models/manifests` --
unlike `ollama show --modelfile`, this needs no running Ollama service, so
`llama-server` can serve the same weights standalone.

Usage: uv run python scripts/ollama_gguf_path.py <model>[:<tag>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"


class OllamaModelNotFound(Exception):
    """Raised when an Ollama-pulled model's manifest or blob can't be found."""


def resolve_gguf_path(model: str) -> Path:
    """Return the on-disk GGUF blob path for an Ollama-pulled `model[:tag]`."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"

    manifest_path = (
        Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / name
    ) / tag
    if not manifest_path.is_file():
        raise OllamaModelNotFound(
            f"no Ollama manifest at {manifest_path} (pull it first with `ollama pull {model}`)"
        )

    manifest = json.loads(manifest_path.read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == _MODEL_LAYER_MEDIA_TYPE:
            digest = str(layer["digest"]).replace(":", "-")
            blob_path = Path.home() / ".ollama" / "models" / "blobs" / digest
            if not blob_path.is_file():
                raise OllamaModelNotFound(f"manifest references missing blob {blob_path}")
            return blob_path

    raise OllamaModelNotFound(f"no model layer found in {manifest_path}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: ollama_gguf_path.py <model>[:<tag>]", file=sys.stderr)
        return 2

    try:
        print(resolve_gguf_path(argv[0]))
        return 0
    except OllamaModelNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
