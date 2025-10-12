"""Prompt loading helpers."""

from pathlib import Path

_BASE_DIR = Path(__file__).parent


def load(prompt_name: str) -> str:
    """Load a prompt by name without the .txt suffix."""

    path = _prompt_path(prompt_name)
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt file not found: {path}") from exc
    if not contents:
        raise RuntimeError(f"Prompt file {path} is empty")
    return contents


def _prompt_path(prompt_name: str) -> Path:
    return _BASE_DIR / f"{prompt_name}.txt"
