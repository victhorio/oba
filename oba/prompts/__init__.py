"""Prompt loading helpers."""

import re
from pathlib import Path

_BASE_DIR = Path(__file__).parent


def load(prompt_name: str, **kwargs: str) -> str:
    path = _prompt_path(prompt_name)
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt '{prompt_name}' file not found") from exc
    if not contents:
        raise RuntimeError(f"Prompt '{prompt_name}' file is empty")

    prompt = contents

    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"
        if placeholder not in prompt:
            raise RuntimeError(f"Placeholder {placeholder} not found in prompt '{prompt_name}'")
        prompt = prompt.replace(placeholder, value)

    match = re.search(r"{[^}]+}", prompt)
    if match:
        raise RuntimeError(
            f"Unreplaced placeholder {match.group(0)} found in prompt '{prompt_name}'"
        )

    return prompt


def _prompt_path(prompt_name: str) -> Path:
    return _BASE_DIR / f"{prompt_name}.txt"
