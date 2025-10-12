import json
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError

_CONFIG_PATH = Path.home() / ".config" / "oba" / "settings.json"
T = TypeVar("T")


class Settings(BaseModel):
    # NOTE: GPT-5 family models don't support temperature/top_p parameters which is why
    #       they're not included in the settings. Source:
    #       https://platform.openai.com/docs/guides/latest-model#gpt-5-parameter-compatibility

    model_id: str = "gpt-5-mini"
    max_history_turns: int = Field(default=20, ge=1)


@lru_cache
def load() -> Settings:
    try:
        return _read_settings(_CONFIG_PATH)
    except FileNotFoundError:
        print(f"No settings file found; creating {_CONFIG_PATH}.")
    except (JSONDecodeError, ValidationError) as exc:
        print(f"Invalid settings file at {_CONFIG_PATH}: {exc}")
        print("Let's create a new one.")

    settings = _interactive_setup()
    _write_settings(settings, _CONFIG_PATH)
    return settings


def _read_settings(path: Path) -> Settings:
    contents = path.read_text(encoding="utf-8")
    data = json.loads(contents)
    return Settings.model_validate(data)


def _interactive_setup() -> Settings:
    defaults = Settings()
    print("Let's set up OBA. Press Enter to accept the default value.")

    model_id = _prompt_with_default("Model ID", defaults.model_id)
    max_history_turns = _prompt_with_default(
        "Maximum history turns", defaults.max_history_turns, parser_fn=int
    )
    return Settings(model_id=model_id, max_history_turns=max_history_turns)


def _prompt_with_default(
    label: str, default: T, parser_fn: Callable[[str], T] = lambda x: x
) -> T:
    prompt_text = f"{label} [{default}]: "

    response = input(prompt_text).strip()
    if not response:
        return default

    try:
        return parser_fn(response)
    except Exception:
        print("Please enter a valid value.")
        return _prompt_with_default(label, default, parser_fn)


def _write_settings(settings: Settings, path: Path) -> None:
    payload: dict[str, Any] = settings.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Settings saved to `{path}`.")
