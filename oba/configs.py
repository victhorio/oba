import json
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError, field_validator

from oba.ag.models.message import ModelID

_CONFIG_PATH = Path.home() / ".config" / "oba" / "settings.json"
T = TypeVar("T")


class Config(BaseModel):
    name: str
    model_id: ModelID = "gpt-5-mini"
    vault_path: str

    @field_validator("vault_path", mode="after")
    @classmethod
    def is_valid_dir(cls, path_str: str) -> str:
        path = Path(path_str)
        if not path.is_dir():
            raise ValueError("vault_path is not a valid directory")
        return path_str


@lru_cache
def load() -> Config:
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


def _read_settings(path: Path) -> Config:
    contents = path.read_text(encoding="utf-8")
    data = json.loads(contents)
    return Config.model_validate(data)


def _interactive_setup() -> Config:
    default_model_id = Config.model_fields["model_id"].default

    print("Let's set up OBA. Press Enter to accept defaults where available.")

    name = _prompt_required("Name")
    model_id = _prompt_with_default("Model ID", default_model_id)
    vault_path = _prompt_required("Path to the Vault")

    try:
        return Config(
            name=name,
            model_id=model_id,
            vault_path=vault_path,
        )
    except ValueError as e:
        # note that a ValidationError is also an instance of ValueError
        print(f"Failed to set these values: {e}")
        return _interactive_setup()


def _prompt_required(label: str) -> str:
    response = input(f"{label}: ").strip()
    if not response:
        print("This value is required.")
        return _prompt_required(label)
    return response


def _prompt_with_default(label: str, default: T, parser_fn: Callable[[str], T] = lambda x: x) -> T:
    prompt_text = f"{label} [{default}]: "

    response = input(prompt_text).strip()
    if not response:
        return default

    try:
        return parser_fn(response)
    except Exception:
        print("Please enter a valid value.")
        return _prompt_with_default(label, default, parser_fn)


def _write_settings(settings: Config, path: Path) -> None:
    payload: dict[str, Any] = settings.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Settings saved to `{path}`.")
