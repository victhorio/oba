import json
import sys
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from attrs import asdict, define

CONFIG_PATH = Path.home() / ".config" / "oba" / "settings.json"


@define
class Config:
    user_name: str
    vault_path: str


@lru_cache
def load() -> Config:
    if config_from_path := _read_from_path(CONFIG_PATH):
        return config_from_path

    try:
        config = _setup_interactive()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(1)

    _write_to_path(config, CONFIG_PATH)
    return config


def _read_from_path(path: Path) -> Config | None:
    try:
        with open(path, "r") as f:
            config_payload = json.load(f)

        try:
            return Config(**config_payload)
        except Exception:
            raise ValidationError()

    except FileNotFoundError:
        return None

    except (JSONDecodeError, ValidationError):
        print(f"Invalid JSON found at {CONFIG_PATH}.")
        result = input("Delete and create a new one? [y/N] ")
        if result.lower() == "y":
            return None
        else:
            sys.exit(1)


def _setup_interactive() -> Config:
    print(apply_oba_highlight("Let's set up oba.\n"))

    while True:
        name = input(apply_oba_highlight("What should oba call you? "))
        if name:
            break

    while True:
        path = Path(input("What's the path to your vault? ")).expanduser()
        if path.is_dir():
            break

        print("\nThat path does not point to a directory.")

    print()

    return Config(
        user_name=name,
        vault_path=path.as_posix(),
    )


def _write_to_path(config: Config, path: Path) -> None:
    payload: dict = asdict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def apply_oba_highlight(msg: str) -> str:
    return msg.replace("oba", "\033[34;1moba\033[0m")


class ValidationError(ValueError): ...
