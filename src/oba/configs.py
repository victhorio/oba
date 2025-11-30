import atexit
import json
import os
import shutil
import sys
import tempfile
from functools import lru_cache
from json import JSONDecodeError

from attrs import asdict, define
from rich import print

CONFIG_FILE_PATH = os.path.expanduser("~/.config/oba/settings.json")


@define
class Config:
    user_name: str
    vault_path: str


@lru_cache
def config_load(is_test: bool) -> Config:
    if is_test:
        return load_test_config()

    if config_from_path := _read_from_path(CONFIG_FILE_PATH):
        return config_from_path

    try:
        config = _setup_interactive()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(1)

    _write_to_path(config, CONFIG_FILE_PATH)
    return config


def special_dir_path(config: Config) -> str:
    path = os.path.join(config.vault_path, ".oba")
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def load_test_config() -> Config:
    example_vault_path = "vault_example"

    if not os.path.isdir(example_vault_path):
        print(
            "[red][bold]Error:[/bold] --test mode can only be run in the same directory as `vault_example/`[/red]"
        )
        sys.exit(1)

    # Create a temp directory and copy vault contents into it
    temp_dir = tempfile.mkdtemp(prefix="oba_test_vault_")
    shutil.copytree(example_vault_path, temp_dir, dirs_exist_ok=True)

    # Clean up on program exit
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    print(f"[white]Using temporary vault at[/white] [orange3]{temp_dir}[/orange3]")
    return Config(
        user_name="Developer",
        vault_path=temp_dir,
    )


def _read_from_path(path: str) -> Config | None:
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
        print(f"Invalid JSON found at {CONFIG_FILE_PATH}.")
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
        path = os.path.expanduser(input("What's the path to your vault? "))
        if os.path.isdir(path):
            break

        print("\nThat path does not point to a directory.")

    print()

    return Config(
        user_name=name,
        vault_path=path,
    )


def _write_to_path(config: Config, path: str) -> None:
    payload: dict[str, object] = asdict(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def apply_oba_highlight(msg: str) -> str:
    return msg.replace("oba", "\033[34;1moba\033[0m")


class ValidationError(ValueError): ...
