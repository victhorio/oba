import atexit
import json
import shutil
import sys
import tempfile
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from attrs import asdict, define
from rich import print

CONFIG_PATH = Path.home() / ".config" / "oba" / "settings.json"


@define
class Config:
    user_name: str
    vault_path: str


@lru_cache
def load(is_test: bool) -> Config:
    if is_test:
        return load_test_config()

    if config_from_path := _read_from_path(CONFIG_PATH):
        return config_from_path

    try:
        config = _setup_interactive()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(1)

    _write_to_path(config, CONFIG_PATH)
    return config


def load_test_config() -> Config:
    vault_example = Path("vault_example")

    if not vault_example.is_dir():
        print(
            "[red][bold]Error:[/bold] --test mode can only be run in the same directory as `vault_example/`[/red]"
        )
        sys.exit(1)

    # Create a temp directory and copy vault contents into it
    temp_dir = tempfile.mkdtemp(prefix="oba_test_vault_")
    shutil.copytree(vault_example, temp_dir, dirs_exist_ok=True)

    # Clean up on program exit
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)

    print(f"[white]Using temporary vault at[/white] [orange3]{temp_dir}[/orange3]")
    return Config(
        user_name="Developer",
        vault_path=temp_dir,
    )


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
    payload: dict[str, object] = asdict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def apply_oba_highlight(msg: str) -> str:
    return msg.replace("oba", "\033[34;1moba\033[0m")


class ValidationError(ValueError): ...
