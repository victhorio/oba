import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel


class FileContent(BaseModel):
    file_name: str
    contents: str


def get_recent_dailies(vault_path_str: str, num_recent_notes: int = 3) -> str:
    if num_recent_notes <= 0:
        return "[system message: no recent daily notes available]"

    vault_path = Path(vault_path_str)
    if not vault_path.is_dir():
        raise ValueError(f"Vault path '{vault_path}' is not a directory")

    daily_folder = _get_daily_folder(vault_path)
    base_path_daily = daily_folder
    daily_entries = sorted(p for p in base_path_daily.iterdir() if p.is_file())
    recent_files_paths = daily_entries[-num_recent_notes:]

    recent_files: list[FileContent] = list()
    for file_path in recent_files_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()
        fc = FileContent(file_name=file_path.name, contents=contents)
        recent_files.append(fc)

    return format_notes(recent_files)


def format_notes(notes: list[FileContent]) -> str:
    template = "<note>\n<name>{name}</name>\n<contents>\n{contents}\n</contents>\n</note>"
    return "\n\n".join(
        template.format(
            name=note.file_name,
            contents=note.contents,
        )
        for note in notes
    )


def _get_daily_folder(vault_path: Path) -> Path:
    matches = [p for p in vault_path.iterdir() if p.is_dir() and "daily" in p.name.lower()]
    if not matches:
        raise RuntimeError("no folder containing 'daily' found in the vault root")
    if len(matches) > 1:
        raise RuntimeError(f"found multiple potential matches for daily folder in vault: {matches}")
    return matches[0]


def read_note(vault_path: str, note_name: str) -> str:
    notes_index = notes_index_build(vault_path)
    if note_name not in notes_index:
        raise FileNotFoundError(f"note '{note_name}' not found")

    with open(notes_index[note_name], "r") as f:
        return f.read()


@cache
def notes_index_build(vault_path: str) -> dict[str, str]:
    """
    Builds a map of {note name -> note file path} for the given vault.
    """

    # TODO: handle note name disambiguation the same way Obisidian does
    index: dict[str, str] = dict()

    for root, _, files in os.walk(vault_path):
        if ".obsidian" in root or ".trash" in root or ".oba" in root:
            continue

        for file in files:
            if file.endswith(".md"):
                if file in index:
                    raise RuntimeError(f"found two notes with the same name: {file}, fix the code")

                # skip empty files
                if not os.path.getsize(os.path.join(root, file)):
                    continue

                filename = file[:-3]
                index[filename] = os.path.join(root, file)

    return index
