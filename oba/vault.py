from pathlib import Path

from pydantic import BaseModel


class FileContent(BaseModel):
    file_name: str
    contents: str


def get_recent_dailies(vault_path_str: str, n: int = 3) -> list[FileContent]:
    if n <= 0:
        return []

    vault_path = Path(vault_path_str)
    if not vault_path.is_dir():
        raise ValueError(f"Vault path '{vault_path}' is not a directory")

    daily_folder = _get_daily_folder(vault_path)
    base_path_daily = daily_folder
    daily_entries = sorted(p for p in base_path_daily.iterdir() if p.is_file())
    recent_files_paths = daily_entries[-n:]

    recent_files: list[FileContent] = list()
    for file_path in recent_files_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()
        fc = FileContent(file_name=file_path.name, contents=contents)
        recent_files.append(fc)

    return recent_files


def format_notes(notes: list[FileContent]) -> str:
    template = "<note>\n<name>{name}</name>\n<contents>\n{contents}\n</contents>\n</note>"
    return "\n\n".join(
        template.format(name=note.file_name, contents=note.contents) for note in notes
    )


def _get_daily_folder(vault_path: Path) -> Path:
    matches = [p for p in vault_path.iterdir() if p.is_dir() and "daily" in p.name.lower()]
    if not matches:
        raise RuntimeError("no folder containing 'daily' found in the vault root")
    if len(matches) > 1:
        raise RuntimeError(f"found multiple potential matches for daily folder in vault: {matches}")
    return matches[0]
