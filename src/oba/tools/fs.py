import json
import os
import subprocess
from functools import partial

from ag.tool import Tool
from pydantic import BaseModel, Field

from ..vault import read_note


class ReadNote(BaseModel):
    """
    Use this function to read a note from the vault. If the note exists, it will be returned as a
    string (with nothing else), and if the note doesn't exist, it will return the following string:
    `[note {note_name} does not exist]`.
    """

    note_name: str = Field(
        description=(
            "The name of the note to read, written in the same way as notes are referenced in the"
            " vault. For example, to read the /AGENTS.md file use `note_name='AGENTS'`, to read"
            " the daily note for 2025-09-11 use `note_name='2025-09-11'`, meaning no actual paths"
            " (just the note name) and without the `.md` extension."
        ),
    )


class ListDir(BaseModel):
    """
    Use this function to list the contents of a directory in the vault. You will be returned a
    string with the contents of the directory, each separated by a newline. Directories will have
    a `/` at the end, regular files will not.
    """

    sub_path: str = Field(
        description=(
            "The path of the directory to list, considering that the vault path will already be"
            " prepended. So for example if you want to list the contents of the root, use `.`"
            " and if you want to list the contents of a `folder` at the root of the vault, use"
            " `folder`."
        ),
    )


class RipGrep(BaseModel):
    """
    Use this function to search the vault for a specific regex pattern using ripgrep. You will be
    return a string with the results of the search, including files with matches as well as a
    snippet of the matches. Only valid vault notes will be included in the search.

    Use `folder` to limit the search to a specific folder, leaving it blank to search the entire
    vault.

    Errors will be returned as "[system message: error message]". If there are too many results, a
    specific error will be returned.
    """

    pattern: str = Field(
        description="The regex pattern to search for inside the notes of the vault.",
    )

    folder: str | None = Field(
        description="The folder to limit the search to, leaving it blank to search the entire vault.",
    )

    case_sensitive: bool = Field(
        description="Whether to perform a case-sensitive search.",
    )


def create_read_note_tool(vault_path: str) -> Tool:
    callable = partial(read_note, vault_path)
    return Tool(spec=ReadNote, callable=callable)


def create_list_dir_tool(vault_path: str) -> Tool:
    IGNORED_ENTRIES = [".DS_Store", ".obsidian", ".trash"]

    def callable(sub_path: str) -> str:
        full_path = os.path.join(vault_path, sub_path)
        if not os.path.isdir(full_path):
            return f"[system message: directory '{sub_path}' does not exist]"

        contents = [
            entry.name + ("/" if entry.is_dir() else "")
            for entry in os.scandir(full_path)
            if entry.name not in IGNORED_ENTRIES
        ]
        return "\n".join(contents)

    return Tool(spec=ListDir, callable=callable)


def create_ripgrep_tool(vault_path: str) -> Tool:
    def callable(pattern: str, folder: str | None, case_sensitive: bool) -> str:
        full_path = os.path.join(vault_path, folder) if folder else vault_path
        if not os.path.isdir(full_path):
            return f"[system message: directory '{folder}' does not exist in the vault]"

        proc = subprocess.Popen(
            [
                "rg",
                "--json",
                "-g",
                "*.md",
                "-g",
                "!.trash/",
                "-g",
                "!.obsidian/",
                "--case-sensitive" if case_sensitive else "--ignore-case",
                pattern,
                full_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert proc.stdout is not None

        matches: dict[str, list[str]] = {}
        for line in proc.stdout:
            if not line:
                continue
            data = json.loads(line)

            if data["type"] == "match":
                note = os.path.basename(data["data"]["path"]["text"]).removesuffix(".md")
                matched_line = data["data"]["lines"]["text"].strip()

                if note not in matches:
                    matches[note] = []
                matches[note].append(matched_line)

        return_code = proc.wait()
        if return_code == 1:
            return "[system message: no matches found]"
        elif return_code != 0:
            stderr_output = proc.stderr.read().strip() if proc.stderr else "unknown error"
            return f"[system message: ripgrep failed: {stderr_output}]"

        result_lines: list[str] = []
        for note, lines in matches.items():
            result_lines.append(f"NOTE {note}")
            for line in lines:
                result_lines.append(f"LINE {line}")

        if len(result_lines) > 120:
            return "[system message: too many matches found, please narrow down the pattern]"

        return "\n".join(result_lines)

    return Tool(spec=RipGrep, callable=callable)
