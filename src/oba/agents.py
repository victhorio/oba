import os
from datetime import datetime
from functools import partial
from typing import Literal

import httpx
from ag import Agent, Tool
from ag.memory import EphemeralMemory
from ag.models import AnthropicModel, CompletionsModel, OpenAIModel
from ag.tools import create_agentic_web_search_tool
from pydantic import BaseModel, Field

from oba import prompts, vault
from oba.configs import Config


def new(
    config: Config,
    model_family: Literal["gpt", "gemini", "claude"],
    client: httpx.AsyncClient,
) -> Agent:
    recent_dailies = vault.get_recent_dailies(config.vault_path)

    try:
        agents_md = vault.read_note(config.vault_path, "AGENTS")
    except FileNotFoundError:
        agents_md = "[system: no AGENTS.md file found in repository]"

    system_prompt = prompts.load(
        prompt_name="system_prompt",
        name=config.user_name,
        now=datetime.now().strftime("%Y-%m-%d %H:%M (%A)"),
        agents_md=agents_md,
        recent_dailies=recent_dailies,
    )

    if model_family == "gpt":
        model = OpenAIModel(
            model_id="gpt-5.1",
            reasoning_effort="medium",
        )
    elif model_family == "claude":
        model = AnthropicModel(
            model_id="claude-sonnet-4-5",
            reasoning_effort=2_048,
        )
    elif model_family == "gemini":
        model = CompletionsModel(
            model_id="gemini-3-pro-preview",
            reasoning_effort="low",
        )
    else:
        # this line should be greyed out by lsp due to exhaustive match
        raise AssertionError(f"Unknown model family: {model_family}")

    return Agent(
        model=model,
        memory=EphemeralMemory(),
        system_prompt=system_prompt,
        tools=[
            create_read_note_tool(config.vault_path),
            create_list_dir_tool(config.vault_path),
            create_agentic_web_search_tool(client),
        ],
        client=client,
    )


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


def create_read_note_tool(vault_path: str) -> Tool:
    callable = partial(vault.read_note, vault_path)
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
