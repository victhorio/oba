from datetime import date
from functools import partial

from pydantic import BaseModel, Field

from oba import prompts, vault
from oba.ag import Agent, Tool
from oba.ag.memory import EphemeralMemory
from oba.ag.models import OpenAIModel
from oba.configs import Config


def new(config: Config) -> Agent:
    recent_dailies = vault.get_recent_dailies(config.vault_path)

    try:
        agents_md = vault.read_note(config.vault_path, "AGENTS")
    except FileNotFoundError:
        agents_md = "[system: no AGENTS.md file found in repository]"

    system_prompt = prompts.load(
        prompt_name="system_prompt",
        name=config.user_name,
        today=date.today().isoformat(),
        agents_md=agents_md,
        recent_dailies=recent_dailies,
    )

    model = OpenAIModel(
        model_id=config.model_id,
        reasoning_effort="low",
    )

    return Agent(
        model=model,
        memory=EphemeralMemory(),
        system_prompt=system_prompt,
        tools=[
            get_read_note_tool(config.vault_path),
        ],
    )


class ReadNote(BaseModel):
    """
    Use this function to read a note from the vault. If the note exists, it will be returned as a
    string (with nothing else), and if the note doesn't exist, it will return the following string:
    `[note {note_name} does not exist]`.
    """

    note_name: str = Field(
        ...,
        description=(
            "The name of the note to read, written in the same way as notes are referenced in the"
            " vault. For example, to read the /AGENTS.md file use `note_name='AGENTS'`, to read"
            " the daily note for 2025-09-11 use `note_name='2025-09-11'`, meaning no actual paths"
            " (just the note name) and without the `.md` extension."
        ),
    )


def get_read_note_tool(vault_path: str) -> Tool:
    callable = partial(vault.read_note, vault_path)
    return Tool(spec=ReadNote, callable=callable)
