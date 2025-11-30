import os
from datetime import datetime
from typing import Literal

import httpx
from ag import Agent
from ag.memory import SQLiteMemory
from ag.models import AnthropicModel, OpenAIModel
from ag.tools import create_agentic_web_search_tool

from oba import prompts, vault
from oba.configs import Config, special_dir_path
from oba.tools.fs import create_list_dir_tool, create_read_note_tool, create_ripgrep_tool


def agent_create(
    config: Config,
    model_family: Literal["gpt", "claude"],
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
    else:
        # this line should be greyed out by lsp due to exhaustive match
        raise AssertionError(f"Unknown model family: {model_family}")

    memory = SQLiteMemory(db_path=os.path.join(special_dir_path(config), "memory.db"))

    return Agent(
        model=model,
        memory=memory,
        system_prompt=system_prompt,
        tools=[
            create_read_note_tool(config.vault_path),
            create_list_dir_tool(config.vault_path),
            create_ripgrep_tool(config.vault_path),
            create_agentic_web_search_tool(client),
        ],
        client=client,
    )
