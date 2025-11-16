from datetime import date

from oba import prompts, vault
from oba.ag import Agent, Response
from oba.ag.memory import EphemeralMemory
from oba.ag.models import OpenAIModel
from oba.configs import Config


def new(config: Config) -> Agent:
    recent_dailies = vault.get_recent_dailies(config.vault_path)
    system_prompt = prompts.load(
        prompt_name="system_prompt",
        user=config.name,
        daily_notes=vault.format_notes(recent_dailies),
        today=date.today().isoformat(),
    )

    model = OpenAIModel(
        model_id=config.model_id,
        reasoning_effort="low",
    )

    return Agent(
        model=model,
        memory=EphemeralMemory(),
        system_prompt=system_prompt,
    )


async def send_message(agent: Agent, message: str, session_id: str) -> Response:
    # TODO: remove this function, redundant now kept during refactor to minimize breakage
    return await agent.run(input=message, session_id=session_id)
