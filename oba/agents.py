from datetime import date

from oba import prompts, vault
from oba.ag import Agent
from oba.ag.memory import EphemeralMemory
from oba.ag.models import OpenAIModel
from oba.configs import Config


def new(config: Config) -> Agent:
    recent_dailies = vault.get_recent_dailies(config.vault_path)
    agents_md = vault.get_agents_md(config.vault_path)

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
    )
