"""
Application settings and configuration helpers.
"""

from functools import lru_cache

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    # NOTE: GPT-5 family models don't support temperature/top_p parameters which is why
    #       they're not included in the settings. Source:
    #       https://platform.openai.com/docs/guides/latest-model#gpt-5-parameter-compatibility

    model_id: str
    max_history_turns: int = Field(ge=1)
    system_prompt: str


@lru_cache
def load() -> Settings:
    """Load and cache Settings."""

    # TODO: read system prompt from a file
    # TODO: read other settings from a ~/.config/oba/settings.json file

    return Settings(
        model_id="gpt-5-mini",
        max_history_turns=20,
        system_prompt="You are a helpful assistant. Your output is printed to a terminal, so don't use Markdown/prefer unicode characters for formatting.",
    )
