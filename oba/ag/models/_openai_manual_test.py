import asyncio
import pprint
from enum import StrEnum, auto

import httpx
from attrs import asdict
from pydantic import BaseModel, ConfigDict, Field

from oba.ag.models.openai import generate
from oba.ag.models.types import Message, Response, StructuredModelT


async def main():
    async with httpx.AsyncClient() as c:
        await asyncio.gather(
            test_regular_message(c),
            test_structured_output_strings(c),
            test_structured_output_complex(c),
        )


async def test_regular_message(c: httpx.AsyncClient):
    messages = [
        Message(
            role="system",
            content="Always include one emoji at the beginning of your response.",
        ),
        Message(
            role="user",
            content="Hey there! What's up?",
        ),
    ]

    response = await generate(c, input=messages, model="gpt-5-nano", reasoning_effort="minimal")
    show_result(response, "simple message")


async def test_structured_output_strings(c: httpx.AsyncClient):
    class CountryPick(BaseModel):
        reasoning: str = Field(description="A short opinionated reasoning behind the pick")
        country: str = Field(description="The name of the country")

    messages = [
        Message(
            role="user",
            content="Hey! If I had to pick a country to live in, which do you suggest? You HAVE to pick one.",
        ),
    ]

    response = await generate(
        c,
        input=messages,
        model="gpt-5-mini",
        reasoning_effort="minimal",
        structured_output=CountryPick,
    )
    show_result(response, "structured output: strings")

    # testing the typechecker
    country_pick: CountryPick
    assert response.structured_output is not None
    country_pick = response.structured_output
    _ = country_pick


async def test_structured_output_complex(c: httpx.AsyncClient):
    class Town(BaseModel):
        """
        Some basic information about a Town the NPC was born.
        """

        model_config = ConfigDict(extra="forbid")
        name: str = Field(description="A simple medieval name for the town")
        population: int = Field(
            description="The population of the town, <100 is small, <500 is regular, and biggest cities should be >1000"
        )

    class Race(StrEnum):
        """
        Race of a character
        """

        HUMAN = auto()
        ELF = auto()
        HALFLING = auto()

    class NPC(BaseModel):
        """
        Information about the NPC
        """

        model_config = ConfigDict(extra="forbid")
        name: str = Field(description="Age of the NPC")
        race: Race
        origin: Town
        age: int = Field(description="A race-appropriate age for the NPC")
        personality_traits: list[str] = Field(
            description="A couple of personality traits for the NPC"
        )

    class NPCBrainstorm(BaseModel):
        npcs: list[NPC] = Field(
            description="One NPC for each race, each from a different origin",
            min_length=3,
            max_length=3,
        )

    messages = [
        Message(
            role="user",
            content="Hey! I need your help coming up with ideas for 3 random NPCs, each from a difference race! Just go for it",
        )
    ]

    response = await generate(
        c,
        input=messages,
        model="gpt-5.1",
        reasoning_effort="low",
        structured_output=NPCBrainstorm,
    )

    show_result(response, "structured output: complex")


def show_result(response: Response[StructuredModelT], name: str) -> None:
    print(f"\033[33;1m--- test: {name} ---\033[0m")
    pprint.pprint(asdict(response), width=110)

    if response.structured_output:
        print("\n\t\033[33mStructured output:\033[0m")
        pprint.pprint(response.structured_output.model_dump(), width=110)
    print("\n\n", end="")


if __name__ == "__main__":
    asyncio.run(main())
