import asyncio
import pprint
from enum import StrEnum, auto

import httpx
from attrs import asdict
from pydantic import BaseModel, ConfigDict, Field

from oba.ag.models.openai import generate
from oba.ag.models.types import Message, MessageTypes, Response, StructuredModelT


async def main() -> float:
    async with httpx.AsyncClient() as c:
        costs = await asyncio.gather(
            test_regular_message(c),
            test_message_history(c),
            test_structured_output_strings(c),
            test_structured_output_complex(c),
        )
        return sum(costs)


async def test_regular_message(c: httpx.AsyncClient) -> float:
    messages: list[MessageTypes] = [
        Message(
            role="system",
            content="Always include one emoji at the beginning of your response.",
        ),
        Message(
            role="user",
            content="Hey there! What's up?",
        ),
    ]

    response = await generate(c, messages=messages, model="gpt-5-nano", reasoning_effort="minimal")
    show_result(response, "simple message")
    return response.total_cost


async def test_message_history(c: httpx.AsyncClient) -> float:
    messages: list[MessageTypes] = [
        Message(
            role="user",
            content="Hey! My name is Victhor, what's your name?",
        )
    ]

    response_a = await generate(
        c,
        messages=messages,
        model="gpt-5-mini",
        reasoning_effort="medium",
    )
    show_result(response_a, "message history: first turn")

    messages.extend(response_a.messages)
    messages.append(
        Message(
            role="user",
            content="Hey, can you remind me what's my name again?",
        )
    )
    response_b = await generate(
        c,
        messages=messages,
        model="gpt-5-mini",
        reasoning_effort="minimal",
    )
    show_result(response_b, "message history: second turn")
    return response_a.total_cost + response_b.total_cost


async def test_structured_output_strings(c: httpx.AsyncClient) -> float:
    class CountryPick(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reasoning: str = Field(description="A short opinionated reasoning behind the pick")
        country: str = Field(description="The name of the country")

    messages: list[MessageTypes] = [
        Message(
            role="user",
            content="Hey! If I had to pick a country to live in, which do you suggest? You HAVE to pick one.",
        ),
    ]

    response = await generate(
        c,
        messages=messages,
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

    return response.total_cost


async def test_structured_output_complex(c: httpx.AsyncClient) -> float:
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
        model_config = ConfigDict(extra="forbid")
        npcs: list[NPC] = Field(
            description="One NPC for each race, each from a different origin",
            min_length=3,
            max_length=3,
        )

    messages: list[MessageTypes] = [
        Message(
            role="user",
            content="Hey! I need your help coming up with ideas for 3 random NPCs, each from a difference race! Just go for it",
        )
    ]

    response = await generate(
        c,
        messages=messages,
        model="gpt-5.1",
        reasoning_effort="low",
        structured_output=NPCBrainstorm,
    )

    show_result(response, "structured output: complex")

    return response.total_cost


def show_result(response: Response[StructuredModelT], name: str) -> None:
    print(f"\033[33;1m--- test: {name} ---\033[0m")
    pprint.pprint(asdict(response), width=110)

    if response.structured_output:
        print("\n\t\033[33mStructured output:\033[0m")
        pprint.pprint(response.structured_output.model_dump(), width=110)
    print("\n\n", end="")
