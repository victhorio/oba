import asyncio
import pprint
from enum import StrEnum, auto
from typing import Literal

import httpx
from attrs import asdict
from pydantic import BaseModel, ConfigDict, Field

import oba.ag._manual_test_utils as test_utils
from oba.ag import Tool
from oba.ag.models.message import Content, Message, ToolResult
from oba.ag.models.model import Response, StructuredModelT
from oba.ag.models.openai import OpenAIModel


async def run_manual_tests() -> None:
    test_utils.print_header("OpenAI Model")

    async with httpx.AsyncClient() as c:
        costs = await asyncio.gather(
            test_regular_message(c),
            test_message_history(c),
            test_structured_output_strings(c),
            test_structured_output_complex(c),
            test_tool_calling(c),
        )

    total_cost = sum(costs)
    test_utils.print_cost(total_cost)


async def test_regular_message(c: httpx.AsyncClient) -> float:
    messages: list[Message] = [
        Content(
            role="system",
            text="Always include one emoji at the beginning of your response.",
        ),
        Content(
            role="user",
            text="Hey there! What's up?",
        ),
    ]

    model = OpenAIModel("gpt-5-nano", reasoning_effort="low")
    response = await model.generate(messages=messages, client=c)
    _show_response(response, "simple message")
    return response.dollar_cost


async def test_message_history(c: httpx.AsyncClient) -> float:
    messages: list[Message] = [
        Content(
            role="user",
            text="Hey! My name is Victhor, what's your name?",
        )
    ]

    model = OpenAIModel("gpt-5-mini", reasoning_effort="medium")
    response_a = await model.generate(
        messages=messages,
        client=c,
    )
    _show_response(response_a, "message history: first turn")

    messages.extend(response_a.messages)
    messages.append(
        Content(
            role="user",
            text="Hey, can you remind me what's my name again?",
        )
    )
    response_b = await model.generate(
        messages=messages,
        client=c,
    )
    _show_response(response_b, "message history: second turn")
    return response_a.dollar_cost + response_b.dollar_cost


async def test_structured_output_strings(c: httpx.AsyncClient) -> float:
    class CountryPick(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reasoning: str = Field(description="A short opinionated reasoning behind the pick")
        country: str = Field(description="The name of the country")

    messages: list[Message] = [
        Content(
            role="user",
            text="Hey! If I had to pick a country to live in, which do you suggest? You HAVE to pick one.",
        ),
    ]

    model = OpenAIModel("gpt-5-mini", reasoning_effort="low")
    response = await model.generate(
        client=c,
        messages=messages,
        structured_output=CountryPick,
    )
    _show_response(response, "structured output: strings")

    # testing the typechecker
    country_pick: CountryPick
    assert response.structured_output is not None
    country_pick = response.structured_output
    _ = country_pick

    return response.dollar_cost


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

    messages: list[Message] = [
        Content(
            role="user",
            text="Hey! I need your help coming up with ideas for 3 random NPCs, each from a difference race! Just go for it",
        )
    ]

    model = OpenAIModel("gpt-5.1", reasoning_effort="low")
    response = await model.generate(
        client=c,
        messages=messages,
        structured_output=NPCBrainstorm,
    )

    _show_response(response, "structured output: complex")

    return response.dollar_cost


async def test_tool_calling(c: httpx.AsyncClient) -> float:
    class GetWeather(BaseModel):
        """
        Returns the weather for the given location.
        """

        city: str = Field(
            description="Name of a city, followed by country. For example: New York, USA"
        )
        unit: Literal["C", "F"] = Field(description="The unit of temperature")
        days_delta: int = Field(
            description="The `delta` of days relative to today. For example: 0 means today, 1 means tomorrow.",
            ge=-7,
            le=7,
        )

    class SearchWikipedia(BaseModel):
        """
        Searches a query in Wikipedia, returning a list of links to relevant articles.
        """

        query: str = Field(description="The query to use in the search")

    tool_deck = [
        Tool(GetWeather, lambda x: ""),
        Tool(SearchWikipedia, lambda x: ""),
    ]

    total_cost = 0.0
    m: list[Message] = [
        Content(
            role="user",
            text="Qual a temperatura aqui no rio de janeiro hoje?",
        )
    ]

    model = OpenAIModel("gpt-5-mini", reasoning_effort="low")
    response = await model.generate(
        client=c,
        messages=m,
        tools=tool_deck,
    )
    total_cost += response.dollar_cost
    _show_response(response, "tool calling: tool call")

    assert response.tool_calls
    assert response.tool_calls[0].name == "GetWeather"

    tr = ToolResult(call_id=response.tool_calls[0].call_id, result="32")
    m.extend(response.messages)
    m.append(tr)

    response = await model.generate(
        client=c,
        messages=m,
        tools=tool_deck,
    )
    total_cost += response.dollar_cost
    _show_response(response, "tool calling: tool result")

    return total_cost


def _show_response(response: Response[StructuredModelT], title: str) -> None:
    test_utils.print_result_header(title)
    pprint.pp(asdict(response), width=110)

    if response.structured_output:
        print("\n\t\033[33mStructured output:\033[0m")
        pprint.pp(response.structured_output.model_dump(), width=110)

    test_utils.print_result_footer()
