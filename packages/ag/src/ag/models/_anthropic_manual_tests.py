import asyncio
import pprint
from typing import Any, Literal

import httpx
from attrs import asdict
from pydantic import BaseModel, Field

import ag._manual_test_utils as test_utils
from ag.models.anthropic import AnthropicModel
from ag.models.message import Content, Message, ToolResult
from ag.models.model import Response, StructuredModelT
from ag.tool import Tool


async def run_manual_tests() -> None:
    test_utils.print_header("Anthropic Model")

    async with httpx.AsyncClient() as c:
        costs = await asyncio.gather(
            test_regular_message(c),
            test_message_history(c),
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

    model = AnthropicModel("claude-haiku-4-5")
    response: Response[Any] = await model.generate(messages=messages, client=c)
    _show_response(response, "simple message")

    return response.dollar_cost


async def test_message_history(c: httpx.AsyncClient) -> float:
    messages: list[Message] = [
        Content(
            role="user",
            text="Hey! My name is Victhor, what's your name?",
        )
    ]

    model = AnthropicModel("claude-haiku-4-5", reasoning_effort=1024, max_output_tokens=2048)
    response_a: Response[Any] = await model.generate(
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
    response_b: Response[Any] = await model.generate(
        messages=messages,
        client=c,
    )
    _show_response(response_b, "message history: second turn")
    return response_a.dollar_cost + response_b.dollar_cost


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
        Tool(GetWeather, lambda x: ""),  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
        Tool(SearchWikipedia, lambda x: ""),  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    ]

    total_cost = 0.0
    m: list[Message] = [
        Content(
            role="user",
            text="Qual a temperatura aqui no rio de janeiro hoje?",
        )
    ]

    model = AnthropicModel("claude-sonnet-4-5", reasoning_effort=1_024)
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

    response: Response[Any] = await model.generate(
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
