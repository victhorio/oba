import asyncio
import pprint

import httpx
from attrs import asdict

from oba.ag.models.anthropic import AnthropicModel
from oba.ag.models.message import Content, Message
from oba.ag.models.model import Response, StructuredModelT


async def main() -> float:
    async with httpx.AsyncClient() as c:
        costs = await asyncio.gather(
            test_regular_message(c),
            test_message_history(c),
        )
    return sum(costs)


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
    response = await model.generate(messages=messages, client=c)
    _show_response(response, "simple message")

    return response.total_cost


async def test_message_history(c: httpx.AsyncClient) -> float:
    messages: list[Message] = [
        Content(
            role="user",
            text="Hey! My name is Victhor, what's your name?",
        )
    ]

    model = AnthropicModel("claude-haiku-4-5", reasoning_effort=1024, max_output_tokens=2048)
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
    return response_a.total_cost + response_b.total_cost


def _show_response(response: Response[StructuredModelT], name: str) -> None:
    print(f"\033[33;1m--- test: {name} ---\033[0m")
    pprint.pp(asdict(response), width=110)

    if response.structured_output:
        print("\n\t\033[33mStructured output:\033[0m")
        pprint.pp(response.structured_output.model_dump(), width=110)
    print("\n\n", end="")
