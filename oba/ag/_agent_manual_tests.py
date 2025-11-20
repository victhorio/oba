import asyncio
import pprint
import random
import tempfile
from typing import Literal, Sequence
from uuid import uuid4

import httpx
from attrs import asdict
from pydantic import BaseModel, Field

import oba.ag._manual_test_utils as test_utils
from oba.ag import Tool
from oba.ag.agent import Agent, Response
from oba.ag.memory import EphemeralMemory, SQLiteMemory
from oba.ag.models.anthropic import AnthropicModel
from oba.ag.models.completions import CompletionsModel
from oba.ag.models.message import Message, Reasoning
from oba.ag.models.openai import OpenAIModel


async def run_manual_tests() -> None:
    test_utils.print_header("Agent")

    async with httpx.AsyncClient() as c:
        costs = await asyncio.gather(
            test_regular_message(c),
            test_message_history_openai(c),
            test_message_history_openai_sqlite(c),
            test_message_history_anthropic(c),
            test_message_history_gemini(c),
            test_single_turn_tool_calling(c),
            test_multi_turn_tool_calling(c),
            test_multi_turn_tool_calling_anthropic(c),
            test_multi_turn_tool_calling_gemini(c),
        )

    total_cost = sum(costs)
    test_utils.print_cost(total_cost)


async def test_regular_message(c: httpx.AsyncClient) -> float:
    model = OpenAIModel(model_id="gpt-5-nano")
    agent = Agent(model=model, client=c)
    response = await agent.run("Hey there! What's up?")
    _show_response(response, "simple message")
    return response.usage.total_cost


async def test_message_history_openai(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = OpenAIModel("gpt-5-mini")
    agent = Agent(model=model, client=c, memory=memory)
    session_id = str(uuid4())

    response_a = await agent.run(
        "Hey! My name is Victhor, what's your name?",
        session_id=session_id,
    )

    response_b = await agent.run(
        "Hey, can you remind me what's my name again?",
        session_id=session_id,
    )

    memory_messages = memory.get_messages(session_id)
    memory_usage = memory.get_usage(session_id)

    assert response_a.usage.total_cost + response_b.usage.total_cost == memory_usage.total_cost
    _show_memory(memory_messages, "message history: memory: openai")

    return memory_usage.total_cost


async def test_message_history_openai_sqlite(c: httpx.AsyncClient) -> float:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        memory_db_path = f.name
    memory = SQLiteMemory(memory_db_path)

    model = OpenAIModel("gpt-5-mini")
    agent = Agent(model=model, client=c, memory=memory)
    session_id = str(uuid4())

    response_a = await agent.run(
        "Hey! My name is Victhor, what's your name?",
        session_id=session_id,
    )

    memory.close()
    del memory
    del agent

    new_memory = SQLiteMemory(memory_db_path)
    new_agent = Agent(model=model, client=c, memory=new_memory)

    response_b = await new_agent.run(
        "Hey, can you remind me what's my name again?",
        session_id=session_id,
    )

    new_memory.close()
    del new_memory
    del new_agent

    after_memory = SQLiteMemory(memory_db_path)

    memory_messages = after_memory.get_messages(session_id)
    memory_usage = after_memory.get_usage(session_id)

    assert (
        abs(response_a.usage.total_cost + response_b.usage.total_cost - memory_usage.total_cost)
        < 1e-10
    )
    _show_memory(memory_messages, "message history: memory: openai + sqlite3")

    return memory_usage.total_cost


async def test_message_history_anthropic(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = AnthropicModel("claude-haiku-4-5", reasoning_effort=1024)
    agent = Agent(model=model, client=c, memory=memory)
    session_id = str(uuid4())

    response_a = await agent.run(
        "Hey! My name is Victhor, what's your name?",
        session_id=session_id,
    )

    response_b = await agent.run(
        "Hey, can you remind me what's my name again?",
        session_id=session_id,
    )

    memory_messages = memory.get_messages(session_id)
    memory_usage = memory.get_usage(session_id)

    assert response_a.usage.total_cost + response_b.usage.total_cost == memory_usage.total_cost
    _show_memory(memory_messages, "message history: memory: anthropic")

    return memory_usage.total_cost


async def test_message_history_gemini(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = CompletionsModel("gemini-2.5-flash", reasoning_effort="low")
    agent = Agent(model=model, client=c, memory=memory)
    session_id = str(uuid4())

    response_a = await agent.run(
        "Hey! My name is Victhor, what's your name?",
        session_id=session_id,
    )

    response_b = await agent.run(
        "Hey, can you remind me what's my name again?",
        session_id=session_id,
    )

    memory_messages = memory.get_messages(session_id)
    memory_usage = memory.get_usage(session_id)

    assert response_a.usage.total_cost + response_b.usage.total_cost == memory_usage.total_cost
    _show_memory(memory_messages, "message history: memory: gemini")

    return memory_usage.total_cost


async def test_single_turn_tool_calling(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = OpenAIModel("gpt-5-mini")
    agent = Agent(
        model=model,
        client=c,
        memory=memory,
        tools=[get_weather, search_wikipedia],
    )
    response = await agent.run("Qual a temperatura aqui no rio de janeiro hoje?")

    session_id = response.session_id
    assert session_id

    _show_memory(memory.get_messages(session_id), "simple tool calling: memory")

    return memory.get_usage(session_id).total_cost


async def test_multi_turn_tool_calling(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = OpenAIModel("gpt-5.1")
    agent = Agent(
        model=model,
        client=c,
        memory=memory,
        tools=[get_weather],
        system_prompt=(
            "If a user asks for temperature in multiple locations,"
            " calmly call the relevant tool for one place at a time."
        ),
    )
    response = await agent.run(
        "What's the weather like today in Rio de Janeiro, Amsterdam and New York? In Celcius.",
        tool_calls_max_turns=5,
    )

    _show_memory(memory.get_messages(response.session_id), "multi turn tool calling: memory")

    return response.usage.total_cost


async def test_multi_turn_tool_calling_anthropic(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = AnthropicModel("claude-sonnet-4-5", reasoning_effort=1_024)
    agent = Agent(
        model=model,
        client=c,
        memory=memory,
        tools=[get_weather],
        system_prompt=(
            "If a user asks for temperature in multiple locations,"
            " calmly call the relevant tool for one place at a time."
        ),
    )
    response = await agent.run(
        "What's the weather like today in Rio de Janeiro, Amsterdam and New York? In Celcius.",
        tool_calls_max_turns=5,
    )

    _show_memory(
        memory.get_messages(response.session_id), "multi turn tool calling: anthropic: memory"
    )

    return response.usage.total_cost


async def test_multi_turn_tool_calling_gemini(c: httpx.AsyncClient) -> float:
    memory = EphemeralMemory()
    model = CompletionsModel("gemini-2.5-flash", reasoning_effort="medium")
    agent = Agent(
        model=model,
        client=c,
        memory=memory,
        tools=[get_weather],
        system_prompt=(
            "If a user asks for temperature in multiple locations,"
            " calmly call the relevant tool for one place at a time."
        ),
    )
    response = await agent.run(
        "What's the weather like today in Rio de Janeiro, Amsterdam and New York? In Celcius.",
        tool_calls_max_turns=5,
    )

    _show_memory(
        memory.get_messages(response.session_id), "multi turn tool calling: gemini: memory"
    )

    return response.usage.total_cost


def _show_response(r: Response, title: str) -> None:
    test_utils.print_result_header(title)

    print(f"\tSession ID: {r.session_id}")
    print("\tUsage:")
    pprint.pp(asdict(r.usage), width=110)
    print("\tContent:")
    print(r.content)

    test_utils.print_result_footer()


def _show_memory(m: Sequence[Message], title: str) -> None:
    test_utils.print_result_header(title)

    mm = [
        x
        if not isinstance(x, Reasoning)
        else Reasoning(encrypted_content=x.encrypted_content[10:20] + "...", content=x.content)
        for x in m
    ]
    pprint.pp(mm, width=110)

    test_utils.print_result_footer()


# tool definitions


class GetWeather(BaseModel):
    """
    Returns the weather for the given location.
    """

    city: str = Field(description="Name of a city, followed by country. For example: New York, USA")
    unit: Literal["C", "F"] = Field(description="The unit of temperature")
    days_delta: int = Field(
        description="The `delta` of days relative to today. For example: 0 means today, 1 means tomorrow.",
        ge=-7,
        le=7,
    )


get_weather = Tool(GetWeather, lambda **kwargs: str(random.randint(20, 30)))


class SearchWikipedia(BaseModel):
    """
    Searches a query in Wikipedia, returning a list of links to relevant articles.
    """

    query: str = Field(description="The query to use in the search")


search_wikipedia = Tool(SearchWikipedia, lambda **kwargs: "ERROR: Bad connection")
