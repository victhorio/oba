# pyright: reportAny=false, reportPrivateUsage=false

from textwrap import dedent
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from oba.ag.agent import Agent, Response
from oba.ag.history import HistoryDb
from oba.ag.tool import tool

from .mock_openai import MockAsyncOpenAI


class GetWeather(BaseModel):
    """
    Returns the weather for a given location.
    """

    # ☝ keep docstring as multiline specifically to test dedenting

    location: str = Field(description="The city and country, e.g. Rio de Janeiro, Brazil")
    unit: Literal["C", "F"] = Field(
        description="Whether to return in Celcius or Fahrenheit",
        default="C",
    )


def get_weather(location: str, unit: str = "C") -> str:
    return f"32°{unit} in {location}"


@pytest.mark.asyncio
async def test_simple_messaging(oai_client: MockAsyncOpenAI):
    oai_client.set_responses(
        oai_client.response_text("Hello, how can I help you?"),
    )

    agent = Agent(model_id="gpt-5-mini", client=oai_client)
    r = await agent.run("Hello!")

    assert isinstance(r, Response), "sanity check for response type"
    assert r.session_id, "should always create a session_id"
    assert r.usage.input_tokens, "should always have some usage"
    assert r.content == "Hello, how can I help you?"


@pytest.mark.asyncio
async def test_simple_tool_calling(oai_client: MockAsyncOpenAI):
    oai_client.set_responses(
        oai_client.response_tool_calls(
            [("GetWeather", {"location": "Rio de Janeiro, Brazil", "unit": "C"})]
        ),
        oai_client.response_text("The weather is 32°C in Rio de Janeiro, Brazil."),
    )

    agent = Agent(
        model_id="gpt-5-mini",
        client=oai_client,
        tools=[tool(GetWeather, get_weather)],
    )
    r = await agent.run(
        "What's the weather like in Rio de Janeiro, Brazil?",
        tool_calls_included_in_content=True,
    )

    assert r.content == (
        dedent(
            """
            [Tool call: GetWeather]

            The weather is 32°C in Rio de Janeiro, Brazil.
            """
        ).strip()
    )

    assert oai_client.responses.create.call_count == 2, "should have made two calls to OpenAI"

    # make sure the second call included the tool call and tool result
    first_call, second_call = oai_client.responses.create.mock_calls
    assert len(first_call.kwargs["input"]) == 1  # only user message
    assert len(second_call.kwargs["input"]) == 3  # user, tool call, tool result
    assert second_call.kwargs["input"][-1]["output"] == get_weather("Rio de Janeiro, Brazil", "C")

    # make sure we're preserving the call_id
    assert second_call.kwargs["input"][-2]["call_id"] == second_call.kwargs["input"][-1]["call_id"]


@pytest.mark.asyncio
async def test_tool_calling_max_turns(oai_client: MockAsyncOpenAI):
    # we'll set up three tool calls but set the limit to 2 and make sure it only got called twice
    # and that our return is just the tool call logs
    oai_client.set_responses(
        oai_client.response_tool_calls(
            [("GetWeather", {"location": "Rio de Janeiro, Brazil", "unit": "C"})]
        ),
        oai_client.response_tool_calls(
            [("GetWeather", {"location": "Amsterdam, Netherlands", "unit": "C"})]
        ),
        oai_client.response_tool_calls(
            [("GetWeather", {"location": "Paris, France", "unit": "C"})]
        ),
    )

    agent = Agent(
        model_id="gpt-5-mini",
        client=oai_client,
        tools=[tool(GetWeather, get_weather)],
    )
    r = await agent.run(
        "What's the weather like in Rio de Janeiro, Amsterdam and Paris?",
        tool_calls_included_in_content=False,
        tool_calls_max_turns=2,
    )

    assert r.content == ""
    assert oai_client.responses.create.call_count == 2

    # note; in practice the last return would not be a tool call because we forced tool_choice="none"
    last_call = oai_client.responses.create.mock_calls[-1]
    assert last_call.kwargs["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_conversation_history_and_system_prompts(oai_client: MockAsyncOpenAI):
    oai_client.set_responses(
        oai_client.response_text("Hello, how can I help you?"),
        oai_client.response_tool_calls(
            [("GetWeather", {"location": "Rio de Janeiro, Brazil", "unit": "C"})]
        ),
        oai_client.response_text("The weather is 32°C in Rio de Janeiro, Brazil."),
        oai_client.response_text("Of course! You're welcome!"),
    )

    db = HistoryDb()
    agent = Agent(
        model_id="gpt-5-mini",
        client=oai_client,
        tools=[tool(GetWeather, get_weather)],
        system_prompt="You are a helpful assistant.",
        history_db=db,
    )

    session_id = "test-session-1"
    _ = await agent.run("Hello!", session_id=session_id)
    u1 = db.get_usage(session_id)
    _ = await agent.run("What's the weather in Rio de Janeiro?", session_id=session_id)
    u2 = db.get_usage(session_id)
    _ = await agent.run("Thanks!", session_id=session_id)
    u3 = db.get_usage(session_id)

    assert len(db._db) == 1
    assert next(iter(db._db)) == session_id

    # Turn 1: 1 (user) + 1 (assistant)
    # Turn 2: 1 (user) + 3 (assistant: tool call + tool result + text)
    # Turn 3: 1 (user) + 1 (assistant)
    assert len(db.get_messages(session_id)) == 8

    # costs accumulate correctly
    assert u1.input_tokens < u2.input_tokens < u3.input_tokens
    assert u1.total_cost < u2.total_cost < u3.total_cost

    # user, tool result, user, user
    assert oai_client.responses.create.call_count == 4

    # make sure each turn we sent system message + full history
    c1, c2, c3, c4 = oai_client.responses.create.mock_calls
    assert len(c1.kwargs["input"]) == 2  # system, user
    assert len(c2.kwargs["input"]) == 4  # system, user, assistant, user
    assert len(c3.kwargs["input"]) == 6  # system, user, assistant, user, tool_call, tool_result
    assert (
        len(c4.kwargs["input"]) == 8
    )  # system, user, assistant, user, tool_call, tool_result, assistant, user
