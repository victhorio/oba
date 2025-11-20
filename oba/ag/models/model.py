from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from attrs import define
from httpx import AsyncClient
from pydantic import BaseModel

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS
from oba.ag.models.message import Content, Message, ModelID, ToolCall, Usage
from oba.ag.tool import Tool

ToolChoice = Literal["none", "auto", "required"]

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class Model(ABC):
    def __init__(
        self,
        model_id: ModelID,
    ):
        self.model_id: ModelID = model_id

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        client: AsyncClient,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool = False,
        timeout: int = 20,
        debug: bool = False,
    ) -> "Response[StructuredModelT]":
        """
        Generates a model response for the given `messages` history.

        Parameters:
            messages:
                The message history expecting model completions.
            client:
                An async httpx client used for the API calls.
            max_output_tokens:
                Optional. Specifies the maximum output tokens for the response. Overrides the
                default value set in model initialization for this specific generation.
            structured_output:
                Optional. A pydantic model specifying the schema for the structured output of the
                response. Cannot be used together with tool calling.
            tools:
                Optional. A list of tools for which the model may make calls. Note that this
                function will only return the tool calls from the model as part of the response and
                will not handle tool calling directly. See the `ag.Agent` abstraction for that.
            tool_choice:
                Optional. Can specify whether tool choices are disallowed/required.
            parallel_tool_calls:
                Defaults to false. Specify whether the model can generate multiple tool calls as
                part of a single response.
            timeout:
                Defaults to 20. Specifies the number of seconds allowed for the API call.
            debug:
                Defaults to false. If enabled, routine will print input/output payloads to stdout.

        Returns a `Response` object containing all relevant fields.
        """

        ...

    def dollar_cost(self, usage: Usage) -> float:
        """Returns the dollar cost of a call given token usage information."""

        assert self.model_id in _MODEL_COSTS, f"Unknown `model`: {self.model_id}"
        in_rate, cin_rate, out_rate = _MODEL_COSTS[self.model_id]
        in_cost = (usage.input_tokens - usage.input_tokens_cached) * in_rate
        cin_cost = usage.input_tokens_cached * cin_rate
        out_cost = usage.output_tokens * out_rate
        return (in_cost + cin_cost + out_cost) / 1e6


@define(slots=True)
class Response(Generic[StructuredModelT]):
    """
    A normalized Response object for Model's `generate` results, containing all relevant details
    of the model response.
    """

    model: Model
    """The model used when making the request."""

    model_api_id: str
    """The model ID returned by the API. Usually includes a date-based variant specifier."""

    usage: Usage
    """Object including token usage information."""

    # we include all the different components of the response (content, reasoning, tool calls) in
    # the exact order we received them since this will be reused repetedly throughout a given
    # conversation thread.
    messages: list[Message]
    """The (potentially multiple) messages that make up this response."""

    content: Content | None
    """The content object present in `messages`, if available."""

    structured_output: StructuredModelT | None
    """If this was a structured output request, this is the parsed PyDantic version of the content."""

    tool_calls: list[ToolCall]
    """The tool call entries present in `messages`. May be empty."""

    @property
    def dollar_cost(self) -> float:
        """The dollar cost of generating this response."""

        return self.model.dollar_cost(self.usage)


# TODO: add cache WRITE costs
# TODO: add variable cost when input tokens >= 200k (gemini/claude)
# Model ID -> (Input cost, Cached input cost, Output cost) per 1M tokens
_MODEL_COSTS: dict[ModelID, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "claude-haiku-4-5": (1.00, 0.100, 5.00),
    "claude-sonnet-4-5": (3.00, 0.300, 15.00),
    "claude-opus-4-1": (15.00, 1.500, 75.00),
    "gemini-2.5-flash": (0.30, 0.030, 1.00),
    "gemini-2.5-pro": (1.25, 0.125, 10.00),
    "gemini-3-pro-preview": (2.00, 0.020, 12.00),
}
