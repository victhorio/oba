from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

from attrs import define, field
from httpx import AsyncClient
from pydantic import BaseModel

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS
from oba.ag.models.message import Content, Message, ModelID, ToolCall, Usage
from oba.ag.tool import Tool

ToolChoice = Literal["none", "auto", "required"]


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


@define(slots=True)
class Response(Generic[StructuredModelT]):
    """
    A normalized response object.
    """

    model: "Model"
    """The model used when making the request."""

    model_api_id: str
    """The model ID returned by the API. Usually includes a date-based variant specifier."""

    usage: Usage
    """Object including token usage information."""

    messages: list[Message]
    """The (potentially multiple) messages that make up this response."""

    structured_output: StructuredModelT | None
    """If this was a structured output request, this is the parsed PyDantic version of the content. Otherwise None."""

    _content_index: int | None = field(default=None, alias="_content_index")
    """The index of `messages` that includes a content. Use `.content` to access it directly."""

    _tool_call_indexes: list[int] = field(factory=list, alias="_tool_call_indexes")
    """The indexes of `messages` that include tool calls. Use `.tool_calls` to access it directly."""

    @property
    def content(self) -> Content | None:
        if not self._content_index:
            return None
        res = self.messages[self._content_index]
        return res  # pyright: ignore[reportReturnType]

    @property
    def tool_calls(self) -> list[ToolCall]:
        res = [self.messages[i] for i in self._tool_call_indexes]
        return res  # pyright: ignore[reportReturnType]

    @property
    def total_cost(self) -> float:
        """The total dollar cost of this response."""
        return self.model.total_cost(self.usage)


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
        client: AsyncClient,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool = False,
        timeout: int = 20,
        debug: bool = False,
    ) -> Response[StructuredModelT]: ...

    def total_cost(self, usage: Usage) -> float:
        assert self.model_id in _MODEL_COSTS, f"Unknown `model`: {self.model_id}"
        in_rate, cin_rate, out_rate = _MODEL_COSTS[self.model_id]
        in_cost = (usage.input_tokens - usage.input_tokens_cached) * in_rate
        cin_cost = usage.input_tokens_cached * cin_rate
        out_cost = usage.output_tokens * out_rate
        return (in_cost + cin_cost + out_cost) / 1e6


# TODO: add cache WRITE costs
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
}
