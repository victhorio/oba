import json
from typing import Generic, Literal, TypeVar

from attrs import define, field
from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
ModelID = Literal[
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
]


@define(slots=True)
class Reasoning:
    # NOTE: We need to store reasoning blocks from OpenAI to maintain it in context,
    #       but they only accept returning encrypted reasoning contents.
    encrypted_content: str
    content: str = field(default="")


@define(slots=True)
class Content:
    role: Role
    # TODO: support non text inputs as well
    text: str


@define(slots=True)
class ToolCall:
    call_id: str
    name: str
    args: str | None = field(default=None)
    parsed_args: dict[str, object] = field(factory=dict)

    def __attrs_post_init__(self):
        if not self.parsed_args and not self.args:
            raise ValueError("either parsed_args or args must be set")
        if self.args and not self.parsed_args:
            self.parsed_args = json.loads(self.args)


@define(slots=True)
class ToolResult:
    call_id: str
    result: str


@define(slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    input_tokens_cached: int = 0
    output_tokens_reasoning: int = 0

    def total_cost(self, model: ModelID) -> float:
        assert model in _MODEL_COSTS, f"Unknown `model`: {model}"
        in_rate, cin_rate, out_rate = _MODEL_COSTS[model]
        in_cost = (self.input_tokens - self.input_tokens_cached) * in_rate
        cin_cost = self.input_tokens_cached * cin_rate
        out_cost = self.output_tokens * out_rate
        return (in_cost + cin_cost + out_cost) / 1e6


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

Message = Content | Reasoning | ToolCall | ToolResult
"""The different message types that can be part of a conversation history: regular content, reasoning from models, tool calls and results."""


@define(slots=True)
class Response(Generic[StructuredModelT]):
    """
    A normalized response object.
    """

    model: ModelID
    """The model ID used when making the request."""

    model_api: str
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
        return self.usage.total_cost(self.model)


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
