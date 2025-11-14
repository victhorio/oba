import json
from typing import Generic, Literal, TypeVar

from attr import field
from attrs import define
from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
ModelID = Literal[
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
]

# Model ID -> (Input cost, Ccahed input cost, Output cost) per 1M tokens
_MODEL_COSTS: dict[ModelID, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
}


@define(slots=True)
class Message:
    role: Role
    # TODO: support non text inputs as well
    content: str


@define(slots=True)
class ToolCall:
    call_id: str
    name: str
    args: str
    _parsed_args: dict[str, object] = field(init=False, factory=dict)

    @property
    def parsed_args(self) -> dict[str, object]:
        if self._parsed_args:
            return self._parsed_args

        self._parsed_args = json.loads(self.args)
        return self._parsed_args


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


@define(slots=True)
class Response(Generic[StructuredModelT]):
    model: ModelID
    model_api: str
    usage: Usage
    content: str
    tool_calls: list[ToolCall]
    structured_output: StructuredModelT | None
    raw_output: list[dict[str, object]]

    @property
    def message(self) -> Message:
        return Message(role="assistant", content=self.content)

    @property
    def total_cost(self) -> float:
        return self.usage.total_cost(self.model)


MessageTypes = Message | ToolCall | Response
