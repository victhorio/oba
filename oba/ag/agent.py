import json
import warnings
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel
from openai import AsyncOpenAI, omit
from openai.types.responses import (
    ResponseOutputText,
    ResponseUsage,
    FunctionToolParam,
    EasyInputMessageParam,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseOutputRefusal,
    ResponseInputItemParam,
)
from typing import Literal

from oba.ag.history import HistoryDb
from oba.ag.tool import Tool, ToolCallable


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0

    def acc(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_cost=self.total_cost + other.total_cost,
        )


class Response(BaseModel):
    model_id: str
    usage: Usage
    content: str


class Agent:
    def __init__(
        self,
        model_id: str,
        history_db: HistoryDb | None = None,
        tools: list[Tool] | None = None,
        client: AsyncOpenAI | None = None,
    ):
        if history_db:
            raise NotImplementedError("history_db is not implemented yet")

        self.model_id: str = model_id
        self.history_db: HistoryDb | None = history_db
        self.client: AsyncOpenAI = client or AsyncOpenAI()

        # we save tools by name
        if tools:
            self.tool_specs: list[FunctionToolParam] = [tool.to_oai() for tool in tools]
            self.tool_callables: dict[str, ToolCallable] = {
                tool.spec.__name__: tool.callable for tool in tools
            }
        else:
            self.tool_specs = []
            self.tool_callables = {}

    async def run(
        self,
        input: str,
        *,
        model_id: str | None = None,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
        timeout_api: float = 30.0,
        tool_calls_safe: bool = True,
        tool_calls_max_turns: int = 3,
    ) -> Response:
        model_id_ = model_id or self.model_id
        reasoning_effort_ = reasoning_effort or "minimal"

        # todo: when implementing historydb, we need to prepend the history here
        messages: list[ResponseInputItemParam] = [self._to_message(input)]

        usage = Usage()
        text_response = ""

        for turn in range(tool_calls_max_turns):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"
            response = await self.client.responses.create(
                input=messages,
                model=model_id_,
                reasoning={"effort": reasoning_effort_},
                tools=self.tool_specs or omit,
                # fail fast if sending too many tokens by mistake, truncation should
                # be handled by ag
                truncation="disabled",
                timeout=timeout_api,
                store=False,
                tool_choice=tool_choice,
            )

            if usage_oai := response.usage:
                cost = self._cost_calculate(usage_oai, model_id_)
                usage = usage.acc(
                    Usage(
                        input_tokens=usage_oai.input_tokens,
                        output_tokens=usage_oai.output_tokens,
                        total_cost=cost,
                    )
                )

            tool_calls: list[ResponseFunctionToolCall] = list()
            for item in response.output:
                if isinstance(item, ResponseFunctionToolCall):
                    tool_calls.append(item)
                elif isinstance(item, ResponseOutputMessage):
                    for content in item.content:
                        if isinstance(content, ResponseOutputText):
                            text_response += content.text
                        elif isinstance(content, ResponseOutputRefusal):  # pyright: ignore[reportUnnecessaryIsInstance]
                            text_response += f"\n[Refusal: {content.refusal}]\n"
                        else:
                            raise AssertionError(
                                f"Content should be either text or refusal, got {type(content)}"
                            )

            if not tool_calls:
                break

            tool_results = [self.tool_call(tc, safe=tool_calls_safe) for tc in tool_calls]
            # let's be nice and actually convert the output to an input type here
            tool_call_params = [ResponseFunctionToolCallParam(tc) for tc in tool_calls]  # pyright: ignore[reportArgumentType]

            messages.extend(tool_call_params)
            messages.extend(tool_results)

        return Response(
            model_id=model_id_,
            usage=usage,
            content=text_response,
        )

    def tool_call(
        self,
        tool_call: ResponseFunctionToolCall,
        safe: bool,
    ) -> FunctionCallOutput:
        if tool_call.name not in self.tool_callables:
            raise ValueError(f"TollCall for a non-registered tool '{tool_call.name}'")

        callable = self.tool_callables[tool_call.name]
        params = json.loads(tool_call.arguments)  # pyright: ignore[reportAny]
        try:
            output = callable(**params)
        except Exception as exc:
            if safe:
                output = f"[Tool '{tool_call.name}' call failed: {exc.__class__.__name__} {exc}]"
            else:
                raise RuntimeError(f"Tool '{tool_call.name}' call failed") from exc

        return FunctionCallOutput(
            call_id=tool_call.call_id,
            output=output,
            type="function_call_output",
        )

    @staticmethod
    def _to_message(input: str) -> EasyInputMessageParam:
        return EasyInputMessageParam(
            type="message",
            role="user",
            content=input,
        )

    @staticmethod
    def _cost_calculate(usage: ResponseUsage, model_id: str) -> float:
        if model_id not in _MODEL_COSTS:
            warnings.warn(f"Unknown model '{model_id}' for cost calculation, defaulting to 0.0")
            return 0.0

        costs = _MODEL_COSTS[model_id]
        input_cost = (usage.input_tokens / 1e6) * costs["input"]
        output_cost = (usage.output_tokens / 1e6) * costs["output"]
        return input_cost + output_cost


_MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.4},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5": {"input": 1.25, "output": 10.0},
}
