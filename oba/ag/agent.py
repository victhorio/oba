import json
import uuid
import warnings
from typing import Literal

from openai import AsyncOpenAI, omit
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel

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
    session_id: str
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
        system_prompt: str | None = None,
    ):
        self.model_id: str = model_id
        self.history_db: HistoryDb | None = history_db
        self.client: AsyncOpenAI = client or AsyncOpenAI()
        self.system_prompt: EasyInputMessageParam | None = (
            self._to_message(system_prompt, role="system") if system_prompt else None
        )

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
        tool_calls_included_in_content: bool = True,
        session_id: str | None = None,
    ) -> Response:
        model_id_ = model_id or self.model_id
        reasoning_effort_ = reasoning_effort or "minimal"
        session_id_ = session_id or str(uuid.uuid4())

        messages_prefix: list[ResponseInputItemParam] = []
        if self.system_prompt:
            messages_prefix.append(self.system_prompt)
        if self.history_db:
            messages_prefix.extend(self.history_db.get_history(session_id_))

        messages_new: list[ResponseInputItemParam] = [self._to_message(input, role="user")]

        usage = Usage()
        full_text_response: list[str] = list()

        for turn in range(tool_calls_max_turns):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"
            response = await self.client.responses.create(
                input=messages_prefix + messages_new,
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
                    if tool_calls_included_in_content:
                        full_text_response.append(f"[Tool call: {item.name}]")
                elif isinstance(item, ResponseOutputMessage):
                    item_text: list[str] = []
                    for content in item.content:
                        if isinstance(content, ResponseOutputText):
                            item_text.append(content.text)
                        elif isinstance(content, ResponseOutputRefusal):  # pyright: ignore[reportUnnecessaryIsInstance]
                            item_text.append(f"\n[Refusal: {content.refusal}]\n")
                        else:
                            raise AssertionError(
                                f"Content should be either text or refusal, got {type(content)}"
                            )
                    item_text_str = "".join(item_text)
                    full_text_response.append(item_text_str)
                    messages_new.append(self._to_message(item_text_str, role="assistant"))

            if not tool_calls:
                break

            tool_results = [self.tool_call(tc, safe=tool_calls_safe) for tc in tool_calls]
            # let's be nice and actually convert the output to an input type here
            tool_call_params = [ResponseFunctionToolCallParam(tc) for tc in tool_calls]  # pyright: ignore[reportArgumentType]

            messages_new.extend(tool_call_params)
            messages_new.extend(tool_results)

        if self.history_db:
            self.history_db.extend_history(session_id_, messages_new)

        return Response(
            session_id=session_id_,
            model_id=model_id_,
            usage=usage,
            content="\n\n".join(full_text_response),
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
    def _to_message(
        input: str,
        role: Literal["system", "user", "assistant"] = "user",
    ) -> EasyInputMessageParam:
        return EasyInputMessageParam(
            type="message",
            role=role,
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
