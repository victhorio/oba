import asyncio
import uuid
from typing import Any, Callable

import httpx
from attrs import define

from oba.ag.common import Usage
from oba.ag.memory import Memory
from oba.ag.models import AnthropicModel, Content, Message, Model, OpenAIModel, ToolCall, ToolResult
from oba.ag.models import Response as ModelResponse
from oba.ag.tool import Tool, ToolCallable


@define
class Response:
    session_id: str
    model_id: str
    usage: Usage
    content: str


class Agent:
    def __init__(
        self,
        model: Model,
        memory: Memory | None = None,
        tools: list[Tool] | None = None,
        client: httpx.AsyncClient | None = None,
        system_prompt: str | None = None,
    ):
        self.model: Model = model
        self.memory: Memory | None = memory
        self.client: httpx.AsyncClient = client or httpx.AsyncClient()
        self.system_prompt: Content | None = None
        if system_prompt:
            self.system_prompt = Content(
                role="system",
                text=system_prompt,
            )

        # TODO: change this to simply be a list[type[BaseModel]] instead
        self.tools: list[Tool] = list()
        self.callables: dict[str, ToolCallable] = dict()

        if tools:
            self.tools = tools
            self.callables = {tool.spec.__name__: tool.callable for tool in tools}

    async def stream(
        self,
        input: str,
        callback: Callable[[str | ToolCall], None],
        *,
        model: Model | None = None,
        timeout_api: int = 60,
        tool_calls_safe: bool = True,
        tool_calls_parallel: bool = True,
        tool_calls_max_turns: int = 3,
        tool_calls_included_in_content: bool = True,
        session_id: str | None = None,
        debug: bool = False,
    ) -> Response:
        model = model or self.model
        # only OpenAI and Anthropic have the .stream method
        assert isinstance(model, (OpenAIModel, AnthropicModel))

        session_id = session_id or str(uuid.uuid4())

        messages_prefix: list[Message] = []
        if self.system_prompt:
            messages_prefix.append(self.system_prompt)
        if self.memory:
            messages_prefix.extend(self.memory.get_messages(session_id))

        messages_new: list[Message] = [Content(role="user", text=input)]

        usage = Usage()
        full_text_response: list[str] = list()

        # since we want to allow tool_calls_max_turns, we give it 1 extra turn
        # at the end to write a response
        for turn in range(tool_calls_max_turns + 1):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"

            async for part in model.stream(
                messages=messages_prefix + messages_new,
                client=self.client,
                tools=self.tools,
                tool_choice=tool_choice,
                parallel_tool_calls=tool_calls_parallel,
                timeout=timeout_api,
                debug=debug,
            ):
                if isinstance(part, (str, ToolCall)):
                    callback(part)
                elif isinstance(part, ModelResponse):
                    response = part
                    break
            else:
                raise AssertionError("did not receive a model response")

            usage = usage.acc(
                Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_cost=response.dollar_cost,
                )
            )

            messages_new.extend(response.messages)
            if response.content:
                full_text_response.append(response.content.text)

            if not response.tool_calls:
                break

            if tool_calls_included_in_content:
                for tc in response.tool_calls:
                    full_text_response.append(f"[Tool call: {tc.name}]")

            tool_results, tool_costs = await self.tool_calls(
                response.tool_calls,
                return_error_strings=tool_calls_safe,
            )

            usage.tool_costs += tool_costs
            messages_new.extend(tool_results)

        if self.memory:
            self.memory.extend(
                session_id=session_id,
                messages=messages_new,
                usage=usage,
            )

        return Response(
            session_id=session_id,
            model_id=model.model_id,
            usage=usage,
            content="\n\n".join(full_text_response),
        )

    async def run(
        self,
        input: str,
        *,
        model: Model | None = None,
        timeout_api: int = 60,
        tool_calls_safe: bool = True,
        tool_calls_max_turns: int = 3,
        tool_calls_parallel: bool = True,
        tool_calls_included_in_content: bool = True,
        session_id: str | None = None,
        debug: bool = False,
    ) -> Response:
        model = model or self.model
        session_id = session_id or str(uuid.uuid4())

        messages_prefix: list[Message] = []
        if self.system_prompt:
            messages_prefix.append(self.system_prompt)
        if self.memory:
            messages_prefix.extend(self.memory.get_messages(session_id))

        messages_new: list[Message] = [Content(role="user", text=input)]

        usage = Usage()
        full_text_response: list[str] = list()

        # since we want to allow tool_calls_max_turns, we give it 1 extra turn
        # at the end to write a response
        for turn in range(tool_calls_max_turns + 1):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"

            response: ModelResponse[Any] = await model.generate(
                client=self.client,
                messages=messages_prefix + messages_new,
                tools=self.tools,
                parallel_tool_calls=tool_calls_parallel,
                timeout=timeout_api,
                tool_choice=tool_choice,
                debug=debug,
            )

            usage = usage.acc(
                Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_cost=response.dollar_cost,
                )
            )

            messages_new.extend(response.messages)
            if response.content:
                full_text_response.append(response.content.text)

            if not response.tool_calls:
                break

            if tool_calls_included_in_content:
                for tc in response.tool_calls:
                    full_text_response.append(f"[Tool call: {tc.name}]")

            tool_results, tool_costs = await self.tool_calls(
                response.tool_calls,
                return_error_strings=tool_calls_safe,
            )

            usage.tool_costs += tool_costs
            messages_new.extend(tool_results)

        if self.memory:
            self.memory.extend(
                session_id=session_id,
                messages=messages_new,
                usage=usage,
            )

        return Response(
            session_id=session_id,
            model_id=model.model_id,
            usage=usage,
            content="\n\n".join(full_text_response),
        )

    async def tool_calls(
        self,
        tool_calls: list[ToolCall],
        return_error_strings: bool,
    ) -> tuple[list[ToolResult], float]:
        tool_results_full = await asyncio.gather(
            *[self.tool_call(tc, return_error_strings=return_error_strings) for tc in tool_calls]
        )

        tool_results: list[ToolResult] = list()
        tool_costs = 0.0
        for tool_result, tool_cost in tool_results_full:
            tool_results.append(tool_result)
            tool_costs += tool_cost

        return tool_results, tool_costs

    async def tool_call(
        self,
        tool_call: ToolCall,
        return_error_strings: bool,
    ) -> tuple[ToolResult, float]:
        name = tool_call.name
        assert name in self.callables, f"TollCall for a non-registered tool '{name}'"

        callable = self.callables[name]
        try:
            output = await callable(**tool_call.parsed_args)
        except Exception as exc:
            if return_error_strings:
                output = f"[Tool '{name}' call failed: {exc.__class__.__name__} {exc}]"
            else:
                raise RuntimeError(f"Tool '{name}' call failed") from exc

        if isinstance(output, tuple):
            assert len(output) == 2, f"'{name}' returned weirdly sized tuple: {output}"

            result, cost = output
            assert isinstance(result, str), f"'{name}' returned a non-string: {type(result)}"
            assert isinstance(cost, float), f"'{name}' returned a non-float: {type(cost)}"

            tool_result = ToolResult(call_id=tool_call.call_id, result=result)
            return tool_result, cost

        if isinstance(output, str):
            return ToolResult(call_id=tool_call.call_id, result=output), 0.0

        raise AssertionError(f"'{name}' returned a weird type: {type(output)}")
