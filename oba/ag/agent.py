import uuid

import httpx
from attrs import define

from oba.ag.common import Usage
from oba.ag.memory import Memory
from oba.ag.models.message import Content, Message, ToolCall, ToolResult
from oba.ag.models.model import Model
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

    async def run(
        self,
        input: str,
        *,
        model: Model | None = None,
        timeout_api: int = 60,
        tool_calls_safe: bool = True,
        tool_calls_max_turns: int = 3,
        tool_calls_included_in_content: bool = True,
        session_id: str | None = None,
        debug: bool = False,
    ) -> Response:
        model_ = model or self.model
        session_id_ = session_id or str(uuid.uuid4())

        messages_prefix: list[Message] = []
        if self.system_prompt:
            messages_prefix.append(self.system_prompt)
        if self.memory:
            messages_prefix.extend(self.memory.get_messages(session_id_))

        messages_new: list[Message] = [Content(role="user", text=input)]

        usage = Usage()
        full_text_response: list[str] = list()

        # since we want to allow tool_calls_max_turns, we give it 1 extra turn
        # at the end to write a response
        for turn in range(tool_calls_max_turns + 1):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"

            response = await model_.generate(
                client=self.client,
                messages=messages_prefix + messages_new,
                tools=self.tools,
                timeout=timeout_api,
                tool_choice=tool_choice,
                debug=debug,
            )

            usage = usage.acc(
                Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_cost=response.total_cost,
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

            tool_results = [
                self.tool_call(tc, return_error_strings=tool_calls_safe)
                for tc in response.tool_calls
            ]

            messages_new.extend(tool_results)

        if self.memory:
            self.memory.extend(
                session_id=session_id_,
                messages=messages_new,
                usage=usage,
            )

        return Response(
            session_id=session_id_,
            model_id=model_.model_id,
            usage=usage,
            content="\n\n".join(full_text_response),
        )

    def tool_call(
        self,
        tool_call: ToolCall,
        return_error_strings: bool,
    ) -> ToolResult:
        if tool_call.name not in self.callables:
            raise ValueError(f"TollCall for a non-registered tool '{tool_call.name}'")

        callable = self.callables[tool_call.name]
        try:
            output = callable(**tool_call.parsed_args)
        except Exception as exc:
            if return_error_strings:
                output = f"[Tool '{tool_call.name}' call failed: {exc.__class__.__name__} {exc}]"
            else:
                raise RuntimeError(f"Tool '{tool_call.name}' call failed") from exc

        return ToolResult(
            call_id=tool_call.call_id,
            result=output,
        )
