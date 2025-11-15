import uuid
from typing import Literal

import httpx
from pydantic import BaseModel

from oba.ag.common import Usage
from oba.ag.history import HistoryDb
from oba.ag.models.openai import generate
from oba.ag.models.types import Message, MessageTypes, ModelID, ToolCall, ToolResult
from oba.ag.tool import Tool, ToolCallable


class Response(BaseModel):
    session_id: str
    model_id: str
    usage: Usage
    content: str


class Agent:
    def __init__(
        self,
        model_id: ModelID,
        history_db: HistoryDb | None = None,
        tools: list[Tool] | None = None,
        client: httpx.AsyncClient | None = None,
        system_prompt: str | None = None,
    ):
        self.model_id: ModelID = model_id
        self.history_db: HistoryDb | None = history_db
        self.client: httpx.AsyncClient = client or httpx.AsyncClient()
        self.system_prompt: Message | None = None
        if system_prompt:
            self.system_prompt = Message(
                role="system",
                content=system_prompt,
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
        model_id: ModelID | None = None,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
        timeout_api: int = 30,
        tool_calls_safe: bool = True,
        tool_calls_max_turns: int = 3,
        tool_calls_included_in_content: bool = True,
        session_id: str | None = None,
        debug: bool = False,
    ) -> Response:
        model_id_ = model_id or self.model_id
        reasoning_effort_ = reasoning_effort or "medium"
        session_id_ = session_id or str(uuid.uuid4())

        messages_prefix: list[MessageTypes] = []
        if self.system_prompt:
            messages_prefix.append(self.system_prompt)
        if self.history_db:
            messages_prefix.extend(self.history_db.get_messages(session_id_))

        messages_new: list[MessageTypes] = [Message(role="user", content=input)]

        usage = Usage()
        full_text_response: list[str] = list()

        # since we want to allow tool_calls_max_turns, we give it 1 extra turn
        # at the end to write a response
        for turn in range(tool_calls_max_turns + 1):
            is_last_turn = turn == tool_calls_max_turns - 1
            tool_choice = "auto" if not is_last_turn else "none"

            response = await generate(
                client=self.client,
                messages=messages_prefix + messages_new,
                model=model_id_,
                reasoning_effort=reasoning_effort_,
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
                full_text_response.append(response.content)

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

        if self.history_db:
            self.history_db.extend(
                session_id=session_id_,
                messages=messages_new,
                usage=usage,
            )

        return Response(
            session_id=session_id_,
            model_id=model_id_,
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
