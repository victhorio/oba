import os
import warnings
from typing import Any, override

from httpx import AsyncClient

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT
from oba.ag.models.message import (
    Content,
    Message,
    ModelID,
    Reasoning,
    ToolCall,
    ToolResult,
    Usage,
)
from oba.ag.models.model import Model, Response, StructuredModelT, ToolChoice
from oba.ag.tool import Tool

_PROVIDER_KEY = "ant"


class AnthropicModel(Model):
    def __init__(
        self,
        model_id: ModelID,
        reasoning_effort: int = 0,
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        super().__init__(model_id)
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set ANTHROPIC_API_KEY in env")

    @override
    async def generate(
        self,
        messages: list[Message],
        client: AsyncClient,
        max_output_tokens: int | None = None,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> Response[StructuredModelT]:
        if structured_output:
            # TODO: under the beta header `structured-outputs-2025-11-13` it's supported now
            raise NotImplementedError("anthropic does not support structured outputs")
        if tools:
            # TODO
            raise NotImplementedError("not implemented yet")

        max_output_tokens = max_output_tokens or self.max_output_tokens

        if isinstance(messages[0], Content) and messages[0].role == "system":
            system_prompt = messages[0].text
            messages = messages[1:]
        else:
            system_prompt = None

        payload = {
            "messages": [_transform_input(m) for m in messages],
            "model": self.model_id,
            "max_tokens": max_output_tokens,
        }

        if self.reasoning_effort:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning_effort}
        else:
            payload["thinking"] = {"type": "disabled"}

        if system_prompt:
            payload["system"] = system_prompt

        if debug:
            import pprint

            print("--- Sent payload to Anthropic ---")
            pprint.pp(payload, width=110)
            print("---------------------------------")

        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=timeout,
        )

        if debug:
            import pprint

            print("--- Returned payload from Anthropic ---")
            pprint.pp(response.json(), width=110)
            print("---------------------------------")

        response.raise_for_status()

        return self._normalize_response(
            response.json(),
        )

    def _normalize_response(
        self,
        r: dict[str, Any],
    ) -> Response:
        required_keys = ("model", "content", "usage")
        for key in required_keys:
            if key not in r:
                raise ValueError(f"response does not contain a `{key}` key")

        usage_raw = r["usage"]
        # TODO: add input tokens cache /writes/
        usage = Usage(
            input_tokens=usage_raw["input_tokens"],
            output_tokens=usage_raw["output_tokens"],
            input_tokens_cached=usage_raw["cache_read_input_tokens"],
            # not available from anthropic API
            output_tokens_reasoning=-1,
        )

        messages: list[Message] = list()
        content_idxs: list[int] = list()
        tool_call_idxs: list[int] = list()

        for message in r["content"]:
            if message["type"] == "thinking":
                reasoning = Reasoning(
                    encrypted_content=message["signature"],
                    content=message["thinking"],
                )
                messages.append(reasoning)
            elif message["type"] == "text":
                content = Content(
                    role="assistant",
                    text=message["text"],
                )
                content_idxs.append(len(messages))
                messages.append(content)
            elif message["type"] == "tool_use":
                tool_call = ToolCall(
                    call_id=message["id"],
                    name=message["name"],
                    parsed_args=message["input"],
                )
                tool_call_idxs.append(len(messages))
                messages.append(tool_call)
            else:
                warnings.warn(
                    f"While parsing Anthropic response found unhandled type: {message['type']}"
                )

        if len(content_idxs) > 1:
            raise AssertionError(
                "Anthropic API returning more than one content per response, restructure Response"
            )

        return Response(
            model=self,
            model_api_id=r["model"],
            usage=usage,
            messages=messages,
            structured_output=None,
            _content_index=content_idxs[0] if content_idxs else None,
            _tool_call_indexes=tool_call_idxs,
        )


def _transform_input(msg: Message) -> dict[str, object]:
    if parsed := msg._cached_parse.get(_PROVIDER_KEY, dict()):
        return parsed

    if isinstance(msg, Content):
        if msg.role == "system":
            raise ValueError("claude does not support mid history system messages")
        parsed: dict[str, object] = {
            "role": msg.role,
            "content": msg.text,
        }
        msg._cached_parse[_PROVIDER_KEY] = parsed
        return parsed

    if isinstance(msg, Reasoning):
        parsed = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "signature": msg.encrypted_content,
                    "thinking": msg.content,
                }
            ],
        }
        msg._cached_parse[_PROVIDER_KEY] = parsed
        return parsed

    if isinstance(msg, ToolCall):
        parsed = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": msg.call_id,
                    "name": msg.name,
                    "input": msg.parsed_args,
                }
            ],
        }
        msg._cached_parse[_PROVIDER_KEY] = parsed
        return parsed

    if isinstance(msg, ToolResult):
        parsed = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.call_id,
                    "content": msg.result,
                }
            ],
        }
        msg._cached_parse[_PROVIDER_KEY] = parsed
        return parsed

    # note: lsp should report unreachable code below, greyed out
    raise ValueError(f"receive invalid message type: {type(msg)}")
