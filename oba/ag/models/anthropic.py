import os
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


class AnthropicModel(Model):
    def __init__(
        self,
        model_id: ModelID,
        reasoning_effort: int = 0,
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        if model_id not in _ANTHROPIC_MODEL_IDS:
            raise ValueError(
                f"received model_id `{model_id}`, but expected one of {_ANTHROPIC_MODEL_IDS}"
            )
        if reasoning_effort < 0:
            raise ValueError(f"received reasoning_effort `{reasoning_effort}`, expected >= 0")
        if max_output_tokens < 1:
            raise ValueError(f"received max_output_tokens `{max_output_tokens}`, expected >= 1")

        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set ANTHROPIC_API_KEY in env")

    @override
    async def generate(
        self,
        messages: list[Message],
        *,
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
            raise ValueError("anthropic does not support structured outputs")
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
            "messages": [_transform_message_to_payload(m) for m in messages],
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

        if not response.is_success:
            print("\033[31;1mERROR:\033[0m: API returned an error")
            try:
                import pprint

                pprint.pp(response.json(), width=110)
            except Exception:
                pass
            response.raise_for_status()

        return self._parse_response(response.json())

    def _parse_response(
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
        messages_content: Content | None = None
        messages_tool_calls: list[ToolCall] = list()

        for message in r["content"]:
            if message["type"] == "thinking":
                reasoning = Reasoning(
                    encrypted_content=message["signature"],
                    content=message["thinking"],
                )

                messages.append(reasoning)

            elif message["type"] == "text":
                # i only expect a single text message per response, so if we've already parsed
                # content into `messages_content` this is unexpected
                if messages_content is not None:
                    raise AssertionError("unexpectedly found response with multiple text entries")

                messages_content = Content(
                    role="assistant",
                    text=message["text"],
                )

                messages.append(messages_content)

            elif message["type"] == "tool_use":
                tool_call = ToolCall(
                    call_id=message["id"],
                    name=message["name"],
                    parsed_args=message["input"],
                )

                messages.append(tool_call)
                messages_tool_calls.append(tool_call)

            else:
                raise AssertionError(f"unexpected/unhandled message type in response: {message}")

        return Response(
            model=self,
            model_api_id=r["model"],
            usage=usage,
            messages=messages,
            content=messages_content,
            tool_calls=messages_tool_calls,
            structured_output=None,
        )


def _transform_message_to_payload(msg: Message) -> dict[str, object]:
    payload: dict[str, object]

    # make sure that we don't need to recompute the payload for this message
    if payload := msg._provider_payload_cache.get(_ANTHROPIC_PROVIDER_ID, dict()):
        return payload

    if isinstance(msg, Content):
        if msg.role == "system":
            raise ValueError("claude does not support mid history system messages")
        payload = {
            "role": msg.role,
            "content": msg.text,
        }
    elif isinstance(msg, Reasoning):
        payload = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "signature": msg.encrypted_content,
                    "thinking": msg.content,
                }
            ],
        }
    elif isinstance(msg, ToolCall):
        payload = {
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
    elif isinstance(msg, ToolResult):
        payload = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.call_id,
                    "content": msg.result,
                }
            ],
        }
    else:
        # this branch should be greyed out by the LSP due to exhaustive match
        raise ValueError(f"receive invalid message type: {type(msg)}")

    # remember to save the result before returning
    msg._provider_payload_cache[_ANTHROPIC_PROVIDER_ID] = payload
    return payload


_ANTHROPIC_PROVIDER_ID = "anthropic"
_ANTHROPIC_MODEL_IDS: list[ModelID] = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
]
