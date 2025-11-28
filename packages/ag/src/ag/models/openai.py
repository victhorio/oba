import json
import os
from typing import Any, AsyncIterator, override

from httpx import AsyncClient
from typing_extensions import Literal

from ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT
from ag.models.message import (
    Content,
    Message,
    ModelID,
    Reasoning,
    ToolCall,
    ToolResult,
    Usage,
)
from ag.models.model import Model, Response, StructuredModelT, ToolChoice
from ag.tool import Tool


class OpenAIModel(Model):
    def __init__(
        self,
        model_id: ModelID,
        reasoning_effort: Literal["none", "low", "medium", "high"] = "low",
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        if model_id not in _OPENAI_MODEL_IDS:
            raise ValueError(
                f"received model_id `{model_id}`, but expected one of {_OPENAI_MODEL_IDS}"
            )
        if reasoning_effort not in ("none", "low", "medium", "high"):
            raise ValueError(f"received invalid reasoning_effort `{reasoning_effort}`")
        if max_output_tokens < 1:
            raise ValueError(f"received max_output_tokens `{max_output_tokens}`, expected >= 1")

        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set OPENAI_API_KEY in env")

    @override
    async def stream(
        self,
        messages: list[Message],
        *,
        client: AsyncClient,
        max_output_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool | None = False,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> AsyncIterator[str | ToolCall | Response[Any]]:
        max_output_tokens = max_output_tokens or self.max_output_tokens

        payload: dict[str, object] = {
            "input": [_transform_message_to_payload(m) for m in messages],
            "model": self.model_id,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            # NOTE: We do not want to rely on OpenAI for storing any messages. However, since
            #       we are not allowed to have the reasoning content, and OpenAI reasoning models
            #       keep their reasoning as part of their context, we need to ask for the API to
            #       include encrypted reasoning content in their response. This means we'll be
            #       able to store it and reuse it later.
            "store": False,
            "include": [
                "reasoning.encrypted_content",
            ],
            "stream": True,
        }

        if tools:
            payload["tools"] = [_parse_tool(tool) for tool in tools]
            payload["parallel_tool_calls"] = parallel_tool_calls
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if debug:
            import pprint

            print("--- Sent payload to OpenAI ---")
            pprint.pp(payload, width=110)
            print("------------------------------")

        async with client.stream(
            "POST",
            _OPENAI_GENERATION_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=timeout,
        ) as response:
            if not response.is_success:
                print("\033[31;1mERROR:\033[0m: API returned an error")
                try:
                    import pprint

                    pprint.pp(response.json(), width=110)
                except Exception:
                    pass
                response.raise_for_status()

            async for line in response.aiter_lines():
                if debug:
                    print("-- line --")
                    print(line)
                    print("-- ---- --")

                if not line or line.startswith("event:"):
                    continue

                if not line.startswith("data: "):
                    raise AssertionError("expected all lines to start with `data:` at this point")

                data = json.loads(line.replace("data: ", ""))
                event_type = data["type"]

                if event_type == "response.output_text.delta":
                    yield data["delta"]
                elif event_type == "response.output_item.done":
                    # we want to yield tool calls early so the caller can print them out
                    if data["item"]["type"] == "function_call":
                        yield ToolCall(
                            call_id=data["item"]["call_id"],
                            name=data["item"]["name"],
                            args=data["item"]["arguments"],
                        )
                elif event_type in _OPENAI_STREAM_ERROR_TYPES:
                    raise RuntimeError(_format_stream_error(event_type, data))
                elif event_type == "response.completed":
                    yield self._parse_response(data["response"], structure=None)

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
        max_output_tokens = max_output_tokens or self.max_output_tokens

        payload: dict[str, object] = {
            "input": [_transform_message_to_payload(m) for m in messages],
            "model": self.model_id,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            # NOTE: We do not want to rely on OpenAI for storing any messages. However, since
            #       we are not allowed to have the reasoning content, and OpenAI reasoning models
            #       keep their reasoning as part of their context, we need to ask for the API to
            #       include encrypted reasoning content in their response. This means we'll be
            #       able to store it and reuse it later.
            "store": False,
            "include": [
                "reasoning.encrypted_content",
            ],
        }

        if structured_output:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": structured_output.__name__,
                    "strict": True,
                    "schema": structured_output.model_json_schema(),
                }
            }

        if tools:
            payload["tools"] = [_parse_tool(tool) for tool in tools]
            payload["parallel_tool_calls"] = parallel_tool_calls
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if debug:
            import pprint

            print("--- Sent payload to OpenAI ---")
            pprint.pp(payload, width=110)
            print("------------------------------")

        response = await client.post(
            _OPENAI_GENERATION_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=timeout,
        )

        if debug:
            import pprint

            print("--- Returned payload from OpenAI ---")
            pprint.pp(response.json(), width=110)
            print("------------------------------------")

        if not response.is_success:
            print("\033[31;1mERROR:\033[0m: API returned an error")
            try:
                import pprint

                pprint.pp(response.json(), width=110)
            except Exception:
                pass
            response.raise_for_status()

        return self._parse_response(response.json(), structure=structured_output)

    def _parse_response(
        self,
        r: dict[str, Any],
        structure: type[StructuredModelT] | None,
    ) -> Response[StructuredModelT]:
        """
        Transforms the OpenAI returned payload into an ag normalized Response object.
        """

        required_keys = ("model", "output", "usage")
        for key in required_keys:
            if key not in r:
                raise AssertionError(f"response does not contain a `{key}` key")

        usage_raw = r["usage"]
        usage = Usage(
            input_tokens=usage_raw["input_tokens"],
            output_tokens=usage_raw["output_tokens"],
            input_tokens_cached=usage_raw["input_tokens_details"]["cached_tokens"],
            output_tokens_reasoning=usage_raw["output_tokens_details"]["reasoning_tokens"],
        )

        messages: list[Message] = list()
        messages_content: Content | None = None
        messages_tool_calls: list[ToolCall] = list()

        for output_item in r["output"]:
            if output_item["type"] == "message":
                # i only expect a single message per response, so if we've already parsed content
                # into `messages_content` this is unexpected
                if messages_content is not None:
                    raise AssertionError("unexpectedly found response with multiple messages")

                # messages will have a list of content items, but i'm only expecting them to have
                # a single item for us to parse
                content_list: list[dict[str, Any]] = output_item["content"]
                if len(content_list) > 1:
                    raise AssertionError("unexpectedly found message with multiple content blocks")

                # i'm further expecting this item to always have a valid/non-empty text field
                content_item = content_list[0]
                content_text = content_item.get("text", "")
                if not content_item.get("text", ""):
                    raise AssertionError("unexpectedly found message content without text")

                messages_content = Content(
                    role="assistant",
                    text=content_text,
                )
                messages.append(messages_content)

            elif output_item["type"] == "function_call":
                tool_call = ToolCall(
                    call_id=output_item["call_id"],
                    name=output_item["name"],
                    args=output_item["arguments"],
                )

                messages.append(tool_call)
                messages_tool_calls.append(tool_call)

            elif output_item["type"] == "reasoning":
                reasoning = Reasoning(encrypted_content=output_item["encrypted_content"])

                messages.append(reasoning)
            else:
                raise AssertionError(
                    f"unexpected/unhandled message type in response: {output_item}"
                )

        structured_output = None
        if structure:
            if not messages_content:
                raise AssertionError("a structured_output response generated no content")

            structured_output = structure.model_validate_json(messages_content.text)

        return Response(
            model=self,
            model_api_id=r["model"],
            usage=usage,
            messages=messages,
            content=messages_content,
            tool_calls=messages_tool_calls,
            structured_output=structured_output,
        )


def _parse_tool(tool: Tool) -> dict[str, object]:
    # TODO: let's have some caching not to reparse it everytime?
    spec_schema = tool.spec.model_json_schema()

    return {
        "name": spec_schema["title"],
        "description": spec_schema["description"],
        "parameters": {
            "type": "object",
            "properties": spec_schema["properties"],
            "required": spec_schema["required"],
            "additionalProperties": False,
        },
        "strict": True,
        "type": "function",
    }


def _format_stream_error(event_type: str, data: dict[str, Any]) -> str:
    assert event_type in _OPENAI_STREAM_ERROR_TYPES

    if event_type == "response.failed":
        error_message = data.get("error", {}).get("message", "ag: unknown error")
        return f"response.failed: {error_message}"
    if event_type == "response.incomplete":
        incomplete_details = data.get("incomplete_details", {}).get("reason", "ag: unknown reason")
        return f"response.incomplete: {incomplete_details}"
    if event_type == "response.refusal.done":
        refusal = data.get("refusal", "ag: unknown refusal")
        return f"response.refusal.done: {refusal}"
    if event_type == "error":
        message = data.get("message", "ag: unknown error")
        return f"error: {message}"

    assert False, "unreachable"


def _transform_message_to_payload(msg: Message) -> dict[str, object]:
    payload: dict[str, object]

    # make sure that we don't need to recompute the payload for this message
    if payload := msg.payload_cache.get(_OPENAI_PROVIDER_ID, dict()):
        return payload

    if isinstance(msg, Content):
        # we need explicit typing here to make it clear to the parser that we do not
        # need this to be inferred as dict[str, str]
        parsed: dict[str, object] = {
            "type": "message",
            "role": msg.role,
            "content": msg.text,
        }
    elif isinstance(msg, Reasoning):
        parsed = {
            "type": "reasoning",
            "encrypted_content": msg.encrypted_content,
            # we need to include the summary field with an empty list even when we're not using
            # reasoning summaries for API compability
            "summary": list(),
        }
    elif isinstance(msg, ToolCall):
        parsed = {
            "type": "function_call",
            "call_id": msg.call_id,
            "name": msg.name,
            "arguments": msg.args,
        }
    elif isinstance(msg, ToolResult):
        parsed = {
            "type": "function_call_output",
            "call_id": msg.call_id,
            "output": msg.result,
        }
    else:
        # this branch should be greyed out by the LSP due to exhaustive match
        raise ValueError(f"received invalid message type: {type(msg)}")

    msg.payload_cache[_OPENAI_PROVIDER_ID] = parsed
    return parsed


_OPENAI_PROVIDER_ID = "openai"
_OPENAI_GENERATION_URL = "https://api.openai.com/v1/responses"
_OPENAI_MODEL_IDS: list[ModelID] = [
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
]

_OPENAI_STREAM_ERROR_TYPES: tuple[str, ...] = (
    "response.failed",
    "response.incomplete",
    "response.refusal.done",
    "error",
)
