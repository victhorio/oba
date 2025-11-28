import json
import os
from typing import Any, AsyncIterator, override

from httpx import AsyncClient

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
    async def stream(
        self,
        messages: list[Message],
        *,
        client: AsyncClient,
        max_output_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = "auto",
        parallel_tool_calls: bool | None = False,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> AsyncIterator[str | ToolCall | Response[Any]]:
        """
        Yields:
        - text deltas as strings as soon as they happen
        - tool calls as soon as they are completed
        - the final parsed response as the last item
        """

        max_output_tokens = max_output_tokens or self.max_output_tokens

        if isinstance(messages[0], Content) and messages[0].role == "system":
            system_prompt = messages[0].text
            messages = messages[1:]
        else:
            system_prompt = None

        payload: dict[str, object] = {
            "messages": [_transform_message_to_payload(m) for m in messages],
            "model": self.model_id,
            "max_tokens": max_output_tokens,
            "stream": True,
        }

        if self.reasoning_effort:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning_effort}
        else:
            payload["thinking"] = {"type": "disabled"}

        if tools:
            payload["tools"] = [_parse_tool(tool) for tool in tools]

            if tool_choice:
                tool_choice_payload: dict[str, object] = {"type": tool_choice}
                if tool_choice != "none" and not parallel_tool_calls:
                    tool_choice_payload["disable_parallel_tool_use"] = True
                payload["tool_choice"] = tool_choice_payload

            else:
                raise ValueError("tool_choice cannot be None for AnthropicModel")

        if system_prompt:
            payload["system"] = system_prompt

        if debug:
            import pprint

            print("--- Sent payload to Anthropic ---")
            pprint.pp(payload, width=110)
            print("---------------------------------")

        async with client.stream(
            "POST",
            _ANTHROPIC_GENERATION_URL,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
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

            response_json: dict[str, Any] = dict()

            async for line in response.aiter_lines():
                if not line or line.startswith("event:"):
                    # since we're guaranteed to have a single json data object with the event name
                    # in the "type" field, we can just ignore event lines
                    continue

                if debug:
                    print("-- line --")
                    print(line)
                    print("-- ---- --")

                if not line.startswith("data:"):
                    raise AssertionError("expected all lines to start with `data:` at this point")

                data = json.loads(line.replace("data:", ""))
                event_type = data["type"]

                if event_type == "message_start":
                    if response_json:
                        raise RuntimeError("multiple messages received")

                    response_json = data["message"]
                elif event_type == "content_block_start":
                    if data["index"] != len(response_json["content"]):
                        raise RuntimeError("content block index mismatch")

                    content_block = data["content_block"]

                    # for tool use we'll actually get a `partial_json` delta we need to manage
                    # so let's store it together with the response and we'll parse it at the coming
                    # content_block_stop
                    content_block["partial_json"] = ""

                    response_json["content"].append(content_block)
                elif event_type == "content_block_delta":
                    idx = data["index"]
                    delta = data["delta"]
                    delta_type = delta["type"]

                    if delta_type == "signature_delta":
                        response_json["content"][idx]["signature"] += delta["signature"]
                    elif delta_type == "thinking_delta":
                        response_json["content"][idx]["thinking"] += delta["thinking"]
                    elif delta_type == "text_delta":
                        text = delta["text"]
                        yield text
                        response_json["content"][idx]["text"] += text
                    elif delta_type == "input_json_delta":
                        response_json["content"][idx]["partial_json"] += delta["partial_json"]

                elif event_type == "content_block_stop":
                    # at content_block_stop, if the last content was a tool call we want to
                    # yield it right now, and we also need to parse the now complete `partial_json`
                    # into the `input` field
                    idx = data["index"]
                    if response_json["content"][idx]["type"] == "tool_use":
                        tool_call = response_json["content"][idx]
                        tool_call["input"] = json.loads(tool_call["partial_json"])
                        yield ToolCall(
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            parsed_args=tool_call["input"],
                        )

                elif event_type == "message_delta":
                    response_json |= data["delta"]
                    response_json["usage"] |= data["usage"]

                elif event_type == "message_stop":
                    yield self._parse_response(response_json)

    @override
    async def generate(
        self,
        messages: list[Message],
        *,
        client: AsyncClient,
        max_output_tokens: int | None = None,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = "auto",
        parallel_tool_calls: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> Response[StructuredModelT]:
        if structured_output:
            # TODO: under the beta header `structured-outputs-2025-11-13` it's supported now
            raise ValueError("anthropic does not support structured outputs")

        max_output_tokens = max_output_tokens or self.max_output_tokens

        if isinstance(messages[0], Content) and messages[0].role == "system":
            system_prompt = messages[0].text
            messages = messages[1:]
        else:
            system_prompt = None

        payload: dict[str, object] = {
            "messages": [_transform_message_to_payload(m) for m in messages],
            "model": self.model_id,
            "max_tokens": max_output_tokens,
        }

        if self.reasoning_effort:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning_effort}
        else:
            payload["thinking"] = {"type": "disabled"}

        if tools:
            payload["tools"] = [_parse_tool(tool) for tool in tools]

            if tool_choice:
                tool_choice_payload: dict[str, object] = {"type": tool_choice}
                if tool_choice != "none" and not parallel_tool_calls:
                    tool_choice_payload["disable_parallel_tool_use"] = True
                payload["tool_choice"] = tool_choice_payload

            else:
                raise ValueError("tool_choice cannot be None for AnthropicModel")

        if system_prompt:
            payload["system"] = system_prompt

        if debug:
            import pprint

            print("--- Sent payload to Anthropic ---")
            pprint.pp(payload, width=110)
            print("---------------------------------")

        response = await client.post(
            _ANTHROPIC_GENERATION_URL,
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
    ) -> Response[Any]:
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
    if payload := msg.payload_cache.get(_ANTHROPIC_PROVIDER_ID, dict()):
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
    msg.payload_cache[_ANTHROPIC_PROVIDER_ID] = payload
    return payload


def _parse_tool(tool: Tool) -> dict[str, object]:
    # TODO: let's have some caching not to reparse it everytime?
    spec_schema = tool.spec.model_json_schema()

    return {
        "name": spec_schema["title"],
        "description": spec_schema["description"],
        "input_schema": {
            "type": "object",
            "properties": spec_schema["properties"],
            "required": spec_schema["required"],
        },
    }


_ANTHROPIC_PROVIDER_ID = "anthropic"
_ANTHROPIC_GENERATION_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_MODEL_IDS: list[ModelID] = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]
