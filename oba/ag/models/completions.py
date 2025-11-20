import os
from typing import Any, Literal, override

from httpx import AsyncClient

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT
from oba.ag.models.message import Content, Message, ModelID, Reasoning, ToolCall, ToolResult, Usage
from oba.ag.models.model import Model, Response, StructuredModelT, ToolChoice
from oba.ag.tool import Tool

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class CompletionsModel(Model):
    """
    Model abstraction for providers that support the OpenAI completions API format.
    Defaults to Gemini's base URL and environment API key.
    """

    def __init__(
        self,
        model_id: ModelID,
        base_url: str = GEMINI_BASE_URL,
        api_key: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        if reasoning_effort not in ["low", "medium", "high"]:
            raise ValueError(f"received invalid reasoning effort: `{reasoning_effort}`")
        if max_output_tokens < 1:
            raise ValueError(f"received max_output_tokens `{max_output_tokens}`, expected >= 1")

        self.model_id = model_id
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set GEMINI_API_KEY in env")

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
            # TODO
            raise NotImplementedError("not implemented yet")

        max_output_tokens = max_output_tokens or self.max_output_tokens

        payload = {
            "messages": [self._transform_message_to_payload(m) for m in messages],
            "model": self.model_id,
            "max_completion_tokens": max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

        if tools:
            payload["tools"] = [_parse_tool(tool) for tool in tools]
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if debug:
            import pprint

            print(f"--- Sent payload to {self.base_url} ---")
            pprint.pp(payload)
            print("----------------------------------------")

        response = await client.post(
            f"{self.base_url}chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=timeout,
        )

        if debug:
            import pprint

            print(f"--- Received response from {self.base_url} ---")
            pprint.pp(response.json())
            print("----------------------------------------")

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
        required_keys = ("model", "choices", "usage")
        for key in required_keys:
            if key not in r:
                raise ValueError(f"missing required key '{key}' in response")

        if len(r["choices"]) != 1:
            raise RuntimeError(f"expected 1 choice, got {len(r['choices'])}")

        usage_raw = r["usage"]
        usage = Usage(
            input_tokens=usage_raw["prompt_tokens"],
            output_tokens=usage_raw["completion_tokens"],
            input_tokens_cached=0,
            output_tokens_reasoning=0,
        )

        message_raw = r["choices"][0]["message"]

        messages: list[Message] = list()
        messages_content: Content | None = None
        messages_tool_calls: list[ToolCall] = list()

        if content := message_raw.get("content"):
            messages_content = Content(
                role="assistant",
                text=content,
            )
            messages.append(messages_content)

        if tool_calls := message_raw.get("tool_calls"):
            for tool_call in tool_calls:
                tool_call = ToolCall(
                    call_id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    args=tool_call["function"]["arguments"],
                )
                messages.append(tool_call)
                messages_tool_calls.append(tool_call)

        return Response(
            model=self,
            model_api_id=r["model"],
            usage=usage,
            messages=messages,
            content=messages_content,
            tool_calls=messages_tool_calls,
            structured_output=None,
        )

    def _transform_message_to_payload(self, msg: Message) -> dict[str, object]:
        payload: dict[str, object]

        # make sure that we don't need to recompute the payload for this message
        if payload := msg._provider_payload_cache.get(self.base_url, dict()):
            return payload

        if isinstance(msg, Content):
            payload = {
                "role": msg.role,
                "content": msg.text,
            }
        elif isinstance(msg, ToolCall):
            payload = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": msg.call_id,
                        "function": {
                            "name": msg.name,
                            "arguments": msg.args,
                        },
                    },
                ],
            }
        elif isinstance(msg, ToolResult):
            payload = {
                "role": "tool",
                "tool_call_id": msg.call_id,
                "content": msg.result,
            }
        elif isinstance(msg, Reasoning):
            # TODO: completion models need to skip reasoning messages
            # TODO: also fix this in openai/anthropic in that they can only accept reasoning messages from themselves
            raise NotImplementedError()
        else:
            # this branch should be greyed out by the LSP due to exhaustive match
            raise ValueError(f"received invalid message type: {type(msg)}")

        msg._provider_payload_cache[self.base_url] = payload
        return payload


def _parse_tool(tool: Tool) -> dict[str, object]:
    # TODO: let's have some caching not to reparse it everytime?
    spec_schema = tool.spec.model_json_schema()

    return {
        "function": {
            "name": spec_schema["title"],
            "description": spec_schema["description"],
            "parameters": {
                "type": "object",
                "properties": spec_schema["properties"],
                "required": spec_schema["required"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "type": "function",
    }
