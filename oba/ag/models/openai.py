import os
import warnings
from typing import Any, override

from httpx import AsyncClient
from typing_extensions import Literal

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TIMEOUT
from oba.ag.models.model import Model, ToolChoice
from oba.ag.models.types import (
    Message,
    MessageTypes,
    ModelID,
    Reasoning,
    Response,
    StructuredModelT,
    ToolCall,
    ToolResult,
    Usage,
)
from oba.ag.tool import Tool

ReasoningEffort = Literal["none", "low", "medium", "high"]


class OpenAIModel(Model):
    def __init__(
        self,
        model_id: ModelID,
        reasoning_effort: ReasoningEffort = "low",
        api_key: str | None = None,
    ):
        super().__init__(model_id)
        self.reasoning_effort: ReasoningEffort | None = reasoning_effort
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set OPENAI_API_KEY in env")

    @override
    async def generate(
        self,
        messages: list[MessageTypes],
        client: AsyncClient,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        debug: bool = False,
    ) -> Response[StructuredModelT]:
        payload = {
            "input": [_transform_input(m) for m in messages],
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
            "https://api.openai.com/v1/responses",
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

        response.raise_for_status()

        return _normalize_response(
            response.json(),
            model=self.model_id,
            structure=structured_output,
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


def _transform_input(msg: MessageTypes) -> dict[str, object]:
    """
    Transforms an ag normalized instance of MessageTypes into an OpenAI compatible payload.
    """

    # TODO: some caching not to reparse same items everytime?

    # NOTE: usually these payloads will have an `id`, and when using store=True we can
    #       actually fully rely on it (specially for reasoning), but since we want a
    #       stateless API we set store=False and therefore can discard all IDs

    if isinstance(msg, Message):
        return {
            "type": "message",
            "role": msg.role,
            "content": msg.content,
        }

    if isinstance(msg, Reasoning):
        return {
            "type": "reasoning",
            "encrypted_content": msg.encrypted_content,
            # NOTE: even when not using reasoning summaries, the API will still require
            #       the summary field with an empty list for validation
            "summary": list(),
        }

    if isinstance(msg, ToolCall):
        return {
            "type": "function_call",
            "call_id": msg.call_id,
            "name": msg.name,
            "arguments": msg.args,
        }

    if isinstance(msg, ToolResult):
        return {
            "type": "function_call_output",
            "call_id": msg.call_id,
            "output": msg.result,
        }

    # note: lsp should report unreachable code below, greyed out
    raise ValueError(f"received invalid message type: {type(msg)}")


def _normalize_response(
    r: dict[str, Any],
    model: ModelID,
    structure: type[StructuredModelT] | None,
) -> Response[StructuredModelT]:
    """
    Transforms the OpenAI returned payload into an ag normalized Response object.
    """

    required_keys = ("model", "output", "usage")
    for key in required_keys:
        if key not in r:
            raise ValueError(f"response does not contain a `{key}` key")

    usage_raw = r["usage"]
    usage = Usage(
        input_tokens=usage_raw["input_tokens"],
        output_tokens=usage_raw["output_tokens"],
        input_tokens_cached=usage_raw["input_tokens_details"]["cached_tokens"],
        output_tokens_reasoning=usage_raw["output_tokens_details"]["reasoning_tokens"],
    )

    out_content_parts: list[str] = list()
    out_tool_calls: list[ToolCall] = list()
    out_reasoning: list[Reasoning] = list()
    for out_item in r["output"]:
        if out_item["type"] == "message":
            for out_content in out_item["content"]:
                if "text" in out_content:
                    out_part = out_content["text"]
                    out_content_parts.append(out_part)
        elif out_item["type"] == "function_call":
            tc = ToolCall(
                call_id=out_item["call_id"],
                name=out_item["name"],
                args=out_item["arguments"],
            )
            out_tool_calls.append(tc)
        elif out_item["type"] == "reasoning":
            reasoning = Reasoning(encrypted_content=out_item["encrypted_content"])
            out_reasoning.append(reasoning)
        else:
            warnings.warn(f"While parsing OpenAI response found unhandled type: {out_item['type']}")

    out_content = "\n".join(out_content_parts)
    out_parsed = structure.model_validate_json(out_content) if structure else None

    if len(out_reasoning) > 1:
        raise AssertionError(
            "a single call returned multiple reasoning blocks, let's adapt the Response type"
        )

    return Response(
        model=model,
        model_api=r["model"],
        usage=usage,
        content=out_content,
        tool_calls=out_tool_calls,
        structured_output=out_parsed,
        reasoning=out_reasoning[0] if out_reasoning else None,
    )
