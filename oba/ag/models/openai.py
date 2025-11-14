import os
from typing import Any, Literal

from httpx import AsyncClient
from pydantic import BaseModel

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


async def generate(
    client: AsyncClient,
    messages: list[MessageTypes],
    model: ModelID,
    max_output_tokens: int | None = None,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None,
    structured_output: type[StructuredModelT] | None = None,
    tools: list[Tool] | None = None,
    tool_choice: Literal["none", "auto", "required"] | None = None,
    parallel_tool_calls: bool = False,
    timeout=20,
    debug: bool = False,
) -> Response[StructuredModelT]:
    api_key = os.getenv("OPENAI_API_KEY")

    payload = {
        "input": [_parse_input(m) for m in messages],
        "model": model,
        "store": False,
        "include": [
            "reasoning.encrypted_content",
        ],
    }

    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
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
        pprint.pprint(payload, width=110)
        print("------------------------------")

    response = await client.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=timeout,
    )

    if debug:
        import pprint

        print("--- Returned payload from OpenAI ---")
        pprint.pprint(response.json(), width=110)
        print("------------------------------------")

    response.raise_for_status()

    return _parse_response(response.json(), model=model, structure=structured_output)


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


def _parse_input(msg: MessageTypes) -> dict[str, object]:
    # TODO: some caching not to reparse same items everytime?

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


def _parse_response(
    r: dict[str, Any],
    model: ModelID,
    structure: type[BaseModel] | None = None,
) -> Response:
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
