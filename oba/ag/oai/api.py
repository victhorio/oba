import os
from typing import Literal

from attrs import define
from httpx import AsyncClient
from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@define(slots=True)
class Message:
    role: Role
    # TODO: support non text inputs as well
    content: str


# TODO: look into include: reasoning.encrypted_content
# TODO: look into include: message.output_text.logprobs
async def generate(
    client: AsyncClient,
    input: list[Message],
    model: str,
    max_output_tokens: int | None = None,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None,
    # temperature: float = 0.5,
    # stream: bool = False,
    text: type[BaseModel] | None = None,
    tools: list[type[BaseModel]] | None = None,
    tool_choice: Literal["none", "auto", "required"] | None = None,
    parallel_tool_calls: bool = False,
):
    api_key = os.getenv("OPENAI_API_KEY")

    payload = {
        "input": [{"role": m.role, "content": m.content, "type": "message"} for m in input],
        "model": model,
    }

    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if text:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": text.__name__,
                "strict": True,
                "schema": text.model_json_schema() | {"additionalProperties": False},
            }
        }

    if tools or tool_choice or parallel_tool_calls:
        raise NotImplementedError("tool calling")

    response = await client.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
    )

    return response.json()
