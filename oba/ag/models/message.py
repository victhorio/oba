import json
from typing import Literal

from attrs import define, field

Role = Literal["system", "user", "assistant"]
ModelID = Literal[
    "gpt-4.1",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
]


@define(slots=True)
class MessageBase:
    _provider_payload_cache: dict[str, dict[str, object]] = field(init=False, factory=dict)
    """
    All `Message` types should have this field which is used to store the result of the
    transformation from our standardized data type into a valid payload (dict[str, object])
    that can be sent in the API call for different providers so that we don't need to repeatedly
    transform the same message throughout different calls relating to the same conversation history.
    The root key of _provider_payload_cache is a provider specific key, so that if we're running
    the same conversation history with models from different providers, we won't have overlaps,
    invalidations or incompatibilities.
    """


@define(slots=True)
class Reasoning(MessageBase):
    # NOTE: We need to store reasoning blocks from OpenAI to maintain it in context,
    #       but they only accept returning encrypted reasoning contents.
    encrypted_content: str
    content: str = field(default="")


@define(slots=True)
class Content(MessageBase):
    role: Role
    # TODO: support non text inputs as well
    text: str


@define(slots=True)
class ToolCall(MessageBase):
    call_id: str
    name: str
    args: str | None = field(default=None)
    parsed_args: dict[str, object] = field(factory=dict)

    def __attrs_post_init__(self):
        if not self.parsed_args and not self.args:
            raise ValueError("either parsed_args or args must be set")
        if self.args and not self.parsed_args:
            self.parsed_args = json.loads(self.args)


@define(slots=True)
class ToolResult(MessageBase):
    call_id: str
    result: str


@define(slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    input_tokens_cached: int = 0
    output_tokens_reasoning: int = 0


Message = Content | Reasoning | ToolCall | ToolResult
"""The different message types that can be part of a conversation history: regular content, reasoning from models, tool calls and results."""
