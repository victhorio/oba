from abc import ABC, abstractmethod
from typing import Literal

from httpx import AsyncClient

from oba.ag.models.constants import DEFAULT_MAX_OUTPUT_TOKENS
from oba.ag.models.types import Message, ModelID, Response, StructuredModelT
from oba.ag.tool import Tool

ToolChoice = Literal["none", "auto", "required"]


class Model(ABC):
    def __init__(
        self,
        model_id: ModelID,
    ):
        self.model_id: ModelID = model_id

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        client: AsyncClient,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_output: type[StructuredModelT] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool = False,
        timeout: int = 20,
        debug: bool = False,
    ) -> Response[StructuredModelT]: ...
