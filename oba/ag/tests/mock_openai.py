# pyright: reportExplicitAny=false, reportAny=false, reportUnannotatedClassAttribute=false

import json
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails


class MockAsyncOpenAI(MagicMock):
    def set_responses(self, *responses: Response) -> None:
        self.responses.create = AsyncMock(side_effect=responses)

    @staticmethod
    def response_text(
        message: str,
    ) -> Response:
        message_obj = ResponseOutputMessage(
            id=str(uuid.uuid4()),
            content=[
                ResponseOutputText(
                    annotations=list(),
                    text=message,
                    type="output_text",
                    logprobs=list(),
                ),
            ],
            role="assistant",
            status="completed",
            type="message",
        )

        r = _response()
        r.output.append(message_obj)
        return r

    @staticmethod
    def response_tool_calls(
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> Response:
        assert len(tool_calls) > 0
        r = _response()
        for tool_name, tool_input in tool_calls:
            tool_input_str = json.dumps(tool_input)
            tool_call_obj = ResponseFunctionToolCall(
                arguments=tool_input_str,
                call_id=str(uuid.uuid4()),
                name=tool_name,
                type="function_call",
                status="completed",
            )
            r.output.append(tool_call_obj)
        return r


def _response() -> Response:
    reasoning = ResponseReasoningItem(
        id=str(uuid.uuid4()),
        summary=[],
        type="reasoning",
    )

    usage = ResponseUsage(
        input_tokens=250,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=150,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=50),
        total_tokens=400,
    )

    return Response(
        id=str(uuid.uuid4()),
        created_at=time.time(),
        model="api-returned-model",
        object="response",
        output=[reasoning],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=list(),
        usage=usage,
    )
