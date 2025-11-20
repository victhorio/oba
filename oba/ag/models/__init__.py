from .anthropic import AnthropicModel
from .completions import CompletionsModel
from .message import Content, Message, ModelID, Reasoning, ToolCall, ToolResult
from .model import Model, Response
from .openai import OpenAIModel


async def run_manual_tests():
    from ._anthropic_manual_tests import run_manual_tests as anthropic_tests
    from ._gemini_manual_test import run_manual_tests as gemini_tests
    from ._openai_manual_test import run_manual_tests as openai_tests

    await anthropic_tests()
    await gemini_tests()
    await openai_tests()


__all__ = [
    "AnthropicModel",
    "CompletionsModel",
    "Content",
    "Message",
    "Model",
    "ModelID",
    "OpenAIModel",
    "Reasoning",
    "Response",
    "ToolCall",
    "ToolResult",
    "run_manual_tests",
]
