from . import message, openai


async def run_manual_tests():
    from ._anthropic_manual_tests import run_manual_tests as anthropic_tests
    from ._openai_manual_test import run_manual_tests as openai_tests

    await anthropic_tests()
    await openai_tests()


__all__ = ["openai", "message", "run_manual_tests"]
