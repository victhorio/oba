from . import openai, types


async def run_manual_tests():
    from ._openai_manual_test import main as openai_main

    total_cost = 0.0
    total_cost += await openai_main()
    print(f"\033[33;1mTotal cost: ${total_cost:.3f}\033[0m")


__all__ = ["openai", "types", "run_manual_tests"]
