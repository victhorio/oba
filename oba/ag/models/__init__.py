from . import openai, types


async def run_manual_tests():
    from ._anthropic_manual_tests import main as anthropic_main
    from ._openai_manual_test import main as openai_main

    total_cost = 0.0

    print("\033[31;1m")
    print("==============================")
    print("= ANTHROPIC TESTS            =")
    print("==============================")
    print("\033[0m")
    total_cost += await anthropic_main()

    print("\033[31;1m")
    print("==============================")
    print("= OPENAI TESTS               =")
    print("==============================")
    print("\033[0m")
    total_cost += await openai_main()

    print(f"\n\033[31;1mTotal cost: ${total_cost:.3f}\033[0m\n")


__all__ = ["openai", "types", "run_manual_tests"]
