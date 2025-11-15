from oba.ag.agent import Agent
from oba.ag.memory import Memory
from oba.ag.tool import tool


async def run_manual_tests():
    from ._agent_manual_tests import main as agent_main

    total_cost = 0.0
    total_cost += await agent_main()
    print(f"\033[33;1mTotal cost: ${total_cost:.3f}\033[0m")


__all__ = ["Agent", "tool", "Memory", "run_manual_tests"]
