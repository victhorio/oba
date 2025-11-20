import argparse
import asyncio
import time
from typing import Literal
from uuid import uuid4

import httpx

from . import agents, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    args = parser.parse_args()

    model = args.model
    assert model in ("gpt", "gemini", "claude")

    return asyncio.run(repl(model))


async def repl(model: Literal["gpt", "gemini", "claude"]) -> int:
    async with httpx.AsyncClient() as client:
        config = configs.load()
        agent = agents.new(config, model, client)
        # seeing as memory is typed as optional, since we plan to use it we need to assert it
        # for the lsp to calm down about accesses to it
        assert agent.memory

        session_id = str(uuid4())

        print(f"{_ANSI_GREY}Using model: {agent.model.model_id}\n{_ANSI_RESET}")

        while True:
            try:
                query = input(f"{_ANSI_BOLD}> ")
                print(_ANSI_RESET, end="\n\n")
            except (EOFError, KeyboardInterrupt):
                print(end="", flush=True)
                break

            if not query.strip():
                continue
            if query.lower() in ("exit", "quit", ":q"):
                break

            tic = time.perf_counter()
            response = await agent.run(input=query, session_id=session_id)
            toc = time.perf_counter() - tic
            print(response.content, end="\n")
            print(
                f"{_ANSI_GREY}"
                f"\tTokens:     {response.usage.input_tokens + response.usage.output_tokens:,}\n"
                f"\tTime taken: {toc:.1f} seconds\n"
                f"{_ANSI_RESET}",
                end="\n\n",
            )

        usage = agent.memory.get_usage(session_id)

    print(f"\n{_ANSI_BOLD}Session summary:{_ANSI_RESET}")
    print(f"\tInput tokens: {usage.input_tokens:,}\tOutput tokens: {usage.output_tokens:,}")
    print(f"\tTotal cost: ${usage.total_cost:.3f}")

    return 0


_ANSI_GREY = "\033[90m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"
