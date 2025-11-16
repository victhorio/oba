import asyncio
from uuid import uuid4

from . import agents, configs


def main() -> int:
    return asyncio.run(repl())


async def repl() -> int:
    config = configs.load()
    agent = agents.new(config)
    # seeing as memory is typed as optional, since we plan to use it we need to assert it
    # for the lsp to calm down about accesses to it
    assert agent.memory

    session_id = str(uuid4())

    print(f"{_ANSI_GREY}Using model: {config.model_id}\n{_ANSI_RESET}")

    while True:
        try:
            query = input(f"{_ANSI_BOLD}> ")
            print(_ANSI_RESET, end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            print(end="", flush=True)
            break

        if not query.strip():
            continue
        if query.lower() in ("exit", "quit", ":q"):
            break

        response = await agent.run(input=query, session_id=session_id)
        print(response.content, end="\n\n")

    usage = agent.memory.get_usage(session_id)
    print(f"\n{_ANSI_BOLD}Session summary:{_ANSI_RESET}")
    print(f"\tInput tokens: {usage.input_tokens:,}\tOutput tokens: {usage.output_tokens:,}")
    print(f"\tTotal cost: ${usage.total_cost:.3f}")

    return 0


_ANSI_GREY = "\033[90m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"
