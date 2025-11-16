import asyncio
from uuid import uuid4

from oba import agents, configs


def main() -> int:
    return asyncio.run(repl())


async def repl() -> int:
    config = configs.load()
    agent = agents.new(config)
    session_id = str(uuid4())

    print(f"{_ANSI_GREY}Using model: {config.model_id}\n{_ANSI_RESET}")

    while True:
        try:
            query = input(f"{_ANSI_BOLD}> ")
            print(_ANSI_RESET, end="")
        except (EOFError, KeyboardInterrupt):
            break

        if not query.strip():
            continue
        if query == "exit":
            break

        response = await agents.send_message(agent, query, session_id)
        print(response.content, end="\n\n")

    assert agent.memory
    usage = agent.memory.get_usage(session_id)

    print(f"\n{_ANSI_BOLD}Session summary:{_ANSI_RESET}")
    print(f"\tInput tokens: {usage.input_tokens:,}\tOutput tokens: {usage.output_tokens:,}")
    print(f"\tTotal cost: ${usage.total_cost:.3f}")

    return 0


_ANSI_GREY = "\033[90m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"
