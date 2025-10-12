import asyncio

import oba.agent
import oba.settings


def main() -> int:
    return asyncio.run(repl())


async def repl() -> int:
    settings = oba.settings.load()
    agent = oba.agent.new(settings)

    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not query.strip():
            continue
        if query == "exit":
            break

        response = await oba.agent.send_message(agent, query)
        print(response.content, end="\n\n")

    return 0
