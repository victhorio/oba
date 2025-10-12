import asyncio

from oba import agents, configs


def main() -> int:
    return asyncio.run(repl())


async def repl() -> int:
    config = configs.load()
    agent = agents.new(config)

    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not query.strip():
            continue
        if query == "exit":
            break

        response = await agents.send_message(agent, query)
        print(response.content, end="\n\n")

    return 0
