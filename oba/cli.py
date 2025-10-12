import asyncio

from pydantic import BaseModel

from oba import agents, configs


class RollingAverage(BaseModel):
    avg: float = 0.0
    n: int = 0

    def update(self, new: float | None) -> "RollingAverage":
        if new is not None:
            self.n += 1
            self.avg += (new - self.avg) / self.n
        return self


def main() -> int:
    return asyncio.run(repl())


async def repl() -> int:
    config = configs.load()
    agent = agents.new(config)
    print(f"{_ANSI_GREY}Using model: {config.model_id}\n{_ANSI_RESET}")

    usage = agents.Usage()
    ttft = RollingAverage()
    duration = RollingAverage()

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

        response = await agents.send_message(agent, query)
        print(response.content, end="\n\n")

        _update_usage(usage, response.usage)
        ttft = ttft.update(response.metrics.ttft)
        duration = duration.update(response.metrics.duration)

    print(f"\n{_ANSI_BOLD}Session summary:{_ANSI_RESET}")
    print(
        f"\tInput tokens: {usage.input_tokens:,}\tOutput tokens: {usage.output_tokens:,}"
    )
    print(f"\tTotal cost: ${usage.total_cost:.6f}")
    if ttft.n:
        print(f"\tAverage time to first token: {ttft.avg:.2f}s over {ttft.n} responses")
    if duration.n:
        print(f"\tAverage duration: {duration.avg:.2f}s over {duration.n} responses")

    return 0


def _update_usage(total: agents.Usage, new: agents.Usage) -> None:
    total.input_tokens += new.input_tokens
    total.output_tokens += new.output_tokens
    total.reasoning_tokens += new.reasoning_tokens
    total.total_tokens += new.total_tokens
    total.total_cost += new.total_cost


_ANSI_GREY = "\033[90m"
_ANSI_BOLD = "\033[1m"
_ANSI_RESET = "\033[0m"
