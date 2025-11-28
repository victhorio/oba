import argparse
import asyncio
import time
from typing import Literal
from uuid import uuid4

import httpx
from ag.models import ToolCall
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.table import Table

from . import agents, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    args = parser.parse_args()

    model = args.model
    assert model in ("gpt", "gemini", "claude")

    return asyncio.run(repl(model))


async def repl(model: Literal["gpt", "gemini", "claude"]) -> int:
    console = Console(highlight=False)

    async with httpx.AsyncClient() as client:
        config = configs.load()
        agent = agents.new(config, model, client)
        # seeing as memory is typed as optional, since we plan to use it we need to assert it
        # for the lsp to calm down about accesses to it
        assert agent.memory

        session_id = str(uuid4())

        console.print(f"[bold][white]Using model:[/white] {agent.model.model_id}[/bold]\n")

        while True:
            try:
                console.print("[bold yellow]> [/bold yellow]", end="")
                query = input("\x1b[33m")
                print("\x1b[0m")
            except (EOFError, KeyboardInterrupt):
                break

            if not query.strip():
                continue
            if query.lower() in ("exit", "quit", ":q"):
                break

            tic = time.perf_counter()

            if model == "gemini":
                response = await agent.run(input=query, session_id=session_id)
                response_md = Markdown(response.content)
                console.print(response_md)
            else:
                with Live("", console=console, vertical_overflow="visible") as live:
                    full_response = [""]

                    def streamer(part: ToolCall | str) -> None:
                        if isinstance(part, ToolCall):
                            delta = _tool_call_delta(part)
                        else:
                            delta = part

                        full_response[0] += delta
                        live.update(Markdown(markup=full_response[0]))

                    response = await agent.stream(
                        input=query,
                        callback=streamer,
                        session_id=session_id,
                    )

            toc = time.perf_counter() - tic

            t = Table(show_header=False, box=box.MINIMAL)
            t.add_row("Time taken", f"{toc:.1f} seconds")
            t.add_row("Tokens", f"{response.usage.input_tokens + response.usage.output_tokens:,}")
            console.print(Padding.indent(t, 8), style="white")

        usage = agent.memory.get_usage(session_id)

    t = Table(title="Session stats", show_header=False, box=box.MINIMAL)
    t.add_row("Input tokens", f"{usage.input_tokens:,}")
    t.add_row("Output tokens", f"{usage.output_tokens:,}")
    t.add_row("Total cost", f"${usage.total_cost:.3f}", style="bright_white")
    t.add_row("Tool costs", f"${usage.tool_costs:.3f}")
    console.print(t, style="white")

    return 0


def _tool_call_delta(part: ToolCall) -> str:
    prefix = f"\n```\n{part.name}(\n"
    suffix = "\n)\n```\n\n"

    lines: list[str] = []
    for k, v in part.parsed_args.items():
        sv = str(v)
        if len(sv) > 80:
            sv = sv[:80] + "..."
        lines.append(f"    {k} = {sv}")

    return prefix + "\n".join(lines) + suffix
