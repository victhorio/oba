import argparse
import asyncio
from typing import Literal
from uuid import uuid4

import httpx
from rich import print as rich_print

from .agent import agent_create
from .configs import config_load
from .tui import ObaTUI


def main() -> int:
    return asyncio.run(main_async())


async def main_async() -> int:
    model, is_test, session_id = _parse_args()

    config = config_load(is_test)
    client = httpx.AsyncClient()

    # we pass the session_id so the setup cost already gets associated when
    # the memory is created
    agent = await agent_create(config, model, client, session_id=session_id)

    app = ObaTUI(agent=agent, session_id=session_id)
    usage = await app.run_async()

    if usage:
        print("╭─ Session Stats ─────────────────────╮")
        print(f"│ Input tokens:        {usage.input_tokens:>13,}  │")
        print(f"│ Cached input tokens: {usage.input_tokens_cached:>13,}  │")
        print(f"│ Output tokens:       {usage.output_tokens:>13,}  │")
        print(f"│ Tokens cost:         ${usage.total_cost:>12.3f}  │")
        print(f"│ Tool costs:          ${usage.tool_costs:>12.3f}  │")
        print("╰─────────────────────────────────────╯")

    rich_print(f"Use session ID [skyblue]{app.session_id}[/skyblue] to continue this conversation.")

    return 0


def _parse_args() -> tuple[Literal["gpt", "claude"], bool, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--session", type=str, help="Session ID to continue conversation", default=None
    )
    args = parser.parse_args()

    model = args.model
    if model not in ("gpt", "claude"):
        raise RuntimeError(f"Invalid model: {model}")

    is_test = args.test
    if not isinstance(is_test, bool):
        raise RuntimeError(f"Invalid test argument: {is_test}")

    session_id = args.session
    if session_id is not None and not isinstance(session_id, str):
        raise RuntimeError("Session ID must be a string")

    session_id = session_id or str(uuid4())

    return model, is_test, session_id
