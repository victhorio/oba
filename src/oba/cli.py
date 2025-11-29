import argparse
from typing import Literal
from uuid import uuid4

import httpx
from ag import Agent
from ag.models import ToolCall
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, RichLog, Static

from . import agents, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    model = args.model
    assert model in ("gpt", "claude")

    is_test: bool = args.test
    assert isinstance(is_test, bool)

    app = ChatApp(model=model, is_test=is_test)
    app.run()
    return 0


class ChatApp(App[None]):
    """A TUI chat application for interacting with AI models."""

    CSS = """
    #conversation {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #message-log {
        height: auto;
    }

    #input-box {
        dock: bottom;
        height: auto;
        max-height: 5;
        margin-top: 1;
    }

    .token-count {
        color: $text-muted;
        margin-bottom: 1;
    }

    .assistant-label {
        color: $secondary;
        text-style: bold;
    }

    .user-label {
        color: $warning;
        text-style: bold;
    }

    .user-message {
        color: $warning;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        model: Literal["gpt", "claude"],
        is_test: bool,
    ) -> None:
        super().__init__()
        self.model_family: Literal["gpt", "claude"] = model
        self.is_test = is_test
        self.session_id = str(uuid4())

        # These will be initialized in on_mount
        self._agent: Agent | None = None
        self._client: httpx.AsyncClient | None = None

        # Track if we're currently waiting for a response
        self._is_processing = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="conversation"):
            yield RichLog(id="message-log", highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Type your message...", id="input-box")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the agent and client when the app mounts."""
        self._client = httpx.AsyncClient()
        config = configs.load(self.is_test)
        self._agent = agents.new(config, self.model_family, self._client)

        # Update the header with model info
        self.title = f"oba • {self._agent.model.model_id}"
        self.sub_title = "AI Chat"

        # Focus the input box
        self.query_one("#input-box", Input).focus()

        # Welcome message
        log = self.query_one("#message-log", RichLog)
        log.write(Text(f"Using model: {self._agent.model.model_id}", style="bold white"))
        log.write(Text("Type your message below. Press Ctrl+C to exit.\n", style="dim"))

    async def on_unmount(self) -> None:
        """Clean up resources when the app unmounts."""
        if self._agent and self._agent.memory:
            usage = self._agent.memory.get_usage(self.session_id)
            # Print session stats to stderr so they appear after app closes
            import sys

            print("\n╭─ Session Stats ─────────────────────╮", file=sys.stderr)
            print(f"│ Input tokens:        {usage.input_tokens:>13,}  │", file=sys.stderr)
            print(f"│ Cached input tokens: {usage.input_tokens_cached:>13,}  │", file=sys.stderr)
            print(f"│ Output tokens:       {usage.output_tokens:>13,}  │", file=sys.stderr)
            print(f"│ Tokens cost:         ${usage.total_cost:>12.3f}  │", file=sys.stderr)
            print(f"│ Tool costs:          ${usage.tool_costs:>12.3f}  │", file=sys.stderr)
            print("╰─────────────────────────────────────╯", file=sys.stderr)

        if self._client:
            await self._client.aclose()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        query = event.value.strip()
        input_widget = event.input

        # Clear the input
        input_widget.clear()

        if not query:
            return

        # Handle exit commands
        if query.lower() in ("exit", "quit", ":q"):
            self.exit()
            return

        if self._is_processing:
            return

        assert self._agent is not None

        conversation = self.query_one("#conversation", VerticalScroll)

        # Display user message by mounting widgets (preserves order with responses)
        user_label = Static("▶ You", classes="user-label")
        user_message = Static(query, classes="user-message")
        await conversation.mount(user_label)
        await conversation.mount(user_message)

        # Scroll to bottom
        conversation.scroll_end(animate=False)

        # Start processing
        self._is_processing = True
        input_widget.placeholder = "Waiting for response..."
        input_widget.disabled = True

        # Run the AI response in a worker
        self._run_query(query)

    @work(exclusive=True)
    async def _run_query(self, query: str) -> None:
        """Run the AI query in a background worker."""
        assert self._agent is not None

        log = self.query_one("#message-log", RichLog)
        conversation = self.query_one("#conversation", VerticalScroll)

        # Track the current response for streaming
        current_response: list[str] = []

        # Mount assistant label and streaming Markdown widget
        assistant_label = Static("◀ Assistant", classes="assistant-label")
        await conversation.mount(assistant_label)
        streaming_widget = Markdown("")
        await conversation.mount(streaming_widget)
        conversation.scroll_end(animate=False)

        try:
            # Buffer for batching string deltas (renders every 10 chunks)
            pending_chunks: list[str] = []

            def flush_buffer() -> None:
                """Flush pending chunks to the streaming widget."""
                if pending_chunks:
                    current_response.extend(pending_chunks)
                    pending_chunks.clear()
                    streaming_widget.update("".join(current_response))
                    conversation.scroll_end(animate=False)

            def streamer(part: ToolCall | str | None) -> None:
                if isinstance(part, ToolCall):
                    # Flush any pending text first, then render tool call immediately
                    flush_buffer()
                    current_response.append(_tool_call_delta(part))
                    streaming_widget.update("".join(current_response))
                    conversation.scroll_end(animate=False)
                elif isinstance(part, str):
                    # Buffer string deltas, render in batches of 10
                    pending_chunks.append(part)
                    if len(pending_chunks) >= 10:
                        flush_buffer()
                else:
                    # Response finished - flush remaining buffer
                    flush_buffer()

            response = await self._agent.stream(
                input=query,
                callback=streamer,
                session_id=self.session_id,
            )

            # Show token usage after the streaming widget
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            token_label = Static(f"  [{total_tokens:,} tokens]", classes="token-count")
            await conversation.mount(token_label)

        except Exception as e:
            log.write(Text(f"Error: {e}", style="bold red"))

        finally:
            # Re-enable input
            self._is_processing = False
            input_widget = self.query_one("#input-box", Input)
            input_widget.disabled = False
            input_widget.placeholder = "Type your message..."
            input_widget.focus()

            # Scroll to bottom
            conversation.scroll_end(animate=False)

    async def action_quit(self) -> None:
        """Handle quit action."""
        self.exit()


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
