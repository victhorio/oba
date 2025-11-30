import time
from dataclasses import dataclass
from uuid import uuid4

from ag import Agent
from ag.common import Usage
from ag.models import ToolCall
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Header, Markdown, RichLog, Static, TextArea


class ObaTUI(App[Usage]):
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    VerticalScroll {
        scrollbar-size-vertical: 1;
        scrollbar-color: skyblue 60%;
        scrollbar-color-hover: skyblue 90%;
        scrollbar-color-active: skyblue 100%;
    }

    TextArea {
        scrollbar-size-vertical: 1;
        scrollbar-color: skyblue 60%;
        scrollbar-color-hover: skyblue 90%;
        scrollbar-color-active: skyblue 100%;
    }

    MarkdownFence{
        margin-bottom: 1;
        margin-top: 0;
        margin-left: 0;
        margin-right: 0;
        background: white 10%;
        & > Label {
            padding: 0;
        }
    }

    #status-bar {
        height: 1;
        border: none;
        color: white;
        content-align: right middle;
    }

    #conversation {
        height: 1fr;
        border: solid orange;
    }

    #message-log {
        height: auto;
    }

    #input-box {
        height: auto;
        min-height: 3;
        max-height: 8;
        border: solid skyblue;
    }

    .response-metadata {
        color: darkgray;
        margin-bottom: 1;
    }

    .user-message {
        color: white;
        padding: 1;
        background: skyblue 10%;
    }

    .streaming-response {
        padding-left: 1;
        padding-right: 1;
        padding-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        agent: Agent,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.session_id = session_id or str(uuid4())

        # tracks if we're currently waiting for a response
        self._is_processing = False

        # tracks the current response string that's being streamed in when a response is being
        # created
        self._current_response = ""

        # buffer for string deltas that are being streamed in when a response is being created
        # since we only want to actually render them in batch for performance reasons
        self._delta_buffer: list[str] = []
        self._delta_buffer_size = 10

        # usage information for status bar
        self._total_tokens = 0
        self._token_cost = 0.0
        self._tool_cost = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="conversation"):
            yield RichLog(id="message-log", highlight=True, markup=True, wrap=True)
        yield ChatTextArea(placeholder="Type your message…", id="input-box")
        yield Static("0 tokens • $0.000 tokens cost • $0.000 tool cost", id="status-bar")

    async def on_mount(self) -> None:
        self.title = f"oba • {self.agent.model.model_id}"

        # focus on input box on startup
        self.query_one("#input-box", ChatTextArea).focus()

    async def on_unmount(self) -> None:
        if self.agent.memory:
            # the return value of `App.run` is fetched through the `_return_value` attribute
            self._return_value = self.agent.memory.get_usage(self.session_id)

    async def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        query = event.value.strip()
        input_widget = event.text_area
        input_widget.clear()

        if not query:
            return

        if query.lower() in ("exit", "quit", ":q"):
            self.exit()
            return

        if self._is_processing:
            return

        conversation = self.query_one("#conversation", VerticalScroll)

        user_message = Static(query, classes="user-message")
        await conversation.mount(user_message)

        # if the user sent something, we want to scroll down to it
        conversation.scroll_end(animate=False)

        # set the correct states before running
        self._is_processing = True
        input_widget.placeholder = "Waiting for response..."
        input_widget.disabled = True

        # run the query in a background worker
        self._generate_response(query)

    @work(exclusive=True)
    async def _generate_response(self, query: str) -> None:
        log = self.query_one("#message-log", RichLog)
        conversation = self.query_one("#conversation", VerticalScroll)

        # Mount streaming Markdown widget
        streaming_widget = Markdown("", classes="streaming-response")
        await conversation.mount(streaming_widget)
        conversation.scroll_end(animate=False)

        # as a sanity check, let's make sure we're starting off with clear state
        self._current_response = ""
        self._delta_buffer.clear()

        try:
            tic = time.perf_counter()
            response = await self.agent.stream(
                input=query,
                callback=lambda delta: self._render_delta(delta, streaming_widget),
                session_id=self.session_id,
            )
            toc = time.perf_counter()

            self._update_status_bar(response.usage)

            # show response metadata after each response
            response_metadata = Static(
                f"  [took {toc - tic:.2f}s]",
                classes="response-metadata",
            )
            await conversation.mount(response_metadata)

        except Exception as e:
            log.write(Text(f"Error: {e}", style="bold red"))

        finally:
            # Re-enable input
            self._is_processing = False
            input_widget = self.query_one("#input-box", ChatTextArea)
            input_widget.disabled = False
            input_widget.placeholder = "Type your message..."
            input_widget.focus()

    def _delta_buffer_flush(self, target_widget: Markdown) -> None:
        if self._delta_buffer:
            self._current_response += "".join(self._delta_buffer)
            self._delta_buffer.clear()
            target_widget.update(self._current_response)

    def _render_delta(self, delta: ToolCall | str | None, target_widget: Markdown) -> None:
        if isinstance(delta, ToolCall):
            # we immediately want to render tool calls, but first we need to make sure
            # we flush the buffer if there's anything there just in case
            self._delta_buffer_flush(target_widget)

            self._current_response += self._tool_call_into_str(delta)
            target_widget.update(self._current_response)

        elif isinstance(delta, str):
            # we only render string deltas once the buffer becomes full
            self._delta_buffer.append(delta)
            if len(self._delta_buffer) >= self._delta_buffer_size:
                self._delta_buffer_flush(target_widget)

        elif delta is None:
            # response finished - flush remaining buffer
            self._delta_buffer_flush(target_widget)

        else:
            # the line below should be greyed out by the LSP based on type checking
            raise ValueError(f"Unexpected delta type: {type(delta)}")

    @staticmethod
    def _tool_call_into_str(delta: ToolCall) -> str:
        prefix = f"```python\n{delta.name}(\n"
        suffix = "\n)\n```\n"

        lines: list[str] = []
        for k, v in delta.parsed_args.items():
            sv = str(v)
            if len(sv) > 80:
                sv = sv[:80] + "..."
            lines.append(f"    {k} = {repr(sv)}")

        return prefix + "\n".join(lines) + suffix

    def _update_status_bar(self, usage: Usage) -> None:
        status_bar = self.query_one("#status-bar", Static)

        self._total_tokens += usage.input_tokens + usage.output_tokens
        self._token_cost += usage.total_cost
        self._tool_cost += usage.tool_costs

        status_bar.update(
            f"{self._total_tokens:,} [green](+ {usage.input_tokens + usage.output_tokens:,})[/green] tokens"
            f" • ${self._token_cost:.3f} [green](+ ${usage.total_cost:.3f})[/green] tokens cost"
            f" • ${self._tool_cost:.3f} [green](+ ${usage.tool_costs:.3f})[/green] tool cost"
        )

    async def action_quit(self) -> None:
        self.exit()


class ChatTextArea(TextArea):
    """A TextArea that submits on ctrl+enter"""

    @dataclass
    class Submitted(Message):
        text_area: ChatTextArea
        value: str

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "ctrl+enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
        else:
            await super()._on_key(event)
