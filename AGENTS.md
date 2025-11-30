NOTICE: This AGENTS.md file was last updated 2025-11-29.

## Overview

**oba** is a terminal AI assistant with access to the user's Obsidian vault. It provides a TUI for
conversing with LLMs that can read, search, and navigate vault notes.

The repo is a uv workspace containing:
- `src/oba/` — the main CLI application
- `packages/ag/` — a custom LLM framework library ("ag" = "we have agno at home")

Requires Python 3.14+. Uses `uv` for dependency management.

Entry point: `oba.cli:main` → parses args, launches `ChatApp` (Textual app).

## Directory Structure

```
src/oba/
├── cli.py          # Main entry, ChatApp (Textual TUI)
├── agent.py        # Agent factory (agent_create)
├── configs.py      # Config loading (~/.config/oba/settings.json)
├── vault.py        # Vault helpers (read_note, get_recent_dailies, index building)
├── prompts/
│   └── system_prompt.txt   # Template with {name}, {now}, {agents_md}, {recent_dailies}
└── tools/
    └── fs.py       # Vault tools: ReadNote, ListDir, RipGrep

packages/ag/src/ag/
├── agent.py        # Agent class (run, stream, tool_calls)
├── tool.py         # Tool abstraction (spec: BaseModel + callable)
├── common.py       # Usage tracking
├── memory/
│   ├── base.py       # Memory ABC
|   ├── sqlite.py     # Sqlite based session storage
│   └── ephemeral.py  # In-memory session storage
├── models/
│   ├── model.py      # Model ABC, Response, cost tables
│   ├── message.py    # Content, Reasoning, ToolCall, ToolResult, Usage
│   ├── openai.py     # OpenAIModel (GPT-4.1, GPT-5 family)
│   └── anthropic.py  # AnthropicModel (Claude family)
└── tools/
    └── perplexity.py  # Web search tools via Perplexity API
```

## The `ag` Library

A minimal, framework-free LLM abstraction. Key concepts:

### Models

All models extend `Model` ABC and implement `generate()` and `stream()`:

```python
from ag.models import OpenAIModel, AnthropicModel

model = OpenAIModel(model_id="gpt-5.1", reasoning_effort="medium")
model = AnthropicModel(model_id="claude-sonnet-4-5", reasoning_effort=2048)
```

- `reasoning_effort`: OpenAI uses `"none"|"low"|"medium"|"high"`, Anthropic uses token budget (int)
- Models handle their own API calls via httpx, no SDK dependencies
- Message history uses normalized `Message` types that cache provider-specific payloads

### Tools

Tools combine a Pydantic spec with a callable:

```python
from ag.tool import Tool
from pydantic import BaseModel, Field

class MyTool(BaseModel):
    """Docstring becomes the tool description."""
    arg: str = Field(description="Argument description")

def my_callable(arg: str) -> str:
    return f"Result: {arg}"

tool = Tool(spec=MyTool, callable=my_callable)
```

- Callables can be sync or async (auto-wrapped)
- Return `str` or `tuple[str, float]` (result, cost)

### Agent

The `Agent` class orchestrates model calls with tool execution:

```python
from ag import Agent

agent = Agent(
    model=model,
    memory=EphemeralMemory(),
    system_prompt="...",
    tools=[tool1, tool2],
    client=httpx.AsyncClient(),
)

# Streaming (for TUI)
response = await agent.stream(
    input="Hello",
    callback=lambda delta: print(delta),
    session_id="session-123",
)

# Non-streaming
response = await agent.run(input="Hello")
```

## oba Application

### Configuration

Config stored at `~/.config/oba/settings.json`:
```json
{"user_name": "...", "vault_path": "/path/to/vault"}
```

First run prompts for setup. Use `--test` to use `vault_example/` as a temporary vault.

### CLI Flags

```bash
oba --model gpt     # Use OpenAI (default)
oba --model claude  # Use Anthropic
oba --test          # Use vault_example/ in temp dir
```

### Vault Tools

The agent has access to:
- **ReadNote**: Read note by name (Obsidian-style, no path/extension)
- **ListDir**: Explore folder structure
- **RipGrep**: Regex search across vault notes
- **AgenticWebSearch**: Perplexity-powered web search

### System Prompt

Located at `src/oba/prompts/system_prompt.txt`. Uses placeholders:
- `{name}` — user name from config
- `{now}` — current datetime
- `{agents_md}` — contents of vault's AGENTS.md (if exists)
- `{recent_dailies}` — last 3 daily notes

## Environment Variables

Required:
- `OPENAI_API_KEY` — for GPT models
- `ANTHROPIC_API_KEY` — for Claude models
- `PERPLEXITY_API_KEY` — for web search tool

## Code Style

- Uses `attrs` for dataclasses (`@define`)
- Uses Pydantic for tool specs and validation
- Type hints everywhere, strict pyright mode
- 100 char line limit (ruff)
- Pattern: exhaustive match checks with comments for LSP graying

## Key Patterns

### Message Types

```python
Message = Content | Reasoning | ToolCall | ToolResult
```

Messages cache provider-specific payloads in `payload_cache` to avoid recomputation during multi-turn conversations.

### Usage Tracking

```python
Usage(input_tokens, input_tokens_cached, output_tokens, total_cost, tool_costs)
```

Aggregated per session and additionally persisted to `~/.config/oba/usage.json` on exit.

### Streaming Callbacks

```python
callback=lambda delta: ...  # Receives str | ToolCall | None
```

- `str`: text delta (buffered and rendered in batches of 10 in the TUI for performance)
- `ToolCall`: completed tool call (rendered immediately)
- `None`: response finished
