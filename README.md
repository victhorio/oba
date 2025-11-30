# oba

Early-stage terminal AI assisant connected to an Obsidian vault. Expect rough edges.

Two interesting components live in this repo:

- `src/oba/`: the CLI agent + Textual TUI app
- `packages/ag/`: a minimal LLM framework used by the app ("we have agno at home")

Some of the things already in here:

- OpenAI + Anthropic integration without vendor SDKs through `ag`
- Textual-based TUI for chat
- Persistent SQLite-backed session memory
- Agent tools for exploring an Obsidian vault (read notes, list dirs, ripgrep)
- Agent tool for performing Perplexity/Sonar backed web search
