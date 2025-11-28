#work #llm #idea

Initial scoping note for a terminal-based AI helper focused on developer workflows.

## Core idea
- Keep the assistant inside the terminal so it can answer repo questions, summarize diffs, and draft commits without switching context.
- Fast startup, minimal config, and sensible defaults for model + repo context.

## Usage scenarios
- Summarize staged changes and propose a concise commit message.
- Answer “what changed in this folder since yesterday?” with links to files.
- Generate a quick TODO list from a diff or a failing test log.

## Must-haves
- Streaming responses so the CLI feels alive even when the model is slow.
- Local context caching to avoid repeating expensive requests.
- Clear logging/trace flag for debugging prompts.

## Nice-to-haves
- Interactive mode that keeps a short session alive.
- Hooks for custom prompts per repo.
- Opt-in analytics to learn which commands people actually use.

Next: feed this into [[250310_termflow_architecture]] and keep UX thoughts in [[250312_termflow_cli-ux]].
