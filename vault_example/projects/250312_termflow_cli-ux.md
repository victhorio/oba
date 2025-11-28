#ux #cli #work

CLI ergonomics and UX ideas pulled from [[2025-03-12]] and constrained by [[250310_termflow_architecture]].

## Command surface
- `tf summary` for staged changes; `tf commit --draft` to propose a message.
- `tf ask "question"` that automatically scopes to the current directory unless told otherwise.
- `--quick/--thorough` flag pair to toggle prompt depth and model choice.

## Streaming + output
- Default to streaming tokens; show a spinner during setup/context gathering.
- Colorized sections for context, answer, and follow-ups; allow `--no-color` for piping.
- Capture prompts/responses in a temp log when `--trace` is enabled.

## Tone + guardrails
- Keep responses terse by default; provide `--verbose` for more detail.
- Friendly errors when no git repo is detected; suggest `--path` overrides.
- Avoid anthropomorphic phrasing; the tool should feel like a sharp CLI, not a chatbot.
