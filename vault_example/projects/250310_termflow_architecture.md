#infra #architecture

High-level sketch for how the CLI talks to the model and manages context. Builds on [[250306_termflow_overview]].

## Components
- **CLI entrypoint:** parses commands/flags, gathers repo context, and hands off to a request builder.
- **Context builder:** collects file snippets, diffs, and git metadata; caches recent context locally.
- **LLM client:** wraps the model provider, supports streaming tokens, retries, and logging.

## Config strategy
- Dotfile in the repo for project-specific prompts and model choices.
- Env vars for provider keys; fallback to config file prompts.
- Cache directory for embeddings/summaries with a simple invalidation strategy.

## Open questions
- How to keep latency reasonable for large repos without heavy preprocessing?
- Should we embed a small history buffer for multi-turn mode?
- What minimal telemetry (if any) is acceptable?

Future notes: UX constraints in [[250312_termflow_cli-ux]] and eval ideas in [[250314_termflow_llm-evals]].
