#eval #llm #work

Ideas for measuring whether the helper is useful in practice. Builds on scope from [[250306_termflow_overview]].

## Prompt sets
- Summarize a multi-file diff and extract TODOs.
- Explain a failing test log and propose fixes.
- Draft a commit message and a short PR description.

## Experiments
- Simple “LLM-as-judge”: compare model answers to reference responses, score for correctness + brevity.
- Time-to-first-token vs total latency, especially when context gathering is heavy.
- Quick/slow mode comparison to see if extra tokens actually help.

## Metrics to track
- User edits to generated text (high edit distance = bad).
- Cache hit rate for repeated questions in the same repo.
- Per-command success rate and frequency to inform defaults.
