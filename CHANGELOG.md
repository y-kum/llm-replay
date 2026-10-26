# Changelog

## 0.1.0 (unreleased)
- Record / replay / auto / off modes; JSONL tapes with redaction of credentials.
- Adapters for OpenAI (chat completions, responses), Anthropic (messages) and Google Gen AI (generate_content,
  generate_content_stream); sync, async and streaming.
- Prompt-drift report on a replay miss (closest recording on the same route, with a request diff).
- `@replayable` decorator for any function.
- pytest plugin (`@pytest.mark.llm_replay`, `--llm-mode`, run report) and CLI (`stats`, `diff`, `check`).
