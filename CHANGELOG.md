# Changelog

## 0.1.1 (2026-10-30)
- First release on PyPI, as `pytest-llm-replay`: PyPI rejected `llm-replay` as too similar to `llmreplay`. The import
  (`llm_replay`) and the command (`llm-replay`) are unchanged. 0.1.0 was tagged but never published.

## 0.1.0 (2026-10-30)
- Record / replay / auto / off modes; JSONL tapes with redaction of credentials.
- Adapters for OpenAI (chat completions, responses), Anthropic (messages) and Google Gen AI (generate_content,
  generate_content_stream); sync, async and streaming.
- Prompt-drift report on a replay miss (closest recording on the same route, with a request diff).
- `@replayable` decorator for any function.
- Responses from OpenAI-compatible providers replay even when they carry values outside the strict schema.
- pytest plugin (`@pytest.mark.llm_replay`, `--llm-mode`, run report) and CLI (`stats`, `diff`, `check`).
