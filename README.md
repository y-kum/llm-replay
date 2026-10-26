# llm-replay

Record your LLM calls once. Replay them in tests and CI: deterministic, free, offline. When a prompt changes,
see exactly what changed instead of a mystery failure.

```bash
pip install llm-replay
```

Works with the official **OpenAI** (and OpenAI-compatible: Groq, Together, vLLM, Ollama), **Anthropic** and
**Google Gen AI** (Gemini, Vertex AI) Python SDKs, sync and async, streaming included. Nothing to change in your
application code.

## Why

Refactoring an LLM pipeline raises one question: *did the answers change?* Live calls are slow, cost money and are
not deterministic, so most teams either skip the check or eyeball a few outputs.

`llm-replay` records each call (request and response) to a small JSONL "tape". Later runs replay from the tape, so:

- tests and CI run in milliseconds, with no API key and no spend;
- the same inputs always give the same outputs, so a failing test means *your code* changed;
- when a request no longer matches a recording, the failure shows the diff of the request (a reworded prompt, a new
  temperature, a different tool list);
- `llm-replay diff` compares two recordings, e.g. before and after upgrading the model, and lists which answers changed.

The first version of this lived inside a production voice-to-design service, where it proved that a rewrite that
cut latency by 31% returned identical results. This is that tool, made general.

## Quick start: pytest

```python
import pytest
from myapp import summarise          # uses openai / anthropic / google-genai inside

@pytest.mark.llm_replay
def test_summary_mentions_price():
    assert "price" in summarise("...")
```

```bash
pytest --llm-mode=record      # once, with your API key: writes tests/tapes/<module>__<test>.jsonl
pytest                        # from then on: replays, no key, no network
```

Commit the tapes. In CI:

```yaml
- run: pytest                 # default mode is replay
- run: llm-replay check       # fails if any call had no recording, and prints the request diff
```

## Quick start: anywhere

```python
import llm_replay

with llm_replay.session("tapes/eval.jsonl", mode="auto") as s:
    run_my_pipeline()
print(s.report.summary())
# llm-replay: 118 replayed, 2 recorded, 0 missing | 512,330 tokens replayed (~$0.3100 not spent)
```

## Modes

| Mode | Model called? | Tape | Use it for |
|---|---|---|---|
| `record` | always | rewritten from scratch | creating or refreshing a tape |
| `replay` | never | read only; a call with no recording fails | tests, CI (default) |
| `auto` | only for calls not on the tape | missing calls are appended | growing a test suite |
| `off` | always | untouched | turning it off without code changes |

Set it with `mode=`, `--llm-mode`, the `LLM_REPLAY_MODE` environment variable, or `llm_replay_mode` in your
pytest ini.

## When a recording is missing

```
ReplayMiss: llm-replay: no recording for openai.chat:gpt-4o-mini (key 817b9ab0ead605d6117c7366).
Closest recorded call is 95% similar; the request changed:
  --- recorded
  +++ this call
  @@ -3,3 +3,3 @@
       {
  -      "content": "Count the words.",
  +      "content": "Count the words. Ignore punctuation.",
         "role": "system"
```

If the change was intended, re-record (`--llm-mode=auto` adds only the new calls), review the tape diff, commit.

## Command line

```bash
llm-replay stats tests/tapes/search.jsonl --prices prices.json   # calls, tokens, what a live run would cost
llm-replay diff old.jsonl new.jsonl --fail-on-change             # answers changed / prompts changed / added / removed
llm-replay check .llm-replay/report.json --max-live 0            # CI gate after a pytest run
```

`prices.json` is your own table (`{"model": {"input": USD per 1M tokens, "output": ...}}`); prices change, so the
library ships none. See `prices.example.json` for the shape.

## Your own functions

Anything that isn't one of the supported SDK calls (a raw HTTP call to a model server, an embedding service, a
vision endpoint) can be recorded with a decorator:

```python
@llm_replay.replayable("vision.classify")
def classify(image_url: str) -> dict:
    return requests.post(VISION_URL, json={"url": image_url}).json()
```

Arguments form the key; the return value must be JSON-serialisable (or pass `dump=` / `load=`). Outside a session
the function runs as usual.

## What is recorded, and what isn't

- **Keyed on:** model, messages/contents, tools, response format and sampling parameters, after normalisation.
- **Ignored for matching:** `timeout`, `user`, `metadata`, `extra_headers`, request ids and other values that don't
  change the answer.
- **Never written:** API keys, auth headers, HTTP clients. Strings that look like keys (`sk-…`, `sk-ant-…`, `AIza…`,
  `gsk_…`, `hf_…`, bearer tokens) are replaced with `[REDACTED]`. Tapes are plain text: read them before committing.
- **Repeated identical calls** (sampling with temperature > 0) replay in the recorded order.
- **Binary inputs** (audio, images) are keyed by their SHA-256 and not stored.

## How it compares

HTTP recorders such as [vcrpy](https://github.com/kevin1024/vcrpy) and pytest-recording replay raw HTTP. They work,
but every prompt or parameter change becomes an opaque mismatch, streaming bodies are awkward, and they know nothing
about tokens or cost. `llm-replay` works at the SDK call: it understands requests and responses, tells you *what*
changed, rebuilds real SDK response objects (so type checks and attribute access still work), and reports tokens
and cost. If you need to record arbitrary HTTP, use vcrpy; if you test LLM pipelines, this is narrower and more
helpful.

## Limitations (v0.1)

- SDK helpers that bypass `create` are not intercepted yet: OpenAI `.parse()` / `.stream()`, Anthropic `.stream()`.
  Use `create(..., stream=True)` or wrap your function with `@replayable`.
- A stream is written to the tape when it is fully consumed or closed.
- Patching is process-wide while a session is open; run sessions one at a time per process (pytest's default).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests
python examples/offline_demo.py      # the whole workflow against a stub model, no key needed
```

Tests run the real SDK clients against an in-process HTTP stub, so they never touch the network.

## License

MIT
