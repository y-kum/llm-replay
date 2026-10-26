"""The whole workflow in one run, no API key needed.

A stub stands in for the model server so this runs anywhere. In your project the client talks to the real API;
nothing else changes.

    python examples/offline_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import openai

import llm_replay
from llm_replay.cli import main as cli

live_calls = {"n": 0}


def stub_model(request: httpx.Request) -> httpx.Response:
    """Pretend model: answers with the number of words in the user's message."""
    live_calls["n"] += 1
    body = json.loads(request.content)
    question = body["messages"][-1]["content"]
    answer = f"Your message has {len(question.split())} words."
    return httpx.Response(
        200,
        json={
            "id": f"stub-{live_calls['n']}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        },
    )


client = openai.OpenAI(
    api_key="not-needed-for-the-stub",
    base_url="http://stub.local/v1",
    http_client=httpx.Client(transport=httpx.MockTransport(stub_model)),
)


def count_words(text: str, prompt_version: int = 1) -> str:
    system = "Count the words." if prompt_version == 1 else "Count the words. Ignore punctuation."
    r = client.chat.completions.create(
        model="demo-model",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
    )
    return r.choices[0].message.content


def main() -> None:
    tape = Path(tempfile.mkdtemp()) / "count_words.jsonl"
    prices = {"demo-model": {"input": 0.5, "output": 1.5}}

    print("1) Record: calls go to the model and are written to the tape")
    with llm_replay.session(tape, mode="record", prices=prices) as s:
        print("  ", count_words("record me once please"))
    print("  ", s.report.summary())

    print("\n2) Replay: same calls, served from the tape, no network")
    with llm_replay.session(tape, mode="replay", prices=prices) as s:
        print("  ", count_words("record me once please"))
    print("  ", s.report.summary(), f"| live calls so far: {live_calls['n']}")

    print("\n3) Someone edits the prompt: replay fails and shows exactly what changed")
    try:
        with llm_replay.session(tape, mode="replay"):
            count_words("record me once please", prompt_version=2)
    except llm_replay.ReplayMiss as e:
        print("   ReplayMiss:", str(e).splitlines()[0])
        print("\n".join("     " + ln for ln in str(e).splitlines()[1:12]))

    print("\n4) Re-record into a new tape and compare the two")
    new_tape = tape.with_name("count_words.v2.jsonl")
    with llm_replay.session(new_tape, mode="record"):
        count_words("record me once please", prompt_version=2)
    cli(["diff", str(tape), str(new_tape)])


if __name__ == "__main__":
    main()
