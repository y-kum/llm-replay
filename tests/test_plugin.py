"""The pytest plugin, run inside a throwaway project with pytester."""

import json
import shutil
import sys

import pytest

TEST_FILE = """
import httpx, openai, pytest

calls = {"n": 0}

def handler(request):
    calls["n"] += 1
    body = __import__("json").loads(request.content)
    return httpx.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "answer to " + body["messages"][0]["content"]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })

client = openai.OpenAI(api_key="sk-test-0000000000000000000000", base_url="http://fake.local/v1", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler)))

def ask(q):
    return client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": q}])

@pytest.mark.llm_replay
def test_default_tape():
    assert ask("{QUESTION}").choices[0].message.content == "answer to {QUESTION}"

@pytest.mark.llm_replay("shared.jsonl")
def test_named_tape(llm_session):
    ask("shared")
    assert llm_session.tape.path.name == "shared.jsonl"
"""


@pytest.fixture(autouse=True)
def _no_bytecode(monkeypatch):
    # The test file is rewritten between in-process runs. Same-size rewrites within one second can make Python
    # reuse a stale .pyc (fast CI runners hit this), so never write or read cached bytecode here.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)


def _write(pytester, question="first question"):
    for cache in pytester.path.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    pytester.makepyfile(test_app=TEST_FILE.replace("{QUESTION}", question))


def test_record_then_replay_then_drift(pytester):
    _write(pytester)
    rec = pytester.runpytest("--llm-mode=record", "-p", "no:cacheprovider", "-p", "no:asyncio")
    rec.assert_outcomes(passed=2)
    tapes = sorted(p.name for p in (pytester.path / "tests" / "tapes").iterdir())
    assert tapes == ["shared.jsonl", "test_app__test_default_tape.jsonl"]
    rec.stdout.fnmatch_lines(["*llm-replay: 0 replayed, 2 recorded, 0 missing*"])

    rep = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:asyncio")  # default mode is replay
    rep.assert_outcomes(passed=2)
    rep.stdout.fnmatch_lines(["*llm-replay: 2 replayed, 0 recorded, 0 missing*16 tokens replayed*"])
    report = json.loads((pytester.path / ".llm-replay" / "report.json").read_text())
    assert report["ok"] is True and report["hits"] == 2

    _write(pytester, question="a reworded, longer second question")  # the prompt changes: replay must fail and say how
    drift = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:asyncio")
    drift.assert_outcomes(passed=1, failed=1)
    drift.stdout.fnmatch_lines(["*ReplayMiss*", "*similar*"])
    report = json.loads((pytester.path / ".llm-replay" / "report.json").read_text())
    assert report["ok"] is False and len(report["misses"]) == 1

    auto = pytester.runpytest("--llm-mode=auto", "-p", "no:cacheprovider", "-p", "no:asyncio")
    auto.assert_outcomes(passed=2)
    auto.stdout.fnmatch_lines(["*1 replayed, 1 recorded, 0 missing*"])


def test_ini_tape_dir(pytester):
    pytester.makeini("[pytest]\nllm_replay_tape_dir = recordings\nllm_replay_mode = auto\n")
    _write(pytester)
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:asyncio").assert_outcomes(passed=2)
    assert (pytester.path / "recordings" / "shared.jsonl").exists()
