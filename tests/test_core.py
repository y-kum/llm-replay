import json

import pytest

import llm_replay
from llm_replay._normalize import make_key, normalize_request
from llm_replay.cli import main as cli
from llm_replay.redact import Redactor


class NotGiven:  # same class name as the SDK sentinels
    pass


def test_sentinels_and_volatile_keys_do_not_affect_the_key():
    a = normalize_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": NotGiven(),
            "timeout": 5,
            "extra_headers": {"X-Trace": "1"},
        }
    )
    b = normalize_request({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert make_key("r:m", a) == make_key("r:m", b)


def test_bytes_are_hashed_not_stored():
    req = normalize_request({"audio": b"\x00\x01" * 1000})
    assert set(req["audio"]) == {"__bytes_sha256__", "len"} and req["audio"]["len"] == 2000


def test_redactor_patterns():
    r = Redactor()
    text = "keys: sk-abcdefghijklmnopqrstuv AIza" + "b" * 35 + " gsk_" + "c" * 30 + " Bearer " + "d" * 30
    out = r(text)
    assert out.count("[REDACTED]") == 4


# -- generic decorator ----------------------------------------------------------------------------------------------
calls = {"n": 0}


@llm_replay.replayable("vision.classify")
def classify(image_id: str, threshold: float = 0.5) -> dict:
    calls["n"] += 1
    return {"image": image_id, "label": "carpet", "score": 0.91}


@llm_replay.replayable("vision.aclassify")
async def aclassify(image_id: str) -> dict:
    calls["n"] += 1
    return {"image": image_id, "label": "tile"}


def test_replayable_function(tmp_path):
    calls["n"] = 0
    tape = tmp_path / "f.jsonl"
    assert classify("x")["label"] == "carpet" and calls["n"] == 1  # outside a session: plain call
    with llm_replay.session(tape, mode="record"):
        classify("room-1", threshold=0.7)
    with llm_replay.session(tape, mode="replay"):
        assert classify("room-1", threshold=0.7)["score"] == 0.91
        with pytest.raises(llm_replay.ReplayMiss):
            classify("room-2", threshold=0.7)
    assert calls["n"] == 2


async def test_replayable_async(tmp_path):
    calls["n"] = 0
    tape = tmp_path / "f.jsonl"
    async with llm_replay.session(tape, mode="record"):
        await aclassify("a")
    async with llm_replay.session(tape, mode="replay"):
        assert (await aclassify("a"))["label"] == "tile"
    assert calls["n"] == 1


# -- tapes, diff and check ------------------------------------------------------------------------------------------
def _ask(client, text):
    return client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])


def test_diff_reports_changed_answers_and_prompts(tmp_path, openai_client, openai_server, capsys):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    with llm_replay.session(old, mode="record"):
        _ask(openai_client, "keep")
        _ask(openai_client, "change my answer")
        _ask(openai_client, "old wording of the prompt")
    openai_server.respond = lambda body, n: __import__("httpx").Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": ("echo:keep" if "keep" in json.dumps(body) else "a different answer"),
                    },
                }
            ],
        },
    )
    with llm_replay.session(new, mode="record"):
        _ask(openai_client, "keep")
        _ask(openai_client, "change my answer")
        _ask(openai_client, "new wording of the prompt")
    code = cli(["diff", str(old), str(new), "--fail-on-change"])
    out = capsys.readouterr().out
    assert code == 1
    assert "1 unchanged, 1 answers changed, 1 prompts changed" in out
    assert "[prompt changed]" in out and "+" in out


def test_stats(tmp_path, openai_client, capsys):
    tape = tmp_path / "t.jsonl"
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"gpt-test": {"input": 1.0, "output": 2.0}}))
    with llm_replay.session(tape, mode="record"):
        _ask(openai_client, "a")
        _ask(openai_client, "b")
    assert cli(["stats", str(tape), "--prices", str(prices)]) == 0
    out = capsys.readouterr().out
    assert "2 recorded calls" in out and "22 in / 8 out" in out and "$0.000038" in out


def test_check_fails_on_misses(tmp_path, capsys):
    r = llm_replay.Report(tape="t", mode="replay")
    r.add_miss("openai.chat:gpt-test", "abc", None)
    path = tmp_path / "report.json"
    r.write(path)
    assert cli(["check", str(path)]) == 1
    ok = llm_replay.Report(tape="t", mode="replay", hits=3)
    ok.write(path)
    assert cli(["check", str(path)]) == 0
    assert cli(["check", str(tmp_path / "missing.json")]) == 2


def test_corrupt_tape_line_is_reported(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        llm_replay.Tape(p)
