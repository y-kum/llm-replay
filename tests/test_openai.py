import json

import pytest
from conftest import FAKE_OPENAI_KEY
from openai.types.chat import ChatCompletion, ChatCompletionChunk

import llm_replay


def ask(client, text, **kw):
    return client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}], **kw)


def test_record_then_replay_without_network(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record") as s:
        live = ask(openai_client, "hello")
    assert s.report.recorded == 1 and openai_server.calls == 1

    openai_server.down()
    with llm_replay.session(tape, mode="replay") as s:
        again = ask(openai_client, "hello")
    assert isinstance(again, ChatCompletion)
    assert again.choices[0].message.content == live.choices[0].message.content == "echo:hello"
    assert s.report.hits == 1 and s.report.tokens_replayed == {"input": 11, "output": 4}


def test_tape_holds_no_api_key_and_redacts_secrets_in_content(tmp_path, openai_client):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record"):
        ask(openai_client, f"my key is {FAKE_OPENAI_KEY}")
    text = tape.read_text(encoding="utf-8")
    assert FAKE_OPENAI_KEY not in text
    assert "[REDACTED]" in text


def test_replay_miss_reports_what_changed(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record"):
        ask(openai_client, "summarise the attached report in three bullet points")
    openai_server.down()
    with pytest.raises(llm_replay.ReplayMiss) as e:
        with llm_replay.session(tape, mode="replay") as s:
            ask(openai_client, "summarise the attached report in five bullet points")
    msg = str(e.value)
    assert "similar" in msg and "-" in msg and "+" in msg
    assert "three" in msg and "five" in msg
    assert len(s.report.misses) == 1


def test_volatile_arguments_do_not_change_the_key(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record"):
        ask(openai_client, "hi", timeout=10, user="alice")
    openai_server.down()
    with llm_replay.session(tape, mode="replay"):
        r = ask(openai_client, "hi", timeout=99, user="bob")
    assert r.choices[0].message.content == "echo:hi"


def test_auto_records_only_what_is_missing(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="auto"):
        ask(openai_client, "a")
    with llm_replay.session(tape, mode="auto") as s:
        ask(openai_client, "a")
        ask(openai_client, "b")
    assert openai_server.calls == 2
    assert (s.report.hits, s.report.recorded) == (1, 1)


def test_repeated_identical_calls_replay_in_recorded_order(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record"):
        ids = [ask(openai_client, "same").id for _ in range(3)]
    assert ids == ["chatcmpl-1", "chatcmpl-2", "chatcmpl-3"]
    openai_server.down()
    with llm_replay.session(tape, mode="replay"):
        replayed = [ask(openai_client, "same").id for _ in range(4)]
    assert replayed == ids + ["chatcmpl-3"]  # extra calls repeat the last recording


def test_streaming_record_and_replay(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="record") as s:
        with ask(openai_client, "stream me", stream=True, stream_options={"include_usage": True}) as stream:
            live = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
    assert "".join(live) == "echo:stream me"
    assert s.report.tokens_live == {"input": 11, "output": 4}
    rec = json.loads(tape.read_text(encoding="utf-8").splitlines()[0])
    assert rec["stream"] is True and len(rec["response"]["chunks"]) == 4

    openai_server.down()
    with llm_replay.session(tape, mode="replay"):
        chunks = list(ask(openai_client, "stream me", stream=True, stream_options={"include_usage": True}))
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "echo:stream me"


def test_off_mode_goes_live_and_writes_nothing(tmp_path, openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    with llm_replay.session(tape, mode="off"):
        ask(openai_client, "x")
        ask(openai_client, "x")
    assert openai_server.calls == 2 and not tape.exists()


def test_outside_a_session_nothing_is_patched(openai_client, openai_server):
    ask(openai_client, "plain")
    assert openai_server.calls == 1


def test_mode_from_environment(tmp_path, openai_client, openai_server, monkeypatch):
    tape = tmp_path / "t.jsonl"
    monkeypatch.setenv("LLM_REPLAY_MODE", "auto")
    with llm_replay.session(tape) as s:
        ask(openai_client, "env")
    assert s.mode is llm_replay.Mode.AUTO and s.report.recorded == 1


async def test_async_and_async_stream(tmp_path, async_openai_client, openai_server):
    tape = tmp_path / "t.jsonl"
    async with llm_replay.session(tape, mode="record"):
        r = await async_openai_client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "x"}]
        )
        stream = await async_openai_client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "y"}], stream=True
        )
        live = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    openai_server.down()
    async with llm_replay.session(tape, mode="replay"):
        r2 = await async_openai_client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "x"}]
        )
        stream2 = await async_openai_client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "y"}], stream=True
        )
        replayed = "".join([c.choices[0].delta.content or "" async for c in stream2 if c.choices])
    assert r2.choices[0].message.content == r.choices[0].message.content
    assert replayed == live == "echo:y"
