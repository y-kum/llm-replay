from anthropic.types import Message

import llm_replay


def ask(client, text, **kw):
    return client.messages.create(
        model="claude-test", max_tokens=64, messages=[{"role": "user", "content": text}], **kw
    )


def test_record_then_replay(tmp_path, anthropic_client, anthropic_server):
    tape = tmp_path / "a.jsonl"
    with llm_replay.session(tape, mode="record") as s:
        live = ask(anthropic_client, "hola")
    assert s.report.tokens_live == {"input": 12, "output": 6}
    anthropic_server.down()
    with llm_replay.session(tape, mode="replay"):
        again = ask(anthropic_client, "hola")
    assert isinstance(again, Message)
    assert again.content[0].text == live.content[0].text == "echo:hola"


def test_stream_record_and_replay(tmp_path, anthropic_client, anthropic_server):
    tape = tmp_path / "a.jsonl"
    with llm_replay.session(tape, mode="record") as s:
        live = [e for e in ask(anthropic_client, "stream", stream=True)]
    assert s.report.tokens_live == {"input": 12, "output": 6}
    anthropic_server.down()
    with llm_replay.session(tape, mode="replay"):
        replayed = list(ask(anthropic_client, "stream", stream=True))
    assert [e.type for e in replayed] == [e.type for e in live]
    text = "".join(e.delta.text for e in replayed if e.type == "content_block_delta")
    assert text == "echo:stream"


async def test_async(tmp_path, async_anthropic_client, anthropic_server):
    tape = tmp_path / "a.jsonl"
    async with llm_replay.session(tape, mode="record"):
        live = await async_anthropic_client.messages.create(
            model="claude-test", max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )
    anthropic_server.down()
    async with llm_replay.session(tape, mode="replay"):
        again = await async_anthropic_client.messages.create(
            model="claude-test", max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )
    assert again.content[0].text == live.content[0].text
