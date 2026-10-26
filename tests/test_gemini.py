"""google-genai: the SDK methods are replaced with fakes before the session starts, so the session wraps the fakes."""

import pytest
from google import genai
from google.genai import models, types

import llm_replay


def _resp(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=7, candidates_token_count=3),
    )


@pytest.fixture
def fake_gemini(monkeypatch):
    calls = {"n": 0}

    def generate_content(self, *, model, contents, config=None):
        calls["n"] += 1
        return _resp(f"echo:{contents}")

    def generate_content_stream(self, *, model, contents, config=None):
        calls["n"] += 1
        return iter([_resp("echo:"), _resp(str(contents))])

    async def agenerate_content(self, *, model, contents, config=None):
        calls["n"] += 1
        return _resp(f"echo:{contents}")

    monkeypatch.setattr(models.Models, "generate_content", generate_content)
    monkeypatch.setattr(models.Models, "generate_content_stream", generate_content_stream)
    monkeypatch.setattr(models.AsyncModels, "generate_content", agenerate_content)
    return calls


def test_record_then_replay(tmp_path, fake_gemini):
    client = genai.Client(api_key="AIza-test-not-a-real-key")
    tape = tmp_path / "g.jsonl"
    cfg = types.GenerateContentConfig(temperature=0)
    with llm_replay.session(tape, mode="record") as s:
        live = client.models.generate_content(model="gemini-test", contents="namaste", config=cfg)
    assert s.report.tokens_live == {"input": 7, "output": 3}
    with llm_replay.session(tape, mode="replay") as s2:
        again = client.models.generate_content(model="gemini-test", contents="namaste", config=cfg)
    assert fake_gemini["n"] == 1 and s2.report.hits == 1
    assert isinstance(again, types.GenerateContentResponse)
    assert again.text == live.text == "echo:namaste"


def test_config_change_is_a_miss(tmp_path, fake_gemini):
    client = genai.Client(api_key="AIza-test-not-a-real-key")
    tape = tmp_path / "g.jsonl"
    with llm_replay.session(tape, mode="record"):
        client.models.generate_content(
            model="gemini-test", contents="hi", config=types.GenerateContentConfig(temperature=0)
        )
    with pytest.raises(llm_replay.ReplayMiss, match="temperature"):
        with llm_replay.session(tape, mode="replay"):
            client.models.generate_content(
                model="gemini-test", contents="hi", config=types.GenerateContentConfig(temperature=0.7)
            )


def test_stream(tmp_path, fake_gemini):
    client = genai.Client(api_key="AIza-test-not-a-real-key")
    tape = tmp_path / "g.jsonl"
    with llm_replay.session(tape, mode="record"):
        live = "".join(c.text for c in client.models.generate_content_stream(model="gemini-test", contents="abc"))
    with llm_replay.session(tape, mode="replay"):
        again = "".join(c.text for c in client.models.generate_content_stream(model="gemini-test", contents="abc"))
    assert again == live == "echo:abc" and fake_gemini["n"] == 1


async def test_async(tmp_path, fake_gemini):
    client = genai.Client(api_key="AIza-test-not-a-real-key")
    tape = tmp_path / "g.jsonl"
    async with llm_replay.session(tape, mode="record"):
        live = await client.aio.models.generate_content(model="gemini-test", contents="q")
    async with llm_replay.session(tape, mode="replay"):
        again = await client.aio.models.generate_content(model="gemini-test", contents="q")
    assert again.text == live.text and fake_gemini["n"] == 1
