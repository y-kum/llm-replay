"""Shared fakes: real SDK clients talking to an in-process httpx transport, so tests never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

pytest_plugins = ["pytester"]

FAKE_OPENAI_KEY = "sk-test-0123456789abcdefghijklmnop"
FAKE_ANTHROPIC_KEY = "sk-ant-test-0123456789abcdefghijklmnop"


class Server:
    """Counts calls and answers with a function of the request body; ``down()`` makes every call fail."""

    def __init__(self, respond: Callable[[dict, int], httpx.Response]) -> None:
        self.respond = respond
        self.calls = 0
        self.is_down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.is_down:
            raise AssertionError(f"network call made while the server is down: {request.url}")
        self.calls += 1
        body = json.loads(request.content or b"{}")
        return self.respond(body, self.calls)

    def down(self) -> None:
        self.is_down = True


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def sse(events: list[tuple[str | None, dict | str]], lib=httpx):
    lines = []
    for name, data in events:
        if name:
            lines.append(f"event: {name}")
        lines.append("data: " + (data if isinstance(data, str) else json.dumps(data)))
        lines.append("")
    return lib.Response(200, content=("\n".join(lines) + "\n").encode(), headers={"content-type": "text/event-stream"})


# -- OpenAI ------------------------------------------------------------------------------------------------------------
def openai_respond(body: dict, n: int) -> httpx.Response:
    text = f"echo:{_last_user_text(body.get('messages', []))}"
    if body.get("stream"):
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1700000000, "model": body["model"]}
        events: list[tuple[str | None, dict | str]] = []
        for i, piece in enumerate([text[:5], text[5:]]):
            delta = {"content": piece} if i else {"role": "assistant", "content": piece}
            events.append((None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}))
        events.append((None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        events.append(
            (None, {**base, "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}})
        )
        events.append((None, "[DONE]"))
        return sse(events)
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{n}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        },
    )


@pytest.fixture
def openai_server():
    return Server(openai_respond)


@pytest.fixture
def openai_client(openai_server):
    import openai

    return openai.OpenAI(
        api_key=FAKE_OPENAI_KEY,
        base_url="http://fake.local/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(openai_server.handler)),
    )


@pytest.fixture
def async_openai_client(openai_server):
    import openai

    async def handler(request: httpx.Request) -> httpx.Response:
        return openai_server.handler(request)

    return openai.AsyncOpenAI(
        api_key=FAKE_OPENAI_KEY,
        base_url="http://fake.local/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


# -- Anthropic ---------------------------------------------------------------------------------------------------------
def anthropic_respond(body: dict, n: int):
    import httpx2  # the Anthropic SDK 2.x uses httpx2 objects

    text = f"echo:{_last_user_text(body.get('messages', []))}"
    msg = {
        "id": f"msg_{n}",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }
    if body.get("stream"):
        return sse(
            lib=httpx2,
            events=[
                ("message_start", {"type": "message_start", "message": msg}),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 6},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ],
        )
    full = {
        **msg,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }
    return httpx2.Response(200, json=full)


@pytest.fixture
def anthropic_server():
    return Server(anthropic_respond)


@pytest.fixture
def anthropic_client(anthropic_server):
    import anthropic
    import httpx2

    return anthropic.Anthropic(
        api_key=FAKE_ANTHROPIC_KEY,
        base_url="http://fake.local",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(anthropic_server.handler)),
    )


@pytest.fixture
def async_anthropic_client(anthropic_server):
    import anthropic
    import httpx2

    async def handler(request):
        return anthropic_server.handler(request)

    return anthropic.AsyncAnthropic(
        api_key=FAKE_ANTHROPIC_KEY,
        base_url="http://fake.local",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
