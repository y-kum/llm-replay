"""Patch SDK methods while a session is active. Each SDK is optional: missing ones are skipped.

Covered (sync and async):
- OpenAI and compatible clients (Groq, Together, local servers): ``chat.completions.create``, ``responses.create``
- Anthropic: ``messages.create``
- Google ``google-genai`` (Gemini, Vertex AI): ``models.generate_content``, ``models.generate_content_stream``

Not covered yet: helpers that bypass ``create`` (OpenAI ``.parse()`` / ``.stream()``, Anthropic ``.stream()``).
Use the ``create(..., stream=True)`` form, or wrap your own function with ``@llm_replay.replayable``.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from ._normalize import normalize_request
from .session import CallSpec, Mode, current

_lock = threading.Lock()
_refcount = 0
_originals: list[tuple[type, str, Any]] = []


def _model(kwargs: dict[str, Any]) -> str:
    m = kwargs.get("model")
    return str(m) if m is not None else "-"


def _get(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _wrap(orig: Callable[..., Any], spec: CallSpec, is_stream: Callable[[dict[str, Any]], bool], is_async: bool):
    if is_async:

        @functools.wraps(orig)
        async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            s = current()
            if s is None or s.mode is Mode.OFF:
                return await orig(self, *args, **kwargs)
            route = f"{spec.provider}:{_model(kwargs)}"
            return await s.acall(
                spec,
                route,
                normalize_request(kwargs, args),
                lambda: orig(self, *args, **kwargs),
                stream=is_stream(kwargs),
            )

        awrapper.__llm_replay_original__ = orig  # type: ignore[attr-defined]
        return awrapper

    @functools.wraps(orig)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        s = current()
        if s is None or s.mode is Mode.OFF:
            return orig(self, *args, **kwargs)
        route = f"{spec.provider}:{_model(kwargs)}"
        return s.call(
            spec, route, normalize_request(kwargs, args), lambda: orig(self, *args, **kwargs), stream=is_stream(kwargs)
        )

    wrapper.__llm_replay_original__ = orig  # type: ignore[attr-defined]
    return wrapper


def _lenient(model: Any) -> Callable[[Any], Any]:
    """Rebuild a response like the SDK does: compatible providers send values outside the strict schema
    (Groq's service_tier="on_demand"), which live calls accept and replay must accept too."""

    def load(data: Any) -> Any:
        try:
            return model.model_validate(data)
        except ValidationError:
            return model.construct(**data)

    return load


def _lenient_event(adapter: Any, event_type: Any, construct_type: Callable[..., Any]) -> Callable[[Any], Any]:
    def load(data: Any) -> Any:
        try:
            return adapter.validate_python(data)
        except ValidationError:
            return construct_type(type_=event_type, value=data)

    return load


def _stream_flag(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("stream"))


def _always(_: dict[str, Any]) -> bool:
    return True


def _never(_: dict[str, Any]) -> bool:
    return False


# -- OpenAI ------------------------------------------------------------------------------------------------------------
def _openai_targets() -> list[tuple[type, str, Callable[[Any], Any]]]:
    try:
        from openai._models import construct_type as openai_construct
        from openai.resources.chat.completions import AsyncCompletions, Completions
        from openai.types.chat import ChatCompletion, ChatCompletionChunk
        from pydantic import TypeAdapter
    except ImportError:
        return []

    def chat_usage(d: Any) -> dict[str, int]:
        u = _get(d, "usage") or {}
        return {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}

    chat = CallSpec(
        provider="openai.chat",
        load=_lenient(ChatCompletion),
        load_chunk=_lenient(ChatCompletionChunk),
        usage=chat_usage,
    )
    targets = [
        (Completions, "create", lambda o: _wrap(o, chat, _stream_flag, False)),
        (AsyncCompletions, "create", lambda o: _wrap(o, chat, _stream_flag, True)),
    ]
    try:
        from openai.resources.responses import AsyncResponses, Responses
        from openai.types.responses import Response, ResponseStreamEvent

        event = TypeAdapter(ResponseStreamEvent)

        def resp_usage(d: Any) -> dict[str, int]:
            u = _get(d, "usage") or _get(d, "response", "usage") or {}
            return {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}

        responses = CallSpec(
            provider="openai.responses",
            load=_lenient(Response),
            load_chunk=_lenient_event(event, ResponseStreamEvent, openai_construct),
            usage=resp_usage,
        )
        targets += [
            (Responses, "create", lambda o: _wrap(o, responses, _stream_flag, False)),
            (AsyncResponses, "create", lambda o: _wrap(o, responses, _stream_flag, True)),
        ]
    except ImportError:
        pass
    return targets


# -- Anthropic ---------------------------------------------------------------------------------------------------------
def _anthropic_targets() -> list[tuple[type, str, Callable[[Any], Any]]]:
    try:
        from anthropic._models import construct_type as anthropic_construct
        from anthropic.resources.messages import AsyncMessages, Messages
        from anthropic.types import Message, RawMessageStreamEvent
        from pydantic import TypeAdapter
    except ImportError:
        return []
    event = TypeAdapter(RawMessageStreamEvent)

    def usage(d: Any) -> dict[str, int]:
        u = _get(d, "usage") or _get(d, "message", "usage") or {}
        return {"input": u.get("input_tokens") or 0, "output": u.get("output_tokens") or 0}

    spec = CallSpec(
        provider="anthropic.messages",
        load=_lenient(Message),
        load_chunk=_lenient_event(event, RawMessageStreamEvent, anthropic_construct),
        usage=usage,
    )
    return [
        (Messages, "create", lambda o: _wrap(o, spec, _stream_flag, False)),
        (AsyncMessages, "create", lambda o: _wrap(o, spec, _stream_flag, True)),
    ]


# -- Google Gemini (google-genai) --------------------------------------------------------------------------------------
def _gemini_targets() -> list[tuple[type, str, Callable[[Any], Any]]]:
    try:
        from google.genai import types
        from google.genai.models import AsyncModels, Models
    except ImportError:
        return []

    def usage(d: Any) -> dict[str, int]:
        u = _get(d, "usage_metadata") or {}
        return {"input": u.get("prompt_token_count") or 0, "output": u.get("candidates_token_count") or 0}

    load = types.GenerateContentResponse.model_validate
    spec = CallSpec(provider="gemini", load=load, load_chunk=load, usage=usage)
    return [
        (Models, "generate_content", lambda o: _wrap(o, spec, _never, False)),
        (AsyncModels, "generate_content", lambda o: _wrap(o, spec, _never, True)),
        (Models, "generate_content_stream", lambda o: _wrap(o, spec, _always, False)),
        (AsyncModels, "generate_content_stream", lambda o: _wrap(o, spec, _always, True)),
    ]


def _targets() -> list[tuple[type, str, Callable[[Any], Any]]]:
    return _openai_targets() + _anthropic_targets() + _gemini_targets()


def install() -> None:
    """Patch every installed SDK. Reference-counted, so nested sessions are fine."""
    global _refcount
    with _lock:
        _refcount += 1
        if _refcount > 1:
            return
        for cls, attr, factory in _targets():
            orig = cls.__dict__.get(attr)
            if orig is None:
                continue
            _originals.append((cls, attr, orig))
            setattr(cls, attr, factory(orig))


def uninstall() -> None:
    global _refcount
    with _lock:
        _refcount = max(0, _refcount - 1)
        if _refcount:
            return
        while _originals:
            cls, attr, orig = _originals.pop()
            setattr(cls, attr, orig)


def installed() -> bool:
    return _refcount > 0
