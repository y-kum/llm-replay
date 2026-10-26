"""Sessions: the record/replay logic every adapter and the generic decorator go through."""

from __future__ import annotations

import contextvars
import enum
import functools
import inspect
import os
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._normalize import make_key, normalize_request, normalize_response
from .redact import Redactor
from .report import PriceTable, Report, load_prices
from .tape import Record, Tape


class Mode(str, enum.Enum):
    RECORD = "record"  # call the model, rewrite the tape from scratch
    REPLAY = "replay"  # never call the model; a call missing from the tape is an error
    AUTO = "auto"  # replay what's on the tape, record what isn't
    OFF = "off"  # do nothing: calls go straight to the model


class ReplayMiss(AssertionError):
    """A call in replay mode has no recording. Subclasses AssertionError so pytest reports a failure."""


@dataclass(frozen=True)
class CallSpec:
    """How to store and rebuild one kind of call."""

    provider: str
    load: Callable[[Any], Any] = lambda d: d  # stored data -> response object
    load_chunk: Callable[[Any], Any] = lambda d: d  # stored chunk -> stream item
    usage: Callable[[Any], dict[str, int]] = lambda d: {}  # stored response/chunk -> {"input": n, "output": n}
    dump: Callable[[Any], Any] = normalize_response  # response/chunk object -> storable data


_current: contextvars.ContextVar[Session | None] = contextvars.ContextVar("llm_replay_session", default=None)


def current() -> Session | None:
    """The active session in this context, if any."""
    return _current.get()


def resolve_mode(mode: str | Mode | None) -> Mode:
    return Mode(mode or os.environ.get("LLM_REPLAY_MODE") or Mode.REPLAY.value)


class Session:
    """Record or replay every supported LLM call made inside ``with session(...)``."""

    def __init__(
        self,
        tape: str | Path | Tape,
        mode: str | Mode | None = None,
        *,
        prices: PriceTable | str | Path | None = None,
        redactor: Redactor | None = None,
        patch_sdks: bool = True,
    ) -> None:
        self.mode = resolve_mode(mode)
        self.tape = tape if isinstance(tape, Tape) else Tape(tape, redactor=redactor)
        self.prices: PriceTable = prices if isinstance(prices, dict) else load_prices(prices)
        self.report = Report(tape=str(self.tape.path), mode=self.mode.value)
        self._patch_sdks = patch_sdks
        self._occurrence: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._token: contextvars.Token | None = None
        if self.mode is Mode.RECORD:
            self.tape.reset()

    # -- context management ---------------------------------------------------------------------------------------
    def __enter__(self) -> Session:
        if self._patch_sdks:
            from .adapters import install

            install()
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        if self._patch_sdks:
            from .adapters import uninstall

            uninstall()

    async def __aenter__(self) -> Session:
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)

    # -- core -------------------------------------------------------------------------------------------------------
    def _find(self, route: str, request: Any) -> tuple[str, Record | None]:
        key = make_key(route, request)
        with self._lock:
            occ = self._occurrence[key]
            self._occurrence[key] += 1
        if self.mode is Mode.RECORD:
            return key, None
        return key, self.tape.lookup(key, occ)

    def _raise_miss(self, route: str, key: str, request: Any) -> None:
        drift = self.tape.nearest(route, request)
        m = self.report.add_miss(route, key, drift)
        msg = f"llm-replay: no recording for {route} (key {key}) in {self.tape.path}."
        if drift is None:
            msg += " Nothing on this route was recorded; run once with mode=record or auto."
        else:
            msg += f" Closest recorded call is {m.similarity:.0%} similar; the request changed:\n{drift.diff}"
        raise ReplayMiss(msg)

    def _save(self, spec: CallSpec, key: str, route: str, request: Any, data: Any, stream: bool, t0: float) -> None:
        if stream:
            usage = _stream_usage(spec, data["chunks"])
        else:
            usage = _safe_usage(spec, data)
        rec = Record(
            key=key,
            route=route,
            request=request,
            response=data,
            stream=stream,
            usage=usage,
            latency_s=round(time.perf_counter() - t0, 4),
        )
        self.tape.append(rec)
        self.report.add_recorded(rec, self.prices)

    def _replay(self, spec: CallSpec, rec: Record, is_async: bool) -> Any:
        self.report.add_hit(rec, self.prices)
        if rec.stream:
            items = [spec.load_chunk(c) for c in rec.response["chunks"]]
            return AsyncReplayStream(items) if is_async else ReplayStream(items)
        return spec.load(rec.response)

    def call(self, spec: CallSpec, route: str, request: Any, live: Callable[[], Any], *, stream: bool = False) -> Any:
        key, rec = self._find(route, request)
        if rec is not None:
            return self._replay(spec, rec, is_async=False)
        if self.mode is Mode.REPLAY:
            self._raise_miss(route, key, request)
        t0 = time.perf_counter()
        resp = live()
        if stream:
            return RecordingStream(
                resp, spec, lambda chunks: self._save(spec, key, route, request, {"chunks": chunks}, True, t0)
            )
        self._save(spec, key, route, request, spec.dump(resp), False, t0)
        return resp

    async def acall(
        self, spec: CallSpec, route: str, request: Any, live: Callable[[], Awaitable[Any]], *, stream: bool = False
    ) -> Any:
        key, rec = self._find(route, request)
        if rec is not None:
            return self._replay(spec, rec, is_async=True)
        if self.mode is Mode.REPLAY:
            self._raise_miss(route, key, request)
        t0 = time.perf_counter()
        resp = live()
        if inspect.isawaitable(resp):
            resp = await resp
        if stream:
            return AsyncRecordingStream(
                resp, spec, lambda chunks: self._save(spec, key, route, request, {"chunks": chunks}, True, t0)
            )
        self._save(spec, key, route, request, spec.dump(resp), False, t0)
        return resp


def session(
    tape: str | Path | Tape,
    mode: str | Mode | None = None,
    *,
    prices: PriceTable | str | Path | None = None,
    redactor: Redactor | None = None,
) -> Session:
    """``with llm_replay.session("tapes/x.jsonl", mode="replay"): ...``

    ``mode`` defaults to the ``LLM_REPLAY_MODE`` environment variable, then ``replay``.
    """
    return Session(tape, mode, prices=prices, redactor=redactor)


# -- usage helpers ----------------------------------------------------------------------------------------------------
def _safe_usage(spec: CallSpec, data: Any) -> dict[str, int]:
    try:
        u = spec.usage(data) or {}
    except Exception:  # usage is informational; never fail a call over it
        return {}
    return {k: int(v) for k, v in u.items() if isinstance(v, (int, float)) and v}


def _stream_usage(spec: CallSpec, chunks: list[Any]) -> dict[str, int]:
    """Streams report usage in different chunks (start, delta, final); keep the largest value of each field."""
    total: dict[str, int] = {}
    for c in chunks:
        for k, v in _safe_usage(spec, c).items():
            total[k] = max(total.get(k, 0), v)
    return total


# -- streams ----------------------------------------------------------------------------------------------------------
class RecordingStream:
    """Passes a live stream through unchanged and writes the recording once it is consumed or closed."""

    def __init__(self, inner: Any, spec: CallSpec, finish: Callable[[list[Any]], None]) -> None:
        self._inner, self._spec, self._finish = inner, spec, finish
        self._chunks: list[Any] = []
        self._done = False

    def _complete(self) -> None:
        if not self._done:
            self._done = True
            self._finish(self._chunks)

    def __iter__(self) -> RecordingStream:
        return self

    def __next__(self) -> Any:
        if not hasattr(self, "_it"):
            self._it = iter(self._inner)
        try:
            item = next(self._it)
        except StopIteration:
            self._complete()
            raise
        self._chunks.append(self._spec.dump(item))
        return item

    def __enter__(self) -> RecordingStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
        self._complete()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AsyncRecordingStream:
    def __init__(self, inner: Any, spec: CallSpec, finish: Callable[[list[Any]], None]) -> None:
        self._inner, self._spec, self._finish = inner, spec, finish
        self._chunks: list[Any] = []
        self._done = False

    def _complete(self) -> None:
        if not self._done:
            self._done = True
            self._finish(self._chunks)

    def __aiter__(self) -> AsyncRecordingStream:
        return self

    async def __anext__(self) -> Any:
        if not hasattr(self, "_it"):
            self._it = self._inner.__aiter__()
        try:
            item = await self._it.__anext__()
        except StopAsyncIteration:
            self._complete()
            raise
        self._chunks.append(self._spec.dump(item))
        return item

    async def __aenter__(self) -> AsyncRecordingStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        close = getattr(self._inner, "close", None) or getattr(self._inner, "aclose", None)
        if callable(close):
            res = close()
            if inspect.isawaitable(res):
                await res
        self._complete()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ReplayStream:
    """A recorded stream played back: iterable, closable, usable as a context manager."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self.response = None

    def __iter__(self):
        return iter(self._items)

    def __enter__(self) -> ReplayStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:
        return None


class AsyncReplayStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self.response = None

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item

        return gen()

    async def __aenter__(self) -> AsyncReplayStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def close(self) -> None:
        return None


# -- generic decorator -------------------------------------------------------------------------------------------------
def replayable(
    name: str,
    *,
    dump: Callable[[Any], Any] = normalize_response,
    load: Callable[[Any], Any] = lambda d: d,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Record/replay any function (e.g. a raw HTTP call to a model server) inside an active session.

    The function's arguments form the key; its return value must be JSON-serialisable (or give ``dump``/``load``).
    Outside a session the function runs normally.
    """
    spec = CallSpec(provider=f"fn.{name}", load=load, dump=dump)
    route = f"fn.{name}:-"

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                s = current()
                if s is None or s.mode is Mode.OFF:
                    return await fn(*args, **kwargs)
                return await s.acall(spec, route, normalize_request(kwargs, args), lambda: fn(*args, **kwargs))

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            s = current()
            if s is None or s.mode is Mode.OFF:
                return fn(*args, **kwargs)
            return s.call(spec, route, normalize_request(kwargs, args), lambda: fn(*args, **kwargs))

        return wrapper

    return deco
