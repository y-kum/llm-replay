"""Turn SDK call arguments and responses into stable, JSON-serialisable data, and key them."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any

# Top-level request arguments that never change what the model answers, or that hold secrets/clients.
VOLATILE_KEYS = frozenset(
    {
        "api_key",
        "extra_headers",
        "extra_query",
        "http_client",
        "idempotency_key",
        "metadata",
        "request_id",
        "timeout",
        "user",
    }
)

# Response fields that carry transport details (headers, dates) rather than the model's answer.
RESPONSE_DROP_KEYS = frozenset({"sdk_http_response"})


class _Skip:
    """Marker for values that should disappear from the normalised form (SDK 'not given' sentinels)."""


SKIP = _Skip()
_SENTINEL_NAMES = {"NotGiven", "Omit"}


def to_jsonable(value: Any, *, drop: frozenset[str] = frozenset()) -> Any:
    """Convert ``value`` to plain JSON types. ``drop`` removes keys at the top level only."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if type(value).__name__ in _SENTINEL_NAMES:
        return SKIP
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest(), "len": len(value)}
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if hasattr(value, "model_dump") and not isinstance(value, type):
        try:
            dumped = value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = value.model_dump()
        return to_jsonable(dumped, drop=drop)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value), drop=drop)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key in drop:
                continue
            j = to_jsonable(v)
            if j is SKIP:
                continue
            out[key] = j
        return out
    if isinstance(value, (list, tuple)):
        return [j for j in (to_jsonable(v) for v in value) if j is not SKIP]
    if isinstance(value, (set, frozenset)):
        items = [j for j in (to_jsonable(v) for v in value) if j is not SKIP]
        return sorted(items, key=canonical)
    if callable(value):
        return {"__callable__": getattr(value, "__qualname__", type(value).__name__)}
    return {"__object__": type(value).__name__}


def canonical(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def make_key(route: str, request: Any) -> str:
    """Stable key for one call: the route (provider:method:model) plus the normalised request."""
    digest = hashlib.sha256(f"{route}\n{canonical(request)}".encode()).hexdigest()
    return digest[:24]


def normalize_request(kwargs: dict[str, Any], args: tuple[Any, ...] = ()) -> dict[str, Any]:
    req = to_jsonable(dict(kwargs), drop=VOLATILE_KEYS)
    if args:
        req["__args__"] = to_jsonable(list(args))
    return req


def normalize_response(resp: Any) -> Any:
    return to_jsonable(resp, drop=RESPONSE_DROP_KEYS)
