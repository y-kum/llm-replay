"""A tape: one JSON line per recorded call, indexed by key and by route."""

from __future__ import annotations

import difflib
import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._normalize import canonical
from .redact import Redactor

TAPE_VERSION = 1


@dataclass
class Record:
    key: str
    route: str
    request: Any
    response: Any
    stream: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    recorded_at: str = ""
    v: int = TAPE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Drift:
    """The closest recorded call on the same route, for a request that has no exact match."""

    record: Record
    similarity: float
    diff: str


class Tape:
    def __init__(self, path: str | Path, *, redactor: Redactor | None = None) -> None:
        self.path = Path(path)
        self._redact = redactor or Redactor()
        self._lock = threading.Lock()
        self._by_key: dict[str, list[Record]] = defaultdict(list)
        self._by_route: dict[str, list[Record]] = defaultdict(list)
        self._order: list[Record] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index(Record.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    raise ValueError(f"{self.path}:{n}: not a valid tape line ({e})") from e

    def _index(self, rec: Record) -> None:
        self._by_key[rec.key].append(rec)
        self._by_route[rec.route].append(rec)
        self._order.append(rec)

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[Record]:
        return iter(list(self._order))

    def lookup(self, key: str, occurrence: int = 0) -> Record | None:
        """The ``occurrence``-th recording of ``key``; repeats the last one if the call happens more often."""
        recs = self._by_key.get(key)
        if not recs:
            return None
        return recs[min(occurrence, len(recs) - 1)]

    def append(self, rec: Record) -> None:
        if not rec.recorded_at:
            rec.recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = self._redact(rec.to_json())
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            self._index(rec)

    def reset(self) -> None:
        """Start the tape from empty (used by record mode)."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._by_key.clear()
            self._by_route.clear()
            self._order.clear()

    def nearest(self, route: str, request: Any, *, max_lines: int = 40) -> Drift | None:
        """Closest recorded request on the same route, with a unified diff of the two requests."""
        candidates = self._by_route.get(route)
        if not candidates:
            return None
        new_text = pretty(request)
        best, best_ratio = None, -1.0
        for rec in candidates:
            ratio = difflib.SequenceMatcher(None, pretty(rec.request), new_text, autojunk=False).ratio()
            if ratio > best_ratio:
                best, best_ratio = rec, ratio
        assert best is not None
        lines = list(
            difflib.unified_diff(
                pretty(best.request).splitlines(), new_text.splitlines(), "recorded", "this call", lineterm="", n=1
            )
        )
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
        return Drift(record=best, similarity=round(best_ratio, 3), diff="\n".join(lines))


def pretty(value: Any) -> str:
    return json.dumps(json.loads(canonical(value)), indent=2, sort_keys=True, ensure_ascii=False)
