"""What happened in a session: replays, recordings, misses (with prompt drift), tokens and cost."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tape import Drift, Record

PriceTable = dict[str, dict[str, float]]  # model -> {"input": USD per 1M tokens, "output": USD per 1M tokens}


def load_prices(path: str | Path | None) -> PriceTable:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_of(route: str) -> str:
    return route.rsplit(":", 1)[-1]


def cost_of(route: str, usage: dict[str, int], prices: PriceTable) -> float:
    p = prices.get(model_of(route))
    if not p:
        return 0.0
    return usage.get("input", 0) / 1e6 * p.get("input", 0.0) + usage.get("output", 0) / 1e6 * p.get("output", 0.0)


def usd(x: float) -> str:
    return f"${x:.6f}" if x < 0.01 else f"${x:.4f}"


def _zero() -> dict[str, int]:
    return {"input": 0, "output": 0}


@dataclass
class Miss:
    route: str
    key: str
    similarity: float | None = None
    diff: str = ""


@dataclass
class Report:
    tape: str = ""
    mode: str = ""
    hits: int = 0
    recorded: int = 0
    misses: list[Miss] = field(default_factory=list)
    tokens_replayed: dict[str, int] = field(default_factory=_zero)
    tokens_live: dict[str, int] = field(default_factory=_zero)
    cost_replayed: float = 0.0
    cost_live: float = 0.0
    by_route: dict[str, dict[str, int]] = field(default_factory=dict)

    # -- events -------------------------------------------------------------------------------------------------
    def _count(self, route: str, what: str) -> None:
        r = self.by_route.setdefault(route, {"hits": 0, "recorded": 0, "misses": 0})
        r[what] += 1

    def add_hit(self, rec: Record, prices: PriceTable) -> None:
        self.hits += 1
        self._count(rec.route, "hits")
        for k in self.tokens_replayed:
            self.tokens_replayed[k] += rec.usage.get(k, 0)
        self.cost_replayed += cost_of(rec.route, rec.usage, prices)

    def add_recorded(self, rec: Record, prices: PriceTable) -> None:
        self.recorded += 1
        self._count(rec.route, "recorded")
        for k in self.tokens_live:
            self.tokens_live[k] += rec.usage.get(k, 0)
        self.cost_live += cost_of(rec.route, rec.usage, prices)

    def add_miss(self, route: str, key: str, drift: Drift | None) -> Miss:
        m = Miss(route=route, key=key)
        if drift:
            m.similarity, m.diff = drift.similarity, drift.diff
        self.misses.append(m)
        self._count(route, "misses")
        return m

    def merge(self, other: Report) -> None:
        self.hits += other.hits
        self.recorded += other.recorded
        self.misses.extend(other.misses)
        for mine, theirs in ((self.tokens_replayed, other.tokens_replayed), (self.tokens_live, other.tokens_live)):
            for k in mine:
                mine[k] += theirs.get(k, 0)
        self.cost_replayed += other.cost_replayed
        self.cost_live += other.cost_live
        for route, counts in other.by_route.items():
            mine_r = self.by_route.setdefault(route, {"hits": 0, "recorded": 0, "misses": 0})
            for k, v in counts.items():
                mine_r[k] = mine_r.get(k, 0) + v

    # -- output -------------------------------------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return not self.misses

    def to_dict(self) -> dict[str, Any]:
        return {
            "tape": self.tape,
            "mode": self.mode,
            "ok": self.ok,
            "hits": self.hits,
            "recorded": self.recorded,
            "misses": [m.__dict__ for m in self.misses],
            "tokens_replayed": self.tokens_replayed,
            "tokens_live": self.tokens_live,
            "cost_replayed_usd": round(self.cost_replayed, 6),
            "cost_live_usd": round(self.cost_live, 6),
            "by_route": self.by_route,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Report:
        r = cls(tape=d.get("tape", ""), mode=d.get("mode", ""), hits=d.get("hits", 0), recorded=d.get("recorded", 0))
        r.misses = [Miss(**m) for m in d.get("misses", [])]
        r.tokens_replayed = dict(d.get("tokens_replayed", _zero()))
        r.tokens_live = dict(d.get("tokens_live", _zero()))
        r.cost_replayed = d.get("cost_replayed_usd", 0.0)
        r.cost_live = d.get("cost_live_usd", 0.0)
        r.by_route = d.get("by_route", {})
        return r

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        line = f"llm-replay: {self.hits} replayed, {self.recorded} recorded, {len(self.misses)} missing"
        tr, tl = self.tokens_replayed, self.tokens_live
        if tr["input"] or tr["output"]:
            line += f" | {tr['input'] + tr['output']:,} tokens replayed"
            if self.cost_replayed:
                line += f" (~{usd(self.cost_replayed)} not spent)"
        if tl["input"] or tl["output"]:
            line += f" | {tl['input'] + tl['output']:,} tokens live"
            if self.cost_live:
                line += f" (~{usd(self.cost_live)})"
        return line

    def miss_details(self) -> str:
        out = []
        for m in self.misses:
            head = f"- {m.route} key={m.key}"
            if m.similarity is not None:
                head += f" (closest recorded call is {m.similarity:.0%} similar)"
            out.append(head)
            if m.diff:
                out.append("\n".join("    " + ln for ln in m.diff.splitlines()))
        return "\n".join(out)
