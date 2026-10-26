"""Compare two tapes: which answers changed, which prompts changed, what was added or removed."""

from __future__ import annotations

import difflib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ._normalize import canonical
from .tape import Record, Tape, pretty


def response_text(rec: Record) -> str:
    """The human-readable answer in a recording, for known providers; canonical JSON otherwise."""
    r = rec.response
    if rec.stream and isinstance(r, dict):
        return "".join(_chunk_text(c) for c in r.get("chunks", []))
    texts: list[str] = []
    if isinstance(r, dict):
        for ch in r.get("choices") or []:  # OpenAI chat
            msg = ch.get("message") or {}
            if msg.get("content"):
                texts.append(str(msg["content"]))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                texts.append(f"[tool {fn.get('name')}] {fn.get('arguments')}")
        for block in r.get("content") or []:  # Anthropic
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    texts.append(f"[tool {block.get('name')}] {canonical(block.get('input'))}")
        for cand in r.get("candidates") or []:  # Gemini
            for part in (cand.get("content") or {}).get("parts") or []:
                if part.get("text"):
                    texts.append(part["text"])
                if part.get("function_call"):
                    texts.append(f"[tool] {canonical(part['function_call'])}")
        for item in r.get("output") or []:  # OpenAI Responses
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("text"):
                    texts.append(c["text"])
    return "\n".join(texts) if texts else canonical(r)


def _chunk_text(c: Any) -> str:
    if not isinstance(c, dict):
        return ""
    out = ""
    for ch in c.get("choices") or []:
        out += ((ch.get("delta") or {}).get("content")) or ""
    delta = c.get("delta") or {}
    if isinstance(delta, dict) and delta.get("type") == "text_delta":
        out += delta.get("text", "")
    if c.get("type") == "response.output_text.delta":
        out += c.get("delta") or ""
    for cand in c.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            out += part.get("text") or ""
    return out


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


@dataclass
class Change:
    route: str
    key: str
    kind: str  # "answer" | "prompt"
    diff: str
    json_fields: list[str] = field(default_factory=list)


@dataclass
class TapeDiff:
    same: int = 0
    changed: list[Change] = field(default_factory=list)
    added: list[Record] = field(default_factory=list)
    removed: list[Record] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.changed or self.added or self.removed)

    def summary(self) -> str:
        answers = sum(1 for c in self.changed if c.kind == "answer")
        prompts = sum(1 for c in self.changed if c.kind == "prompt")
        return (
            f"{self.same} unchanged, {answers} answers changed, {prompts} prompts changed, "
            f"{len(self.added)} added, {len(self.removed)} removed"
        )


def _unified(a: str, b: str, max_lines: int = 40) -> str:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "old", "new", lineterm="", n=1))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def _json_field_changes(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out += _json_field_changes(a[k], b[k], p)
        return out
    return [] if a == b else [prefix or "$"]


def diff_tapes(old: Tape, new: Tape) -> TapeDiff:
    result = TapeDiff()
    old_keys = {r.key: r for r in old}
    new_keys = {r.key: r for r in new}
    for key, n in new_keys.items():
        o = old_keys.get(key)
        if o is None:
            continue
        ta, tb = response_text(o), response_text(n)
        if ta == tb:
            result.same += 1
            continue
        ja, jb = _try_json(ta), _try_json(tb)
        fields = _json_field_changes(ja, jb) if ja is not None and jb is not None else []
        result.changed.append(Change(n.route, key, "answer", _unified(ta, tb), fields))

    # Unmatched keys: pair them up by route, in order, as "the prompt changed" when both sides have one.
    only_old: dict[str, list[Record]] = defaultdict(list)
    only_new: dict[str, list[Record]] = defaultdict(list)
    for r in old:
        if r.key not in new_keys:
            only_old[r.route].append(r)
    for r in new:
        if r.key not in old_keys:
            only_new[r.route].append(r)
    for route in sorted(set(only_old) | set(only_new)):
        olds, news = only_old.get(route, []), only_new.get(route, [])
        for o, n in zip(olds, news, strict=False):  # unequal lengths are expected
            result.changed.append(Change(route, n.key, "prompt", _unified(pretty(o.request), pretty(n.request))))
        result.removed += olds[len(news) :]
        result.added += news[len(olds) :]
    return result
