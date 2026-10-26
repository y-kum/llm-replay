"""Redact credentials before anything is written to a tape."""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

DEFAULT_PATTERNS = [
    r"sk-ant-[A-Za-z0-9_\-]{16,}",  # Anthropic
    r"sk-[A-Za-z0-9_\-]{16,}",  # OpenAI and compatible
    r"AIza[0-9A-Za-z_\-]{30,}",  # Google API keys
    r"gsk_[A-Za-z0-9]{20,}",  # Groq
    r"hf_[A-Za-z0-9]{20,}",  # Hugging Face
    r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",  # bearer tokens
]


class Redactor:
    def __init__(self, patterns: list[str] | None = None, extra: list[str] | None = None) -> None:
        pats = list(DEFAULT_PATTERNS if patterns is None else patterns) + list(extra or [])
        self._rx = [re.compile(p) for p in pats]

    def __call__(self, text: str) -> str:
        for rx in self._rx:
            text = rx.sub(REDACTED, text)
        return text


NO_REDACTION = Redactor(patterns=[])
