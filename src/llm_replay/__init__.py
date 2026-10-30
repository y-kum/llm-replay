"""llm-replay: record LLM calls once, replay them in tests and CI.

    import llm_replay

    with llm_replay.session("tests/tapes/summarise.jsonl", mode="auto"):
        answer = client.chat.completions.create(model="gpt-4o-mini", messages=[...])

Modes: ``record`` (live, rewrite tape) · ``replay`` (tape only; a missing call fails) · ``auto`` (replay, record
what's missing) · ``off``. Default: ``LLM_REPLAY_MODE`` env var, else ``replay``.
"""

from .redact import Redactor
from .report import Report
from .session import Mode, ReplayMiss, Session, current, replayable, session
from .tape import Record, Tape

__all__ = [
    "Mode",
    "Record",
    "Redactor",
    "ReplayMiss",
    "Report",
    "Session",
    "Tape",
    "current",
    "replayable",
    "session",
]
__version__ = "0.1.2"
