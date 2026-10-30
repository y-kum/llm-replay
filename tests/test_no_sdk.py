"""A bare `pip install` has no dependencies, so sessions must work with no SDK and no pydantic present."""

from __future__ import annotations

import subprocess
import sys

SCRIPT = """
import sys
for name in ("pydantic", "openai", "anthropic", "google", "google.genai"):
    sys.modules[name] = None  # makes `import name` raise ImportError, as if it were not installed

import llm_replay

calls = []


@llm_replay.replayable("ask")
def ask(q):
    calls.append(q)
    return {"answer": q.upper()}


with llm_replay.session(sys.argv[1], mode="record"):
    first = ask("hello")
with llm_replay.session(sys.argv[1], mode="replay"):
    second = ask("hello")
assert first == second == {"answer": "HELLO"}
assert calls == ["hello"]
"""


def test_session_works_without_any_sdk(tmp_path):
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "t.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
