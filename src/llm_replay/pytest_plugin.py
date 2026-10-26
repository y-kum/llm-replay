"""pytest integration.

    @pytest.mark.llm_replay                 # tape: <tape dir>/<module>__<test>.jsonl
    @pytest.mark.llm_replay("search.jsonl") # explicit tape name (shared between tests is fine)
    def test_answer(): ...

Options: ``--llm-mode {record,replay,auto,off}`` (else ``LLM_REPLAY_MODE``, else ini ``llm_replay_mode``, else
replay) · ``--llm-tape-dir`` (else ini ``llm_replay_tape_dir``, else ``tests/tapes``) · ``--llm-prices FILE``.
A JSON report of the whole run is written to ``.llm-replay/report.json`` for ``llm-replay check``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from .report import Report, load_prices
from .session import Mode, Session

_REPORT_KEY = pytest.StashKey[Report]()


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("llm-replay")
    g.addoption("--llm-mode", choices=[m.value for m in Mode], default=None, help="record, replay, auto or off")
    g.addoption("--llm-tape-dir", default=None, help="directory for tapes (default tests/tapes)")
    g.addoption("--llm-prices", default=None, help="JSON price table for cost estimates")
    g.addoption("--llm-report", default=".llm-replay/report.json", help="where to write the run report")
    parser.addini("llm_replay_mode", "default llm-replay mode", default="replay")
    parser.addini("llm_replay_tape_dir", "default tape directory", default="tests/tapes")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "llm_replay(tape=None): record/replay LLM calls made by this test")
    config.stash[_REPORT_KEY] = Report(mode=_mode(config).value)


def _mode(config: pytest.Config) -> Mode:
    return Mode(config.getoption("--llm-mode") or os.environ.get("LLM_REPLAY_MODE") or config.getini("llm_replay_mode"))


def _tape_path(item: pytest.Item, name: str | None) -> Path:
    base = Path(item.config.getoption("--llm-tape-dir") or item.config.getini("llm_replay_tape_dir"))
    if not base.is_absolute():
        base = Path(item.config.rootpath) / base
    if name:
        return base / name
    module = Path(str(item.fspath)).stem
    test = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.name)
    return base / f"{module}__{test}.jsonl"


def _open_session(item: pytest.Item, name: str | None) -> Session:
    config = item.config
    prices = load_prices(config.getoption("--llm-prices"))
    return Session(_tape_path(item, name), _mode(config), prices=prices)


@pytest.fixture(autouse=True)
def _llm_replay_marker(request: pytest.FixtureRequest):
    marker = request.node.get_closest_marker("llm_replay")
    if marker is None:
        yield None
        return
    name = marker.args[0] if marker.args else marker.kwargs.get("tape")
    s = _open_session(request.node, name)
    with s:
        try:
            yield s
        finally:
            request.config.stash[_REPORT_KEY].merge(s.report)


@pytest.fixture
def llm_session(request: pytest.FixtureRequest, _llm_replay_marker):
    """The active session (from the marker), or a new one on the default tape for this test."""
    if _llm_replay_marker is not None:
        yield _llm_replay_marker
        return
    s = _open_session(request.node, None)
    with s:
        try:
            yield s
        finally:
            request.config.stash[_REPORT_KEY].merge(s.report)


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    report = config.stash.get(_REPORT_KEY, None)
    if report is None or not (report.hits or report.recorded or report.misses):
        return
    path = Path(config.getoption("--llm-report"))
    if not path.is_absolute():
        path = Path(config.rootpath) / path
    report.write(path)
    terminalreporter.write_sep("-", "llm-replay")
    terminalreporter.write_line(report.summary())
    if report.misses:
        terminalreporter.write_line(report.miss_details())
    terminalreporter.write_line(f"report: {path}")
