"""Command line: ``llm-replay stats|diff|check``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .diffing import diff_tapes
from .report import Report, cost_of, load_prices, usd
from .tape import Tape


def _stats(args: argparse.Namespace) -> int:
    prices = load_prices(args.prices)
    tape = Tape(args.tape)
    calls = Counter(r.route for r in tape)
    tokens_in = sum(r.usage.get("input", 0) for r in tape)
    tokens_out = sum(r.usage.get("output", 0) for r in tape)
    cost = sum(cost_of(r.route, r.usage, prices) for r in tape)
    latency = sum(r.latency_s for r in tape)
    print(f"{args.tape}: {len(tape)} recorded calls")
    for route, n in calls.most_common():
        print(f"  {n:>5}  {route}")
    print(f"tokens: {tokens_in:,} in / {tokens_out:,} out")
    print(f"live latency when recorded: {latency:.1f} s total")
    if prices:
        print(f"a live run of this tape would cost about {usd(cost)}")
    return 0


def _diff(args: argparse.Namespace) -> int:
    d = diff_tapes(Tape(args.old), Tape(args.new))
    if args.json:
        print(
            json.dumps(
                {
                    "summary": d.summary(),
                    "changed": [c.__dict__ for c in d.changed],
                    "added": [r.key for r in d.added],
                    "removed": [r.key for r in d.removed],
                },
                indent=2,
            )
        )
    else:
        print(d.summary())
        for c in d.changed:
            what = "answer changed" if c.kind == "answer" else "prompt changed"
            print(f"\n[{what}] {c.route} key={c.key}")
            if c.json_fields:
                print("  JSON fields: " + ", ".join(c.json_fields))
            print("\n".join("  " + ln for ln in c.diff.splitlines()))
        for r in d.added:
            print(f"\n[added] {r.route} key={r.key}")
        for r in d.removed:
            print(f"\n[removed] {r.route} key={r.key}")
    return 1 if (args.fail_on_change and not d.clean) else 0


def _check(args: argparse.Namespace) -> int:
    path = Path(args.report)
    if not path.exists():
        print(f"llm-replay: no report at {path}; did the test run use the llm-replay pytest plugin?", file=sys.stderr)
        return 2
    report = Report.from_dict(json.loads(path.read_text(encoding="utf-8")))
    print(report.summary())
    if report.misses:
        print("\nCalls with no recording (re-record with --llm-mode=auto, review the diff, commit the tape):")
        print(report.miss_details())
        return 1
    if args.max_live is not None and report.recorded > args.max_live:
        print(f"\n{report.recorded} live calls were made; the limit is {args.max_live}.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm-replay", description="Inspect and compare llm-replay tapes.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="calls, tokens and cost recorded on a tape")
    s.add_argument("tape")
    s.add_argument("--prices", help="JSON price table: {model: {input: USD/1M, output: USD/1M}}")
    s.set_defaults(fn=_stats)

    d = sub.add_parser("diff", help="compare two tapes (e.g. before and after re-recording)")
    d.add_argument("old")
    d.add_argument("new")
    d.add_argument("--json", action="store_true")
    d.add_argument("--fail-on-change", action="store_true", help="exit 1 if anything differs")
    d.set_defaults(fn=_diff)

    c = sub.add_parser("check", help="fail CI if a test run had calls missing from its tapes")
    c.add_argument("report", nargs="?", default=".llm-replay/report.json")
    c.add_argument("--max-live", type=int, default=None, help="fail if more than N calls went to a live model")
    c.set_defaults(fn=_check)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
