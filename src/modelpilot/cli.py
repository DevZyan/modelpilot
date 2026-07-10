"""Command-line interface: ``modelpilot data.csv --target col``.

Point it at a CSV, get the profile, patterns and ranked recommendations in the
terminal — and optionally an HTML/JSON report, a cross-validated leaderboard, or
a feature-importance explanation of the top pick.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .report import analyze


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelpilot",
        description="Profile a dataset and get told which model will perform best.",
    )
    p.add_argument("data", help="path to a CSV file")
    p.add_argument("-t", "--target", required=True, help="name of the target column")
    p.add_argument("--task", choices=["classification", "regression"],
                   help="force the task type (inferred by default)")
    p.add_argument("-k", "--top", type=int, default=5, help="how many models to rank")
    p.add_argument("-b", "--benchmark", action="store_true",
                   help="cross-validate the shortlist and print a leaderboard")
    p.add_argument("--cv", type=int, default=5, help="CV folds for --benchmark")
    p.add_argument("--explain", action="store_true",
                   help="fit the top pick and list its most influential features")
    p.add_argument("--html", metavar="PATH", help="write a self-contained HTML report")
    p.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    p.add_argument("--no-weights", action="store_true",
                   help="use the raw heuristic instead of the bundled calibration")
    p.add_argument("--no-probe", action="store_true",
                   help="skip the (slower) linear-vs-tree nonlinearity probe")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"error: file not found: {args.data}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"error: could not read {args.data}: {exc}", file=sys.stderr)
        return 2

    if args.target not in df.columns:
        print(f"error: target column {args.target!r} not in {list(df.columns)}",
              file=sys.stderr)
        return 2

    report = analyze(
        df,
        target=args.target,
        task=args.task,
        probe_nonlinearity=not args.no_probe,
        weights=False if args.no_weights else None,
    )

    report.summary()
    if args.benchmark:
        print()
        report.leaderboard_summary(top_k=args.top, cv=args.cv)
    if args.explain:
        print()
        report.explain().summary()

    if args.html:
        report.save_html(args.html, include_leaderboard=args.benchmark,
                         top_k=args.top, cv=args.cv)
        print(f"\nHTML report written to {args.html}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"JSON report written to {args.json}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
