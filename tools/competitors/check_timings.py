#!/usr/bin/env python3
"""Validate that competitor timing files contain complete usable measurements."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List


def is_timing(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def read_rows(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream, delimiter=delimiter))
    except (OSError, csv.Error) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def check_heavydb(args: argparse.Namespace) -> List[str]:
    rows = read_rows(args.file, delimiter="\t")
    profile_rows = read_rows(args.profile_file, delimiter="\t")
    queries = args.queries.split()
    expected_runs = set(range(1, args.iterations + 1))
    errors = []

    unexpected = sorted({row.get("query", "") for row in rows} - set(queries))
    if unexpected:
        errors.append(f"unexpected queries: {', '.join(unexpected)}")

    for query in queries:
        query_rows = [row for row in rows if row.get("query") == query]
        runs = []
        for row in query_rows:
            try:
                run = int(row.get("run", ""))
            except ValueError:
                errors.append(f"{query}: invalid run number {row.get('run')!r}")
                continue
            runs.append(run)
            timing = row.get("execution_ms", row.get("ms", ""))
            if not is_timing(timing):
                errors.append(f"{query} run {run}: invalid timing {timing!r}")

        if len(query_rows) != args.iterations:
            errors.append(
                f"{query}: expected {args.iterations} rows, found {len(query_rows)}"
            )
        if set(runs) != expected_runs or len(runs) != len(set(runs)):
            errors.append(
                f"{query}: expected run numbers 1..{args.iterations}, found {runs}"
            )

    preferred_marker = "HEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS"
    preferred_rows = [
        row for row in profile_rows if row.get("marker") == preferred_marker
    ]
    unexpected = sorted(
        {row.get("query", "") for row in preferred_rows} - set(queries)
    )
    if unexpected:
        errors.append(f"unexpected profile queries: {', '.join(unexpected)}")

    for query in queries:
        query_rows = [row for row in preferred_rows if row.get("query") == query]
        runs = []
        for row in query_rows:
            try:
                run = int(row.get("run", ""))
                marker_count = int(row.get("marker_count", ""))
            except ValueError:
                errors.append(f"{query}: invalid preferred-marker row {row!r}")
                continue
            runs.append(run)
            if marker_count <= 0:
                errors.append(
                    f"{query} run {run}: preferred marker count is {marker_count}"
                )
            elapsed = row.get("elapsed_ms_sum", "")
            if not is_timing(elapsed):
                errors.append(
                    f"{query} run {run}: invalid preferred-marker timing {elapsed!r}"
                )

        if len(query_rows) != args.iterations:
            errors.append(
                f"{query}: expected {args.iterations} preferred-marker rows, "
                f"found {len(query_rows)}"
            )
        if set(runs) != expected_runs or len(runs) != len(set(runs)):
            errors.append(
                f"{query}: expected preferred-marker runs 1..{args.iterations}, "
                f"found {runs}"
            )
    return errors


def check_sirius(args: argparse.Namespace) -> List[str]:
    rows = read_rows(args.file)
    # from-line is the number of physical lines present before this invocation
    # appended its rows. Timing CSV fields never contain embedded newlines.
    rows = rows[args.from_line - 1 :]
    errors = []

    if len(rows) != args.iterations:
        errors.append(
            f"{args.query}: expected {args.iterations} rows, found {len(rows)}"
        )

    iterations = []
    for row in rows:
        if row.get("query") != args.query:
            errors.append(f"unexpected query {row.get('query')!r}")
        if args.mode and row.get("mode") != args.mode:
            errors.append(
                f"{args.query}: expected mode {args.mode!r}, found {row.get('mode')!r}"
            )
        try:
            iteration = int(row.get("iteration", ""))
        except ValueError:
            errors.append(
                f"{args.query}: invalid iteration {row.get('iteration')!r}"
            )
            continue
        iterations.append(iteration)
        if row.get("status") != "ok":
            errors.append(
                f"{args.query} iteration {iteration}: status is {row.get('status')!r}"
            )
        for field in ("runtime_s", "execute_ms"):
            if not is_timing(row.get(field, "")):
                errors.append(
                    f"{args.query} iteration {iteration}: invalid {field} "
                    f"{row.get(field)!r}"
                )

    expected_iterations = set(range(1, args.iterations + 1))
    if set(iterations) != expected_iterations or len(iterations) != len(set(iterations)):
        errors.append(
            f"{args.query}: expected iterations 1..{args.iterations}, "
            f"found {iterations}"
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="system", required=True)

    heavydb = subparsers.add_parser("heavydb")
    heavydb.add_argument("--file", type=Path, required=True)
    heavydb.add_argument("--profile-file", type=Path, required=True)
    heavydb.add_argument("--queries", required=True)
    heavydb.add_argument("--iterations", type=int, required=True)

    sirius = subparsers.add_parser("sirius")
    sirius.add_argument("--file", type=Path, required=True)
    sirius.add_argument("--query", required=True)
    sirius.add_argument("--mode", default="")
    sirius.add_argument("--iterations", type=int, required=True)
    sirius.add_argument("--from-line", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        print("timing validation failed: iterations must be positive", file=sys.stderr)
        return 2
    if args.system == "sirius" and args.from_line <= 0:
        print("timing validation failed: from-line must be positive", file=sys.stderr)
        return 2

    try:
        errors = check_heavydb(args) if args.system == "heavydb" else check_sirius(args)
    except ValueError as error:
        errors = [str(error)]

    if errors:
        print(f"timing validation failed for {args.file}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
