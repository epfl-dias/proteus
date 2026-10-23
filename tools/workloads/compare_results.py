#!/usr/bin/env python3
"""Validate and compare exact Proteus CSV result files."""

import collections
import csv
import itertools
import sys


SCHEMAS = {
    ("ssb", "Q1_1"): 1,
    ("ssb", "Q2_1"): 3,
    ("ssb", "Q3_1"): 4,
    ("ssb", "Q4_1"): 3,
    ("tpch", "Q5"): 2,
    ("tpch", "Q12"): 3,
    ("stack", "Q11"): 1,
    ("stack", "Q14"): 1,
}

# Field indices in the projected CSV. Rows may permute only within an equal
# SQL ORDER BY key. Q5 projects (n_name, revenue), while Q3.1 projects
# (c_nation, s_nation, d_year, revenue).
TIE_KEYS = {("tpch", "Q5"): (1,), ("ssb", "Q3_1"): (2, 3)}


class SchemaError(ValueError):
    """The result does not match the published query's output schema."""


def read_result(workload, query, path):
    expected_fields = SCHEMAS.get((workload, query))
    if expected_fields is None:
        raise SchemaError(f"unsupported query schema: {workload}:{query}")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise SchemaError("empty output")
    for line_number, row in enumerate(rows, 1):
        if len(row) != expected_fields:
            raise SchemaError(
                f"line {line_number}: expected {expected_fields} fields, got {len(row)}"
            )
    return rows


def compare(workload, query, reference, candidate):
    left = read_result(workload, query, reference)
    right = read_result(workload, query, candidate)
    keys = TIE_KEYS.get((workload, query))
    if keys is None:
        return left == right

    def groups(rows):
        return [
            (key, collections.Counter(map(tuple, group)))
            for key, group in itertools.groupby(
                rows, lambda row: tuple(row[index] for index in keys)
            )
        ]

    # Preserve key-group order and duplicate multiplicities. Only SQL-legal
    # permutations within an equal ORDER BY key are ignored.
    return groups(left) == groups(right)


def main(argv):
    if len(argv) == 5 and argv[1] == "--validate":
        read_result(argv[2], argv[3], argv[4])
        return 0
    if len(argv) != 5:
        raise ValueError(
            "usage: compare_results.py WORKLOAD QUERY REFERENCE CANDIDATE"
        )
    if not compare(*argv[1:]):
        print("result mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SchemaError as error:
        print(f"schema error: {error}", file=sys.stderr)
        sys.exit(3)
    except (OSError, ValueError, IndexError, csv.Error) as error:
        print(f"comparison error: {error}", file=sys.stderr)
        sys.exit(2)
