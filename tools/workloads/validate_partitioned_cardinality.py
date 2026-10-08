#!/usr/bin/env python3
"""Validate a prepared dataset against a logical reference dataset."""

import argparse
import json
from pathlib import Path

from materialize_logical_columns import catalog_columns, selected_column_names


def metadata_path(dataset, base_name):
    paths = [
        dataset / f"{base_name}.metadata.json",
        dataset / f"{base_name}_0_1.metadata.json",
    ]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing metadata for {base_name} in {dataset}")


def read_metadata(dataset, base_name):
    return json.loads(metadata_path(dataset, base_name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--candidate-directory", required=True, type=Path)
    parser.add_argument("--columns-file", type=Path)
    parser.add_argument("--compression-algorithm")
    parser.add_argument("--tile-size", type=int)
    args = parser.parse_args()

    catalog_path = args.reference_directory / "catalog.json"
    columns = catalog_columns(catalog_path)
    selected = selected_column_names(args.columns_file)
    if selected is not None:
        columns = [(relation, column) for relation, column in columns
                   if column in selected]

    checked = 0
    for relation_file, column in columns:
        base_name = f"{relation_file}.{column}"
        reference = read_metadata(args.reference_directory, base_name)
        candidate = read_metadata(args.candidate_directory, base_name)
        expected_rows = sum(reference["value_counts"])
        actual_rows = sum(candidate["value_counts"])
        if actual_rows != expected_rows:
            raise ValueError(
                f"{base_name}: expected {expected_rows} rows, got {actual_rows}"
            )
        if (args.compression_algorithm is not None and
                candidate.get("compression_algorithm") !=
                args.compression_algorithm):
            raise ValueError(
                f"{base_name}: expected compression_algorithm="
                f"{args.compression_algorithm}, got "
                f"{candidate.get('compression_algorithm')}"
            )
        if (args.tile_size is not None and
                candidate.get("tile_size") != args.tile_size):
            raise ValueError(
                f"{base_name}: expected tile_size={args.tile_size}, got "
                f"{candidate.get('tile_size')}"
            )
        checked += 1

    if checked == 0:
        raise ValueError("no columns were checked")
    print(f"cardinality/format audit: PASS ({checked} columns)")


if __name__ == "__main__":
    main()
