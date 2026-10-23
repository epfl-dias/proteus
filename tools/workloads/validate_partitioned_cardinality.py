#!/usr/bin/env python3
"""Validate a prepared dataset against a logical reference dataset."""

import argparse
import json
from pathlib import Path

from materialize_logical_columns import (
    block_offsets,
    catalog_columns,
    selected_column_names,
)


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


def validate_data_file(dataset, base_name, metadata):
    data_file = metadata.get("data_file")
    if not data_file:
        raise ValueError(f"{base_name}: metadata has no data_file")
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = dataset / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"{base_name}: missing data file {data_path}")

    sizes = metadata.get("block_sizes")
    if sizes is None:
        raise ValueError(f"{base_name}: metadata has no block_sizes")
    offsets = block_offsets(metadata)
    if len(offsets) != len(sizes):
        raise ValueError(
            f"{base_name}: block_offsets and block_sizes have different lengths"
        )
    required_bytes = max(
        (offset + size for offset, size in zip(offsets, sizes)), default=0
    )
    actual_bytes = data_path.stat().st_size
    if actual_bytes < required_bytes:
        raise ValueError(
            f"{base_name}: data file has {actual_bytes} bytes, "
            f"metadata requires at least {required_bytes}"
        )


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
        validate_data_file(args.candidate_directory, base_name, candidate)
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
