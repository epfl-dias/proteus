#!/usr/bin/env python3
"""Materialize logical column bytes from a partitioned Proteus dataset.

Proteus data files are 4 KiB aligned on disk. Repartitioning those files by
physical file size turns alignment padding into rows. This helper reconstructs
flat input columns using the logical block ranges recorded in metadata.
"""

import argparse
import json
import shutil
from pathlib import Path


ALIGNMENT = 4096
ELEMENT_WIDTH = 4


def catalog_columns(catalog_path):
    catalog = json.loads(catalog_path.read_text())
    columns = []
    for relation in catalog.values():
        relation_file = Path(relation["path"]).name
        # The published queries scan the dictionary-encoded base relations.
        # Auxiliary string-normalization relations reuse several column names
        # but are not fixed-width query inputs for this workflow.
        if Path(relation_file).stem.endswith("_sn"):
            continue
        attributes = relation["type"]["inner"]["attributes"]
        columns.extend((relation_file, attr["attrName"]) for attr in attributes)
    return columns


def selected_column_names(path):
    if path is None:
        return None
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    if not names:
        raise ValueError(f"column manifest is empty: {path}")
    return names


def block_offsets(metadata):
    if "block_offsets" in metadata:
        return metadata["block_offsets"]

    offsets = []
    offset = 0
    for size in metadata["block_sizes"]:
        offsets.append(offset)
        offset += ((size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    return offsets


def materialize_column(input_dir, output_dir, relation_file, column):
    base_name = f"{relation_file}.{column}"
    metadata_path = input_dir / f"{base_name}.metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing column metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    sizes = metadata["block_sizes"]
    offsets = block_offsets(metadata)
    counts = metadata["value_counts"]
    if not (len(sizes) == len(offsets) == len(counts)):
        raise ValueError(f"inconsistent block metadata: {metadata_path}")

    logical_bytes = sum(sizes)
    logical_rows = sum(counts)
    if logical_bytes != logical_rows * ELEMENT_WIDTH:
        raise ValueError(
            f"{base_name} is not a {ELEMENT_WIDTH}-byte fixed-width column: "
            f"{logical_bytes} bytes for {logical_rows} rows"
        )

    source_path = input_dir / metadata["data_file"]
    destination_path = output_dir / base_name
    with source_path.open("rb") as source, destination_path.open("wb") as dest:
        for offset, size in zip(offsets, sizes):
            source.seek(offset)
            block = source.read(size)
            if len(block) != size:
                raise EOFError(
                    f"short read from {source_path}: expected {size} bytes "
                    f"at offset {offset}, got {len(block)}"
                )
            dest.write(block)

    dictionary = input_dir / f"{base_name}.dict"
    if dictionary.is_file():
        shutil.copy2(dictionary, output_dir / dictionary.name)

    print(f"{base_name}: {logical_rows:,} rows, {logical_bytes:,} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--columns-file", type=Path)
    args = parser.parse_args()

    input_dir = args.input_directory.resolve()
    output_dir = args.output_directory.resolve()
    catalog_path = input_dir / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing catalog: {catalog_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(catalog_path, output_dir / "catalog.json")

    selected = selected_column_names(args.columns_file)
    columns = catalog_columns(catalog_path)
    if selected is not None:
        catalog_names = {column for _, column in columns}
        unknown = selected - catalog_names
        if unknown:
            raise ValueError(
                "column manifest contains names absent from the catalog: "
                + ", ".join(sorted(unknown))
            )
        columns = [(relation, column) for relation, column in columns
                   if column in selected]

    for relation_file, column in columns:
        materialize_column(input_dir, output_dir, relation_file, column)


if __name__ == "__main__":
    main()
