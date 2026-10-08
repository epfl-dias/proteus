#!/usr/bin/env python3
"""Combine streaming timings with exact input and staging allocations."""

import csv
import glob
import os
import re
import sys


if len(sys.argv) != 3:
    raise SystemExit("usage: assemble_streaming_csv.py OUT_BASE footprint.csv")

base, footprint_csv = map(os.path.abspath, sys.argv[1:])
footprint = {}
with open(footprint_csv, newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        footprint[(row["workload"], row["query"], row["codec"])] = row

time_re = re.compile(r"\S+ time=([\d.]+|NA) ms result_vs_none=(\S+)")
staging_re = re.compile(r"streaming_staging_bytes=(\d+)")
gmem_re = re.compile(r"ptxas info\s*: (\d+) bytes gmem")

rows = []
pattern = os.path.join(base, "runs", "**", "w*", "progress.log")
for progress_path in sorted(glob.glob(pattern, recursive=True)):
    run_dir = os.path.dirname(progress_path)
    relative = os.path.relpath(run_dir, os.path.join(base, "runs"))
    parts = relative.split(os.sep)
    if len(parts) == 4:
        workload, query, codec, window_name = parts
        tile = 1024
    elif len(parts) == 6:
        workload, query, codec, tile_name, _, window_name = parts
        tile = int(tile_name[1:])
    else:
        raise ValueError(f"unexpected streaming run path: {relative}")
    window = int(window_name[1:])
    log_path = os.path.join(run_dir, f"{codec.lower()}_stream_{query}.log")

    timing = status = ""
    with open(progress_path, encoding="utf-8", errors="replace") as source:
        for line in source:
            match = time_re.match(line.strip())
            if match:
                timing, status = match.groups()

    with open(log_path, encoding="utf-8", errors="replace") as source:
        log_text = source.read()
    staging_allocations = [int(value) for value in staging_re.findall(log_text)]
    gmem_allocations = [int(value) for value in gmem_re.findall(log_text)]

    # Identical scan pipelines can reuse one compiled artifact, so ptxas may
    # report a static allocation size once even when multiple relations use it.
    # The footprint still sums every plan instance; this is only a size check.
    ptxas_match = set(staging_allocations).issubset(gmem_allocations)

    footprint_codec = "RFOR512" if codec == "RFOR" and tile == 512 else codec
    fp = footprint[(workload, query, footprint_codec)]
    compressed_bytes = int(fp["space_bytes"])
    staging_bytes = sum(staging_allocations)
    total_bytes = compressed_bytes + staging_bytes

    plan_tpb = re.search(r"cb2t_block_threads=(\d+)", log_text)
    plan_ipt = re.search(r"cb2t_items_per_thread=(\d+)", log_text)
    rows.append({
        "workload": workload,
        "query": query,
        "codec": codec,
        "mode": "stream",
        "tile": tile,
        "tpb": plan_tpb.group(1) if plan_tpb else "",
        "ipt": plan_ipt.group(1) if plan_ipt else "",
        "streaming_grid_rows": window,
        "time_ms": timing,
        "compressed_bytes": compressed_bytes,
        "staging_bytes": staging_bytes,
        "total_footprint_bytes": total_bytes,
        "compressed_GB": f"{compressed_bytes / 1e9:.6f}",
        "staging_GB": f"{staging_bytes / 1e9:.6f}",
        "total_footprint_GB": f"{total_bytes / 1e9:.6f}",
        "n_scanned_cols": fp["n_scanned_cols"],
        "n_streaming_pipelines": len(staging_allocations),
        "ptxas_staging_match": str(ptxas_match).lower(),
        "result_status": status,
        "log": os.path.relpath(log_path, base),
    })

fieldnames = [
    "workload", "query", "codec", "mode", "tile", "tpb", "ipt",
    "streaming_grid_rows", "time_ms", "compressed_bytes", "staging_bytes",
    "total_footprint_bytes", "compressed_GB", "staging_GB",
    "total_footprint_GB", "n_scanned_cols", "n_streaming_pipelines",
    "ptxas_staging_match", "result_status", "log",
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
