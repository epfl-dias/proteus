#!/usr/bin/env python3
"""Aggregate raw Nsight Compute CSV exports for the streaming experiment."""

import csv
import os
import sys
from collections import Counter


BYTE_SCALE = {
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
}
TIME_TO_MS = {
    "nsecond": 1e-6,
    "usecond": 1e-3,
    "msecond": 1.0,
    "second": 1e3,
}


def number(value):
    if not value:
        return 0.0
    return float(value.replace(",", ""))


def bytes_value(value, unit):
    base_unit = unit.split("/", 1)[0]
    return number(value) * BYTE_SCALE.get(base_unit, 1.0)


def read_raw(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as source:
        reader = csv.reader(source)
        header = next(reader)
        units = next(reader)
        rows = [dict(zip(header, row)) for row in reader if row]
    return rows, dict(zip(header, units))


def weighted(rows, value_column, weight_column, weight_scale):
    weighted_sum = total_weight = 0.0
    for row in rows:
        weight = number(row.get(weight_column, "")) * weight_scale
        weighted_sum += number(row.get(value_column, "")) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


fields = [
    "case", "profiled_kernel_launches", "profiled_query_executions",
    "kernels_per_query", "gpu_time_ms", "dram_read_GB",
    "dram_write_GB", "global_load_GB", "global_store_GB", "local_load_GB",
    "local_store_GB", "time_weighted_l2_hit_pct",
    "time_weighted_occupancy_pct", "max_registers_per_thread",
    "max_shared_mem_per_block_bytes", "raw_csv",
]
writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
writer.writeheader()

for path in sys.argv[1:]:
    rows, units = read_raw(path)
    case_name = os.path.basename(path)
    if case_name.endswith(".raw.csv"):
        case_name = case_name[:-len(".raw.csv")]
    kernel_counts = Counter(row.get("Kernel Name", "") for row in rows)
    query_executions = min(kernel_counts.values()) if kernel_counts else 1
    kernels_per_query = len(rows) / query_executions
    duration_col = "gpu__time_duration.sum"
    duration_scale = TIME_TO_MS.get(units.get(duration_col, ""), 1.0)

    def sum_bytes(column):
        return sum(
            bytes_value(row.get(column, ""), units.get(column, ""))
            for row in rows
        ) / query_executions

    writer.writerow({
        "case": case_name,
        "profiled_kernel_launches": len(rows),
        "profiled_query_executions": query_executions,
        "kernels_per_query": f"{kernels_per_query:.0f}",
        "gpu_time_ms": f"{sum(number(row.get(duration_col, '')) * duration_scale for row in rows) / query_executions:.6f}",
        "dram_read_GB": f"{sum_bytes('dram__bytes_read.sum') / 1e9:.6f}",
        "dram_write_GB": f"{sum_bytes('dram__bytes_write.sum') / 1e9:.6f}",
        "global_load_GB": f"{sum_bytes('l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum') / 1e9:.6f}",
        "global_store_GB": f"{sum_bytes('l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum') / 1e9:.6f}",
        "local_load_GB": f"{sum_bytes('l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum') / 1e9:.6f}",
        "local_store_GB": f"{sum_bytes('l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum') / 1e9:.6f}",
        "time_weighted_l2_hit_pct": f"{weighted(rows, 'lts__t_sector_hit_rate.pct', duration_col, duration_scale):.3f}",
        "time_weighted_occupancy_pct": f"{weighted(rows, 'sm__warps_active.avg.pct_of_peak_sustained_active', duration_col, duration_scale):.3f}",
        "max_registers_per_thread": f"{max((number(row.get('launch__registers_per_thread', '')) for row in rows), default=0):.0f}",
        "max_shared_mem_per_block_bytes": f"{max((bytes_value(row.get('launch__shared_mem_per_block', ''), units.get('launch__shared_mem_per_block', '')) for row in rows), default=0):.0f}",
        "raw_csv": path,
    })
