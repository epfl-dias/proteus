#!/usr/bin/env zsh
# Usage: ./PAPER_scripts.sh [MODE]
#
# MODE (default: all):
#   timing   -- run timing benchmarks (NONE / RFOR / FLBP decompression modes), all SSB queries
#   ncu      -- run NCU hardware counter profiling (all SSB queries)
#   nsys     -- run nsys timeline profiling (Q1.1 only)
#   analyze  -- aggregate/extract summaries from existing timing/ncu/nsys outputs
#   deep     -- focused per-kernel NCU/PTXAS profiling for launch-geometry inversions
#   all      -- timing + ncu + nsys + analyze

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RESULTS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${RESULTS_ROOT}/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/cmake-build-debug-new_dias49"

TIMING_DIR="${RESULTS_ROOT}/timing_logs"
NCU_DIR="${RESULTS_ROOT}/ncu"
NSYS_DIR="${RESULTS_ROOT}/nsys"
DEEP_NCU_DIR="${RESULTS_ROOT}/deep_ncu"
ANALYSIS_DIR="${RESULTS_ROOT}/analysis"
ANALYSIS_TIMING_DIR="${ANALYSIS_DIR}/timing"
ANALYSIS_NCU_DIR="${ANALYSIS_DIR}/ncu"
ANALYSIS_NSYS_DIR="${ANALYSIS_DIR}/nsys"
ANALYSIS_EXPLAIN_DIR="${ANALYSIS_DIR}/explanations"
ANALYSIS_DEEP_NCU_DIR="${ANALYSIS_DIR}/deep_ncu"

mkdir -p \
    "$RESULTS_ROOT" \
    "$SCRIPT_DIR" \
    "$TIMING_DIR" \
    "$NCU_DIR" \
    "$NSYS_DIR" \
    "$DEEP_NCU_DIR" \
    "$ANALYSIS_DIR" \
    "$ANALYSIS_TIMING_DIR" \
    "$ANALYSIS_NCU_DIR" \
    "$ANALYSIS_NSYS_DIR" \
    "$ANALYSIS_EXPLAIN_DIR" \
    "$ANALYSIS_DEEP_NCU_DIR"

cd "${BUILD_ROOT}/opt/pelago"

export LD_LIBRARY_PATH=../lib:/scratch/pelago/llvm-14/opt/lib
export PROTEUS_NVPTX_MAX_REGS=48 # global limit for register usage in generated kernels, can be overridden by --max_registers_per_thread
export CUDA_VISIBLE_DEVICES=1

EXECUTABLE="${BUILD_ROOT}/apps/standalones/adm/adm-play/proteusadm-play"
NSYS="/usr/local/cuda-12.4/bin/nsys"
NCU="/usr/local/cuda-12.4/bin/ncu"
NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block"
DEFAULT_DEEP_NCU_METRICS="gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed,sm__sass_thread_inst_executed_op_integer_pred_on.sum.peak_sustained,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"
DEEP_NCU_METRICS="${PAPER_DEEP_NCU_METRICS:-$DEFAULT_DEEP_NCU_METRICS}"
NCU_EXPORTER_TOOL_CANDIDATES=(
    "${SCRIPT_DIR}/my_ncu_exporter.py"
    "${REPO_ROOT}/tools/tracing/PAPER_TRACING/my_ncu_exporter.py"
)
# Full paper runs need CB2T tuple/OID bindings enabled. Several join pipelines
# depend on them during query codegen. Paper automation also suppresses query
# result printing to keep logs and profilers focused on performance signals.
SAFE_CB2T_ARGS=(
    --emit_tuple_oid_bindings=true
    --print_res=false
)

if [[ ! -x "$NCU" ]]; then
    echo "ERROR: required NCU binary not found or not executable: $NCU" >&2
    exit 1
fi

MODE="${1:-all}"
#DEFAULT_ALL_QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
DEFAULT_ALL_QUERIES=(11 21 31 41)
if [[ -n "${PAPER_SSB_QUERIES:-}" ]]; then
    ALL_QUERIES=(${=PAPER_SSB_QUERIES})
else
    ALL_QUERIES=("${DEFAULT_ALL_QUERIES[@]}")
fi
FASTEST_CSV="${ANALYSIS_TIMING_DIR}/ssb_fastest_codec_configs.csv"
CONFIG_SENSITIVITY_CSV="${ANALYSIS_TIMING_DIR}/ssb_config_sensitivity.csv"
CONFIG_SENSITIVITY_FIG_PREFIX="${RESULTS_ROOT}/figs/ssb_config_sensitivity_all_queries"
NCU_SUMMARY_CSV="${ANALYSIS_NCU_DIR}/ssb_ncu_kernel_summary_all_queries.csv"
NCU_SUMMARY_MD="${ANALYSIS_NCU_DIR}/ssb_ncu_kernel_summary_all_queries.md"
NCU_EXPORTER_CSV="${ANALYSIS_NCU_DIR}/ssb_ncu_metrics_all_queries.csv"
DEEP_NCU_PER_KERNEL_TMP_CSV="${DEEP_NCU_DIR}/.ssb_flbp_decom_reg_128x8_vs_256x4_per_kernel.tmp.csv"
DEEP_NCU_HOT_KERNEL_CSV="${ANALYSIS_DEEP_NCU_DIR}/ssb_decom_reg_128x8_vs_256x4_hot_pipeline.csv"
EXPLAIN_MERGED_CSV="${ANALYSIS_EXPLAIN_DIR}/ssb_fastest_configs_with_ncu.csv"
EXPLAIN_COMPARISON_CSV="${ANALYSIS_EXPLAIN_DIR}/ssb_mode_comparisons.csv"
EXPLAIN_SUMMARY_MD="${ANALYSIS_EXPLAIN_DIR}/ssb_behavior_summary.md"
ANALYZE_FOCUS_QUERY="${PAPER_ANALYZE_FOCUS_QUERY:-Q4.1}"

# Canonical benchmark configuration matrix used by timing + NCU + nsys.
# Format: "label|args"
ALL_CONFIGS=(
    "NONE|--compression_type=NONE"
    "RFOR_decom_op_t512_64x8|--compression_type=RFOR --decom_op --tile_size=512 --threads_per_block=64 --items_per_thread=8"
    "RFOR_decom_op_t512_128x4|--compression_type=RFOR --decom_op --tile_size=512 --threads_per_block=128 --items_per_thread=4"
    "RFOR_decom_op_t512_256x2|--compression_type=RFOR --decom_op --tile_size=512 --threads_per_block=256 --items_per_thread=2"
    "RFOR_decom_op_t1024_512x2|--compression_type=RFOR --decom_op --tile_size=1024 --threads_per_block=512 --items_per_thread=2"
    "RFOR_decom_op_t1024_256x4|--compression_type=RFOR --decom_op --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "RFOR_decom_op_t1024_128x8|--compression_type=RFOR --decom_op --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    #"RFOR_decom_op_t1024_64x16|--compression_type=RFOR --decom_op --tile_size=1024 --threads_per_block=64 --items_per_thread=16"  # poor performance
    "RFOR_decom_reg_t512_64x8|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=512 --threads_per_block=64 --items_per_thread=8"
    "RFOR_decom_reg_t512_128x4|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=512 --threads_per_block=128 --items_per_thread=4"
    "RFOR_decom_reg_t512_256x2|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=512 --threads_per_block=256 --items_per_thread=2"
    "RFOR_decom_reg_t1024_512x2|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=512 --items_per_thread=2"
    "RFOR_decom_reg_t1024_256x4|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "RFOR_decom_reg_t1024_128x8|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    #"RFOR_decom_reg_t1024_64x16|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=64 --items_per_thread=16"  # poor performance
    "RFOR_pre_decom_t512_64x8|--compression_type=RFOR --pre_decom --tile_size=512 --threads_per_block=64 --items_per_thread=8"
    "RFOR_pre_decom_t512_128x4|--compression_type=RFOR --pre_decom --tile_size=512 --threads_per_block=128 --items_per_thread=4"
    "RFOR_pre_decom_t512_256x2|--compression_type=RFOR --pre_decom --tile_size=512 --threads_per_block=256 --items_per_thread=2"
    "RFOR_pre_decom_t1024_512x2|--compression_type=RFOR --pre_decom --tile_size=1024 --threads_per_block=512 --items_per_thread=2"
    "RFOR_pre_decom_t1024_256x4|--compression_type=RFOR --pre_decom --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "RFOR_pre_decom_t1024_128x8|--compression_type=RFOR --pre_decom --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    #"RFOR_pre_decom_t1024_64x16|--compression_type=RFOR --pre_decom --tile_size=1024 --threads_per_block=64 --items_per_thread=16"  # poor performance
    "FLBP_decom_op_t1024_512x2|--compression_type=FLBP --decom_op --threads_per_block=512 --items_per_thread=2"
    "FLBP_decom_op_t1024_256x4|--compression_type=FLBP --decom_op --threads_per_block=256 --items_per_thread=4"
    "FLBP_decom_op_t1024_128x8|--compression_type=FLBP --decom_op --threads_per_block=128 --items_per_thread=8"
    "FLBP_decom_op_t1024_64x16|--compression_type=FLBP --decom_op --threads_per_block=64 --items_per_thread=16"
    "FLBP_decom_reg_t1024_512x2|--compression_type=FLBP --decom_reg --threads_per_block=512 --items_per_thread=2"
    "FLBP_decom_reg_t1024_256x4|--compression_type=FLBP --decom_reg --threads_per_block=256 --items_per_thread=4"
    "FLBP_decom_reg_t1024_128x8|--compression_type=FLBP --decom_reg --threads_per_block=128 --items_per_thread=8"
    "FLBP_decom_reg_t1024_64x16|--compression_type=FLBP --decom_reg --threads_per_block=64 --items_per_thread=16"
    "FLBP_pre_decom_t1024_512x2|--compression_type=FLBP --pre_decom --tile_size=1024 --threads_per_block=512 --items_per_thread=2"
    "FLBP_pre_decom_t1024_256x4|--compression_type=FLBP --pre_decom --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "FLBP_pre_decom_t1024_128x8|--compression_type=FLBP --pre_decom --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    "FLBP_pre_decom_t1024_64x16|--compression_type=FLBP --pre_decom --tile_size=1024 --threads_per_block=64 --items_per_thread=16"
)

# Focused matrix for launch-geometry behavior inversions:
# - FLBP/RFOR decom_reg compare 128x8 vs 256x4 on Q1.1/Q3.1/Q4.1.
# - RFOR decom_op compares t512 128x4 vs 256x2 on Q2.1/Q4.1.
# Format: "query|label|args"
DEEP_NCU_CONFIGS=(
    "11|FLBP_decom_reg_t1024_128x8|--compression_type=FLBP --decom_reg --threads_per_block=128 --items_per_thread=8"
    "11|FLBP_decom_reg_t1024_256x4|--compression_type=FLBP --decom_reg --threads_per_block=256 --items_per_thread=4"
    "31|FLBP_decom_reg_t1024_128x8|--compression_type=FLBP --decom_reg --threads_per_block=128 --items_per_thread=8"
    "31|FLBP_decom_reg_t1024_256x4|--compression_type=FLBP --decom_reg --threads_per_block=256 --items_per_thread=4"
    "41|FLBP_decom_reg_t1024_128x8|--compression_type=FLBP --decom_reg --threads_per_block=128 --items_per_thread=8"
    "41|FLBP_decom_reg_t1024_256x4|--compression_type=FLBP --decom_reg --threads_per_block=256 --items_per_thread=4"
    "11|RFOR_decom_reg_t1024_128x8|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    "11|RFOR_decom_reg_t1024_256x4|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "31|RFOR_decom_reg_t1024_128x8|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    "31|RFOR_decom_reg_t1024_256x4|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "41|RFOR_decom_reg_t1024_128x8|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=128 --items_per_thread=8"
    "41|RFOR_decom_reg_t1024_256x4|--compression_type=RFOR --decom_reg --precompute_rl --tile_size=1024 --threads_per_block=256 --items_per_thread=4"
    "21|RFOR_decom_op_t512_128x4|--compression_type=RFOR --decom_op --precompute_rl --tile_size=512 --threads_per_block=128 --items_per_thread=4"
    "21|RFOR_decom_op_t512_256x2|--compression_type=RFOR --decom_op --precompute_rl --tile_size=512 --threads_per_block=256 --items_per_thread=2"
    "41|RFOR_decom_op_t512_128x4|--compression_type=RFOR --decom_op --precompute_rl --tile_size=512 --threads_per_block=128 --items_per_thread=4"
    "41|RFOR_decom_op_t512_256x2|--compression_type=RFOR --decom_op --precompute_rl --tile_size=512 --threads_per_block=256 --items_per_thread=2"
)

extract_last_match_number() {
    local pattern="$1"
    local log_file="$2"
    local value
    value=$(grep -Eo "$pattern" "$log_file" 2>/dev/null | sed 's/.*=//' | tail -n1)
    if [[ -z "$value" ]]; then
        value="NA"
    fi
    echo "$value"
}

file_matches_regex() {
    local pattern="$1"
    local input_file="$2"
    grep -Eq "$pattern" "$input_file" 2>/dev/null
}

extract_last_regex_match() {
    local pattern="$1"
    local input_file="$2"
    grep -Eo "$pattern" "$input_file" 2>/dev/null | tail -n1
}

resolve_first_existing_file() {
    local path
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

resolve_ncu_exporter_tool() {
    local exporter_tool
    exporter_tool=$(resolve_first_existing_file "${NCU_EXPORTER_TOOL_CANDIDATES[@]}")
    if [[ -z "$exporter_tool" ]]; then
        echo "ERROR: could not find my_ncu_exporter.py in any known location." >&2
        return 1
    fi
    echo "$exporter_tool"
}

extract_ncu_metrics_csv() {
    local report_file="$1"
    local metric_list="$2"
    "$NCU" --import "$report_file" --csv 2>/dev/null | awk -v metric_list="$metric_list" '
        function ncu_byte_unit_scale(unit) {
            if (unit ~ /^Tbyte(\/.*)?$/) {
                return 1000 * 1000 * 1000 * 1000;
            }
            if (unit ~ /^Gbyte(\/.*)?$/) {
                return 1000 * 1000 * 1000;
            }
            if (unit ~ /^Mbyte(\/.*)?$/) {
                return 1000 * 1000;
            }
            if (unit ~ /^Kbyte(\/.*)?$/) {
                return 1000;
            }
            return 1;
        }
        function normalize_ncu_metric_value(metric, raw_value, unit, value, scale) {
            value = raw_value;
            gsub(/,/, "", value);
            scale = ncu_byte_unit_scale(unit);
            if (scale != 1 || unit ~ /^byte(\/.*)?$/) {
                return value * scale;
            }
            return value + 0;
        }
        BEGIN {
            mlen = split(metric_list, mlist, ",");
            for (i = 1; i <= mlen; i++) {
                want[mlist[i]] = 1;
            }
            parsed_header = 0;
        }
        parsed_header == 0 {
            gsub(/^"/, "", $0);
            gsub(/"$/, "", $0);
            n = split($0, h, "\",\"");
            if (h[1] != "ID") {
                next;
            }
            for (i = 1; i <= n; i++) {
                idx[h[i]] = i;
            }
            parsed_header = 1;
            next;
        }
        {
            gsub(/^"/, "", $0);
            gsub(/"$/, "", $0);
            n = split($0, a, "\",\"");
            if (!("Kernel Name" in idx) || !("Metric Name" in idx) || !("Metric Value" in idx) || !("Metric Unit" in idx)) {
                next;
            }
            kernel = a[idx["Kernel Name"]];
            metric = a[idx["Metric Name"]];
            value = a[idx["Metric Value"]];
            unit = a[idx["Metric Unit"]];
            if (!(metric in want) || value == "n/a" || value == "") {
                next;
            }
            value = normalize_ncu_metric_value(metric, value, unit);
            sum_all[metric] += value;
            cnt_all[metric] += 1;
            if (kernel ~ /_pip/) {
                sum_pip[metric] += value;
                cnt_pip[metric] += 1;
            }
        }
        END {
            for (i = 1; i <= mlen; i++) {
                m = mlist[i];
                if (cnt_pip[m] > 0) {
                    val = sum_pip[m] / cnt_pip[m];
                } else if (cnt_all[m] > 0) {
                    val = sum_all[m] / cnt_all[m];
                } else {
                    val = "NA";
                }
                printf "%s", val;
                if (i < mlen) {
                    printf ",";
                } else {
                    printf "\n";
                }
            }
        }
    '
}

append_markdown_table_from_csv() {
    local input_csv="$1"
    local output_md="$2"
    {
        echo "| query | strategy | proteus_kernel_num_regs | proteus_kernel_static_smem_bytes | occ_blocks_per_sm | occ_warps_per_sm | occ_theoretical_pct | ptxas_used_registers | ptxas_smem_bytes | ncu_launch_regs_per_thread | ncu_launch_smem_static_bytes | ncu_launch_smem_dynamic_bytes | ncu_launch_smem_total_bytes | ncu_sm_warps_active_pct | ncu_l1_hit_rate | ncu_l2_hit_rate | ncu_dram_bytes_read | ncu_dram_bytes_write |"
        echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        tail -n +2 "$input_csv" | awk -F',' '{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18}'
    } > "$output_md"
}

validate_ncu_export_coverage() {
    local exporter_csv="$1"
    shift
    python3 - "$exporter_csv" "$@" <<'PY'
import csv
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(0)

exporter_csv = Path(sys.argv[1])
expected_reports = [Path(arg).resolve() for arg in sys.argv[2:]]

if not exporter_csv.exists():
    print(f"WARNING: NCU exporter CSV not found: {exporter_csv}")
    raise SystemExit(0)

seen_reports = set()
with exporter_csv.open() as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        filename = (row.get("Filename") or "").strip()
        if filename:
            seen_reports.add(Path(filename).resolve())

missing = [
    report for report in expected_reports
    if report.exists() and report not in seen_reports
]

if missing:
    print("WARNING: exporter CSV is missing some generated NCU reports:")
    for report in missing:
        print(f"  {report.name}")
PY
}

run_ptxas_analysis() {
    local target_dir="$1"
    local ptxas_script="$SCRIPT_DIR/analyze_ptxas.sh"
    if [[ -x "$ptxas_script" ]]; then
        "$ptxas_script" "$target_dir"
    else
        echo "WARNING: ptxas analyzer not found/executable: $ptxas_script"
    fi
}

run_timing_case() {
    local label="$1"
    shift
    local -a extra_args=("$@")
    local result_file="$TIMING_DIR/results_q99_in_GPU_${label}.txt"
    local log_file="$TIMING_DIR/log_q99_in_GPU_${label}.txt"
    NUM_ITERATIONS=10

    echo "[collect:timing] Running case: $label"
    rm -f "$result_file"
    "$EXECUTABLE" --num_iterations=$NUM_ITERATIONS --query=99 --in-gpu \
        "${SAFE_CB2T_ARGS[@]}" "${extra_args[@]}" \
        --result_file="$result_file" \
        > "$log_file" 2>&1
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "WARNING: timing case failed ($label). See $log_file"
        return
    fi

    if [[ ! -s "$result_file" ]]; then
        echo "WARNING: timing case produced empty result file ($label). See $log_file"
        return
    fi

    if ! file_matches_regex '^query,compression_type,' "$result_file"; then
        echo "WARNING: timing case produced non-CSV result file ($label). See $log_file"
    fi
}

# ---------------------------------------------------------------------------
# Timing benchmarks
# ---------------------------------------------------------------------------
run_timing() {
    echo "=== Collect: timing ==="
    echo "Raw timing outputs will be written under: $TIMING_DIR"

    local cfg cfg_label cfg_args
    for cfg in "${ALL_CONFIGS[@]}"; do
        cfg_label="${cfg%%|*}"
        cfg_args="${cfg#*|}"
        run_timing_case "$cfg_label" ${=cfg_args}
    done

    # one_kernel sensitivity (Q1.1 only, uncomment to enable)
    #"$EXECUTABLE" --num_iterations=10 --query=11 --compression_type=NONE --in-gpu \
    #    --result_file="$TIMING_DIR/results_q11_in_GPU_NONE_ONE_KERNEL_ON.txt" \
    #    > "$TIMING_DIR/log_q11_in_GPU_NONE_ONE_KERNEL_ON.txt" 2>&1
    #"$EXECUTABLE" --num_iterations=10 --query=11 --compression_type=NONE --in-gpu --one_kernel=false \
    #    --result_file="$TIMING_DIR/results_q11_in_GPU_NONE_ONE_KERNEL_OFF.txt" \
    #    > "$TIMING_DIR/log_q11_in_GPU_NONE_ONE_KERNEL_OFF.txt" 2>&1

    echo "[collect:timing] Done."
    echo "[collect:timing] Result files: $TIMING_DIR/results_q99_*.txt"
    echo "[collect:timing] Log files   : $TIMING_DIR/log_q99_*.txt"
}

# ---------------------------------------------------------------------------
# NCU hardware counter profiling (all SSB queries)
# ---------------------------------------------------------------------------
run_ncu() {
    echo "=== Collect: NCU ==="
    echo "Raw NCU outputs will be written under: $NCU_DIR"
    echo "[collect:ncu] Queries: ${ALL_QUERIES[*]}"

    local ncu_flags=(
        --metrics "$NCU_METRICS"
        --config-file off
        --force-overwrite
        --replay-mode application
        --app-replay-buffer file
        --app-replay-mode relaxed
        --nvtx
        --profile-from-start 0
        --check-exit-code no
    )

    local query strategy_label strategy_args safe_label app_log report_path report_file

    for query in "${ALL_QUERIES[@]}"; do
        for strategy in "${ALL_CONFIGS[@]}"; do
            strategy_label="${strategy%%|*}"
            strategy_args="${strategy#*|}"
            safe_label="${strategy_label}"
            app_log="$NCU_DIR/log_q${query}_in_GPU_${safe_label}_ncu.txt"
            report_path="$NCU_DIR/ncu_Q${query}_in_GPU_${safe_label}"
            report_file="${report_path}.ncu-rep"

            "$EXECUTABLE" --num_iterations=1 --query="$query" --in-gpu \
                "${SAFE_CB2T_ARGS[@]}" --warm_up=false ${=strategy_args} \
                > "$app_log" 2>&1

            "$NCU" "${ncu_flags[@]}" --export "$report_path" \
                "$EXECUTABLE" --num_iterations=1 --query="$query" --in-gpu \
                "${SAFE_CB2T_ARGS[@]}" --warm_up=false ${=strategy_args}
            local ncu_exit_code=$?
            if [[ $ncu_exit_code -ne 0 || ! -f "$report_file" ]]; then
                echo "WARNING: missing NCU report for Q${query} ${strategy_label}. See $app_log"
                continue
            fi
        done
    done

    echo "[collect:ncu] Done."
    echo "[collect:ncu] Report files: $NCU_DIR/ncu_Q*_in_GPU_*.ncu-rep"
    echo "[collect:ncu] App logs    : $NCU_DIR/log_q*_in_GPU_*_ncu.txt"
}

write_deep_ncu_csv_header() {
    local output_csv="$1"
    local metric_list="$2"
    {
        printf "report_file,query,strategy,kernel_id,kernel_name,is_pipeline,is_focus_kernel"
        local metric
        for metric in ${(s:,:)metric_list}; do
            printf ",%s" "$metric"
        done
        printf "\n"
    } > "$output_csv"
}

append_deep_ncu_per_kernel_csv() {
    local report_file="$1"
    local query="$2"
    local strategy="$3"
    local output_csv="$4"
    local metric_list="$5"

    "$NCU" --import "$report_file" --csv 2>/dev/null | awk \
        -v report_file="$report_file" \
        -v query="$query" \
        -v strategy="$strategy" \
        -v metric_list="$metric_list" \
        -v output_csv="$output_csv" '
        function ncu_byte_unit_scale(unit) {
            if (unit ~ /^Tbyte(\/.*)?$/) return 1000 * 1000 * 1000 * 1000;
            if (unit ~ /^Gbyte(\/.*)?$/) return 1000 * 1000 * 1000;
            if (unit ~ /^Mbyte(\/.*)?$/) return 1000 * 1000;
            if (unit ~ /^Kbyte(\/.*)?$/) return 1000;
            return 1;
        }
        function ncu_time_unit_scale_ns(unit) {
            if (unit ~ /^second(\/.*)?$/) return 1000 * 1000 * 1000;
            if (unit ~ /^msecond(\/.*)?$/) return 1000 * 1000;
            if (unit ~ /^usecond(\/.*)?$/) return 1000;
            if (unit ~ /^nsecond(\/.*)?$/) return 1;
            return 0;
        }
        function normalize_ncu_metric_value(metric, raw_value, unit, value, scale) {
            value = raw_value;
            gsub(/,/, "", value);
            scale = ncu_byte_unit_scale(unit);
            if (scale != 1 || unit ~ /^byte(\/.*)?$/) return value * scale;
            scale = ncu_time_unit_scale_ns(unit);
            if (scale != 0) return value * scale;
            return value + 0;
        }
        function csvq(s) {
            gsub(/"/, "\"\"", s);
            return "\"" s "\"";
        }
        BEGIN {
            mlen = split(metric_list, mlist, ",");
            for (i = 1; i <= mlen; i++) want[mlist[i]] = 1;
            parsed_header = 0;
        }
        parsed_header == 0 {
            gsub(/^"/, "", $0);
            gsub(/"$/, "", $0);
            n = split($0, h, "\",\"");
            if (h[1] != "ID") next;
            for (i = 1; i <= n; i++) idx[h[i]] = i;
            parsed_header = 1;
            next;
        }
        {
            gsub(/^"/, "", $0);
            gsub(/"$/, "", $0);
            n = split($0, a, "\",\"");
            if (!("ID" in idx) || !("Kernel Name" in idx) ||
                !("Metric Name" in idx) || !("Metric Value" in idx) ||
                !("Metric Unit" in idx)) {
                next;
            }
            kernel_id = a[idx["ID"]];
            kernel = a[idx["Kernel Name"]];
            metric = a[idx["Metric Name"]];
            value = a[idx["Metric Value"]];
            unit = a[idx["Metric Unit"]];
            if (!(metric in want) || value == "n/a" || value == "") next;

            key = kernel_id SUBSEP kernel;
            if (!(key in seen)) {
                seen[key] = 1;
                ordered[++ordered_len] = key;
                kid[key] = kernel_id;
                kname[key] = kernel;
            }
            vals[key, metric] = normalize_ncu_metric_value(metric, value, unit);
        }
        END {
            for (o = 1; o <= ordered_len; o++) {
                key = ordered[o];
                kernel = kname[key];
                is_pipeline = (kernel ~ /_pip/) ? "true" : "false";
                is_focus_kernel = (kernel ~ /ssb100_Q(1_1|2_1|3_1|4_1)nmv_pip/) ? "true" : "false";
                printf "%s,%s,%s,%s,%s,%s,%s", \
                    csvq(report_file), csvq(query), csvq(strategy), csvq(kid[key]), \
                    csvq(kernel), is_pipeline, is_focus_kernel >> output_csv;
                for (i = 1; i <= mlen; i++) {
                    metric = mlist[i];
                    if ((key, metric) in vals) {
                        printf ",%s", vals[key, metric] >> output_csv;
                    } else {
                        printf ",NA" >> output_csv;
                    }
                }
                printf "\n" >> output_csv;
            }
        }
    '
}

write_deep_hot_pipeline_csv() {
    local per_kernel_csv="$1"
    local hot_kernel_csv="$2"
    python3 - "$per_kernel_csv" "$hot_kernel_csv" <<'PY'
import csv
import re
import sys
from pathlib import Path

per_kernel_csv = Path(sys.argv[1])
hot_kernel_csv = Path(sys.argv[2])

if not per_kernel_csv.exists():
    raise SystemExit(f"missing per-kernel CSV: {per_kernel_csv}")

with per_kernel_csv.open(newline="") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
    fieldnames = reader.fieldnames or []

ptxas_fieldnames = [
    "ptxas_used_registers",
    "ptxas_smem_bytes",
    "ptxas_stack_bytes",
    "ptxas_spill_stores_bytes",
    "ptxas_spill_loads_bytes",
    "ptxas_register_cap_warning",
]
for field in ptxas_fieldnames:
    if field not in fieldnames:
        fieldnames.append(field)

if not rows:
    hot_kernel_csv.write_text("")
    raise SystemExit(0)

time_col = "gpu__time_duration.sum"
if time_col not in fieldnames:
    raise SystemExit(f"missing {time_col} in {per_kernel_csv}")

def metric_value(row, column):
    raw = row.get(column, "NA")
    if raw in ("", "NA", None):
        return None
    try:
        return float(raw)
    except ValueError:
        return None

preferred = {
    "Q11": "ssb100_Q1_1nmv_pip5",
    "Q21": "ssb100_Q2_1nmv_pip6",
    "Q31": "ssb100_Q3_1nmv_pip7",
    "Q41": "ssb100_Q4_1nmv_pip7",
}
group_rows = {}
for row in rows:
    key = (row.get("query", ""), row.get("strategy", ""), row.get("report_file", ""))
    group_rows.setdefault(key, []).append(row)

groups = {}
for key, candidates in group_rows.items():
    query = key[0]
    preferred_kernel = preferred.get(query)
    if preferred_kernel:
        for row in candidates:
            if preferred_kernel in (row.get("kernel_name") or ""):
                groups[key] = row
                break
        if key in groups:
            continue

    pipeline_candidates = [
        row for row in candidates
        if row.get("is_pipeline") == "true" and metric_value(row, time_col) is not None
    ]
    if pipeline_candidates:
        groups[key] = max(pipeline_candidates, key=lambda row: metric_value(row, time_col))
        continue

    timed_candidates = [
        row for row in candidates
        if metric_value(row, time_col) is not None
    ]
    if timed_candidates:
        groups[key] = max(timed_candidates, key=lambda row: metric_value(row, time_col))

def parse_ptxas_for_kernel(report_file, kernel_name):
    result = {
        "ptxas_used_registers": "NA",
        "ptxas_smem_bytes": "NA",
        "ptxas_stack_bytes": "0",
        "ptxas_spill_stores_bytes": "0",
        "ptxas_spill_loads_bytes": "0",
        "ptxas_register_cap_warning": "false",
    }
    report_path = Path(report_file)
    if not report_path.name.startswith("ncu_") or not report_path.name.endswith(".ncu-rep"):
        return result
    safe_label = report_path.name[len("ncu_"):-len(".ncu-rep")]
    log_paths = [
        report_path.with_name(f"log_{safe_label}_ncu_app.txt"),
        report_path.with_name(f"log_{safe_label}_timing.txt"),
    ]

    pending_kernel = None
    saw_cap_warning = False
    used_line_re = re.compile(r"Used\s+(\d+)\s+registers")
    smem_re = re.compile(r"(\d+)\s+bytes\s+smem")
    stack_re = re.compile(r"(\d+)\s+bytes\s+cumulative stack size")
    spill_stores_re = re.compile(r"(\d+)\s+bytes\s+spill stores")
    spill_loads_re = re.compile(r"(\d+)\s+bytes\s+spill loads")

    for log_path in log_paths:
        if not log_path.exists():
            continue
        pending_kernel = None
        with log_path.open(errors="replace") as fh:
            for line in fh:
                if "ptxas" not in line:
                    continue
                if (
                    re.search(r"warning.*[Rr]egister", line)
                    or "maxrregcount" in line
                    or "Overriding maximum register" in line
                ):
                    saw_cap_warning = True
                if f"Function properties for {kernel_name}" in line:
                    pending_kernel = kernel_name
                    continue
                if pending_kernel != kernel_name:
                    continue
                stack_match = stack_re.search(line)
                spill_stores_match = spill_stores_re.search(line)
                spill_loads_match = spill_loads_re.search(line)
                if stack_match:
                    result["ptxas_stack_bytes"] = stack_match.group(1)
                if spill_stores_match:
                    result["ptxas_spill_stores_bytes"] = spill_stores_match.group(1)
                if spill_loads_match:
                    result["ptxas_spill_loads_bytes"] = spill_loads_match.group(1)
                if "Used " not in line or " registers" not in line:
                    continue

                used_match = used_line_re.search(line)
                smem_match = smem_re.search(line)
                if used_match:
                    result["ptxas_used_registers"] = used_match.group(1)
                if smem_match:
                    result["ptxas_smem_bytes"] = smem_match.group(1)
                if stack_match:
                    result["ptxas_stack_bytes"] = stack_match.group(1)
                if spill_stores_match:
                    result["ptxas_spill_stores_bytes"] = spill_stores_match.group(1)
                if spill_loads_match:
                    result["ptxas_spill_loads_bytes"] = spill_loads_match.group(1)
                result["ptxas_register_cap_warning"] = "true" if saw_cap_warning else "false"
                pending_kernel = None

    return result

for row in groups.values():
    row.update(parse_ptxas_for_kernel(
        row.get("report_file", ""),
        row.get("kernel_name", ""),
    ))

with hot_kernel_csv.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for key in sorted(groups):
        writer.writerow(groups[key])

if not groups:
    raise SystemExit(
        "no hot pipeline rows selected; inspect the temporary per-kernel CSV"
    )
PY
}

# ---------------------------------------------------------------------------
# Focused per-kernel profiling for launch-geometry inversions
# ---------------------------------------------------------------------------
run_deep() {
    echo "=== Collect: deep NCU ==="
    echo "[collect:deep] Focus: FLBP/RFOR decom_reg 128x8 vs 256x4 plus RFOR decom_op t512 128x4 vs 256x2"
    echo "[collect:deep] Raw reports: $DEEP_NCU_DIR"
    echo "[collect:deep] Hot-pipeline CSV: $DEEP_NCU_HOT_KERNEL_CSV"
    echo "[collect:deep] Metrics: $DEEP_NCU_METRICS"

    local timing_iterations="${PAPER_DEEP_TIMING_ITERATIONS:-5}"
    local ncu_iterations="${PAPER_DEEP_NCU_ITERATIONS:-1}"
    local ncu_warmup="${PAPER_DEEP_WARMUP:-false}"
    local ncu_flags=(
        --metrics "$DEEP_NCU_METRICS"
        --config-file off
        --force-overwrite
        --replay-mode application
        --app-replay-buffer file
        --app-replay-mode relaxed
        --nvtx
        --profile-from-start 0
        --check-exit-code no
    )

    write_deep_ncu_csv_header "$DEEP_NCU_PER_KERNEL_TMP_CSV" "$DEEP_NCU_METRICS"

    local cfg query strategy_label strategy_args safe_label
    local timing_result timing_log app_log report_path report_file
    for cfg in "${DEEP_NCU_CONFIGS[@]}"; do
        query="${cfg%%|*}"
        local rest="${cfg#*|}"
        strategy_label="${rest%%|*}"
        strategy_args="${rest#*|}"
        safe_label="Q${query}_${strategy_label}"
        timing_result="$DEEP_NCU_DIR/results_${safe_label}.txt"
        timing_log="$DEEP_NCU_DIR/log_${safe_label}_timing.txt"
        app_log="$DEEP_NCU_DIR/log_${safe_label}_ncu_app.txt"
        report_path="$DEEP_NCU_DIR/ncu_${safe_label}"
        report_file="${report_path}.ncu-rep"

        echo "[collect:deep] Timing ${safe_label}"
        rm -f "$timing_result"
        "$EXECUTABLE" --num_iterations="$timing_iterations" --query="$query" --in-gpu \
            "${SAFE_CB2T_ARGS[@]}" ${=strategy_args} \
            --result_file="$timing_result" \
            > "$timing_log" 2>&1
        local timing_exit_code=$?
        if [[ $timing_exit_code -ne 0 ]]; then
            echo "WARNING: deep timing failed for ${safe_label}. See $timing_log"
        fi

        echo "[collect:deep] NCU ${safe_label}"
        "$EXECUTABLE" --num_iterations="$ncu_iterations" --query="$query" --in-gpu \
            "${SAFE_CB2T_ARGS[@]}" --warm_up="$ncu_warmup" ${=strategy_args} \
            > "$app_log" 2>&1

        "$NCU" "${ncu_flags[@]}" --export "$report_path" \
            "$EXECUTABLE" --num_iterations="$ncu_iterations" --query="$query" --in-gpu \
            "${SAFE_CB2T_ARGS[@]}" --warm_up="$ncu_warmup" ${=strategy_args}
        local ncu_exit_code=$?
        if [[ $ncu_exit_code -ne 0 || ! -f "$report_file" ]]; then
            echo "WARNING: missing deep NCU report for ${safe_label}. See $app_log"
            continue
        fi

        append_deep_ncu_per_kernel_csv \
            "$report_file" "Q${query}" "$strategy_label" \
            "$DEEP_NCU_PER_KERNEL_TMP_CSV" "$DEEP_NCU_METRICS"
    done

    run_ptxas_analysis "$DEEP_NCU_DIR"
    if write_deep_hot_pipeline_csv "$DEEP_NCU_PER_KERNEL_TMP_CSV" "$DEEP_NCU_HOT_KERNEL_CSV"; then
        rm -f "$DEEP_NCU_PER_KERNEL_TMP_CSV"
    else
        echo "WARNING: hot-pipeline extraction failed."
        echo "WARNING: kept temporary per-kernel CSV: $DEEP_NCU_PER_KERNEL_TMP_CSV"
    fi
    echo "[collect:deep] Done."
    echo "[collect:deep] Timing/result logs: $DEEP_NCU_DIR"
    echo "[collect:deep] Hot-pipeline CSV: $DEEP_NCU_HOT_KERNEL_CSV"
}

# ---------------------------------------------------------------------------
# nsys timeline profiling (Q1.1 only)
# ---------------------------------------------------------------------------
run_nsys() {
    echo "=== Collect: nsys ==="
    echo "Raw nsys outputs will be written under: $NSYS_DIR"

    local nsys_flags=(
        profile
        --trace=cuda,nvtx
        --stats=false
        --sample=none
        --cpuctxsw=none
        --gpu-metrics-device=1
        --capture-range=cudaProfilerApi
        --capture-range-end=none
        -f true
    )

    local cfg cfg_label cfg_args
    for cfg in "${ALL_CONFIGS[@]}"; do
        cfg_label="${cfg%%|*}"
        cfg_args="${cfg#*|}"
        "$NSYS" "${nsys_flags[@]}" -o "$NSYS_DIR/profilingQ11_in_GPU_${cfg_label}" \
            "$EXECUTABLE" --num_iterations=1 --query=11 --in-gpu \
            "${SAFE_CB2T_ARGS[@]}" --warm_up=false ${=cfg_args}
    done

    echo "[collect:nsys] Done."
    echo "[collect:nsys] Report files: $NSYS_DIR/profilingQ11_*.nsys-rep"
}

analyze_timing() {
    echo "=== Analyze: timing ==="
    echo "[analyze:timing] Reading raw timing outputs from: $TIMING_DIR"
    echo "[analyze:timing] Writing derived outputs to     : $ANALYSIS_TIMING_DIR"
    if python3 "$SCRIPT_DIR/summarize_fastest_codec_configs.py" \
        --results-dir "$TIMING_DIR" \
        --output "$FASTEST_CSV"; then
        echo "[analyze:timing] Fastest configurations: $FASTEST_CSV"
    else
        echo "WARNING: [analyze:timing] failed to compute fastest configurations."
    fi
    if python3 "$SCRIPT_DIR/fig_timing_config_sensitivity.py" \
        --timing-dir "$TIMING_DIR" \
        --analysis-dir "$ANALYSIS_TIMING_DIR" \
        --figs-dir "$RESULTS_ROOT/figs"; then
        echo "[analyze:timing] Config sensitivity CSV: $CONFIG_SENSITIVITY_CSV"
        echo "[analyze:timing] Config sensitivity PDF: ${CONFIG_SENSITIVITY_FIG_PREFIX}.pdf"
        echo "[analyze:timing] Config sensitivity PNG: ${CONFIG_SENSITIVITY_FIG_PREFIX}.png"
    else
        echo "WARNING: [analyze:timing] failed to generate config sensitivity figure."
    fi
    run_ptxas_analysis "$TIMING_DIR"
}

analyze_ncu() {
    echo "=== Analyze: NCU ==="
    echo "[analyze:ncu] Reading raw NCU outputs from: $NCU_DIR"
    echo "[analyze:ncu] Writing derived outputs to : $ANALYSIS_NCU_DIR"

    local extract_metrics="launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum"
    local exporter_tool
    exporter_tool=$(resolve_ncu_exporter_tool) || return 1

    echo "query,strategy,proteus_kernel_num_regs,proteus_kernel_static_smem_bytes,occ_blocks_per_sm,occ_warps_per_sm,occ_theoretical_pct,ptxas_used_registers,ptxas_smem_bytes,ncu_launch_regs_per_thread,ncu_launch_smem_static_bytes,ncu_launch_smem_dynamic_bytes,ncu_launch_smem_total_bytes,ncu_sm_warps_active_pct,ncu_l1_hit_rate,ncu_l2_hit_rate,ncu_dram_bytes_read,ncu_dram_bytes_write" > "$NCU_SUMMARY_CSV"

    local query strategy strategy_label safe_label app_log report_file
    local proteus_kernel_num_regs proteus_kernel_static_smem_bytes occ_blocks_per_sm occ_warps_per_sm occ_theoretical_pct
    local ptxas_used_registers ptxas_smem_bytes ncu_values
    local ncu_report_files=()

    for query in "${ALL_QUERIES[@]}"; do
        for strategy in "${ALL_CONFIGS[@]}"; do
            strategy_label="${strategy%%|*}"
            safe_label="${strategy_label}"
            app_log="$NCU_DIR/log_q${query}_in_GPU_${safe_label}_ncu.txt"
            report_file="$NCU_DIR/ncu_Q${query}_in_GPU_${safe_label}.ncu-rep"

            if [[ ! -f "$report_file" ]]; then
                echo "WARNING: [analyze:ncu] missing NCU report for Q${query} ${strategy_label}. Skipping."
                continue
            fi
            if [[ ! -f "$app_log" ]]; then
                echo "WARNING: [analyze:ncu] missing NCU app log for Q${query} ${strategy_label}. Skipping summary row."
                continue
            fi
            ncu_report_files+=("$report_file")

            proteus_kernel_num_regs=$(extract_last_match_number "kernel_num_regs=[0-9]+" "$app_log")
            proteus_kernel_static_smem_bytes=$(extract_last_match_number "kernel_static_smem_bytes=[0-9]+" "$app_log")
            occ_blocks_per_sm=$(extract_last_match_number "occ_blocks_per_sm=[0-9]+" "$app_log")
            occ_warps_per_sm=$(extract_last_match_number "occ_warps_per_sm=[0-9]+" "$app_log")
            occ_theoretical_pct=$(extract_last_match_number 'occ_theoretical_pct=[0-9]+(\.[0-9]+)?' "$app_log")
            ptxas_used_registers=$(extract_last_regex_match "Used [0-9]+ registers" "$app_log" | sed -E "s/Used ([0-9]+) registers/\\1/")
            ptxas_smem_bytes=$(extract_last_regex_match "Used [0-9]+ registers, [0-9]+ bytes smem" "$app_log" | sed -E "s/Used [0-9]+ registers, ([0-9]+) bytes smem/\\1/")
            if [[ -z "$ptxas_used_registers" ]]; then
                ptxas_used_registers="NA"
            fi
            if [[ -z "$ptxas_smem_bytes" ]]; then
                ptxas_smem_bytes="NA"
            fi
            ncu_values=$(extract_ncu_metrics_csv "$report_file" "$extract_metrics")

            echo "Q${query},${strategy_label},${proteus_kernel_num_regs},${proteus_kernel_static_smem_bytes},${occ_blocks_per_sm},${occ_warps_per_sm},${occ_theoretical_pct},${ptxas_used_registers},${ptxas_smem_bytes},${ncu_values}" >> "$NCU_SUMMARY_CSV"
        done
    done

    append_markdown_table_from_csv "$NCU_SUMMARY_CSV" "$NCU_SUMMARY_MD"

    : > "$NCU_EXPORTER_CSV"
    if (( ${#ncu_report_files[@]} > 0 )); then
        python3 "$exporter_tool" \
            --path "${ncu_report_files[@]}" \
            --metrics-names \
                launch__registers_per_thread \
                launch__shared_mem_per_block_static \
                launch__shared_mem_per_block_dynamic \
                launch__shared_mem_per_block \
                sm__warps_active.avg.pct_of_peak_sustained_active \
                l1tex__t_sector_hit_rate.pct \
                lts__t_sector_hit_rate.pct \
                dram__bytes_read.sum \
                dram__bytes_write.sum \
            --output "$NCU_EXPORTER_CSV"
        validate_ncu_export_coverage "$NCU_EXPORTER_CSV" "${ncu_report_files[@]}"
    else
        echo "WARNING: [analyze:ncu] no NCU reports found to export."
    fi

    echo "[analyze:ncu] Summary CSV : $NCU_SUMMARY_CSV"
    echo "[analyze:ncu] Summary MD  : $NCU_SUMMARY_MD"
    echo "[analyze:ncu] Exporter CSV: $NCU_EXPORTER_CSV"
    echo "[analyze:ncu] Exporter tool: $exporter_tool"
    run_ptxas_analysis "$NCU_DIR"
}

analyze_nsys() {
    echo "=== Analyze: nsys ==="
    echo "[analyze:nsys] Reading raw nsys outputs from: $NSYS_DIR"
    echo "[analyze:nsys] Writing derived outputs to   : $ANALYSIS_NSYS_DIR"
    run_ptxas_analysis "$NSYS_DIR"
}

analyze_explanations() {
    echo "=== Analyze: explanations ==="
    echo "[analyze:explanations] Reading timing summary from: $FASTEST_CSV"
    echo "[analyze:explanations] Reading NCU summary from   : $NCU_SUMMARY_CSV"
    echo "[analyze:explanations] Writing derived outputs to : $ANALYSIS_EXPLAIN_DIR"

    if [[ ! -f "$FASTEST_CSV" ]]; then
        echo "WARNING: [analyze:explanations] missing timing summary: $FASTEST_CSV"
        return
    fi

    local -a explain_args=(
        --timing-summary "$FASTEST_CSV"
        --merged-output "$EXPLAIN_MERGED_CSV"
        --comparison-output "$EXPLAIN_COMPARISON_CSV"
        --summary-md "$EXPLAIN_SUMMARY_MD"
        --focus-query "$ANALYZE_FOCUS_QUERY"
    )

    if [[ -f "$NCU_SUMMARY_CSV" ]]; then
        explain_args+=(--ncu-summary "$NCU_SUMMARY_CSV")
    else
        echo "WARNING: [analyze:explanations] NCU summary missing; timing-only explanation will be generated."
    fi

    if python3 "$SCRIPT_DIR/analyze_codec_behavior.py" "${explain_args[@]}"; then
        echo "[analyze:explanations] Merged CSV : $EXPLAIN_MERGED_CSV"
        echo "[analyze:explanations] Compare CSV: $EXPLAIN_COMPARISON_CSV"
        echo "[analyze:explanations] Summary MD : $EXPLAIN_SUMMARY_MD"
    else
        echo "WARNING: [analyze:explanations] failed to generate behavior summary."
    fi
}

run_analyze() {
    echo "=== Analyze phase ==="
    echo "[analyze] Raw artifact roots:"
    echo "[analyze]   timing: $TIMING_DIR"
    echo "[analyze]   ncu   : $NCU_DIR"
    echo "[analyze]   nsys  : $NSYS_DIR"
    echo "[analyze] Derived outputs root: $ANALYSIS_DIR"
    analyze_timing
    analyze_ncu
    analyze_nsys
    analyze_explanations
    echo "[analyze] Done."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    timing) run_timing ;;
    ncu)    run_ncu    ;;
    nsys)   run_nsys   ;;
    deep)   run_deep   ;;
    analyze) run_analyze ;;
    all)    run_timing; run_ncu; run_nsys; run_analyze ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [timing|ncu|nsys|deep|analyze|all]"
        exit 1
        ;;
esac

# After a timing run, analyse ptxas register info with:
#   ./results/scripts/analyze_ptxas.sh ./results/timing_logs
