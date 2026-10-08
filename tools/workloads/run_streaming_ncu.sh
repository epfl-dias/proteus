#!/usr/bin/env bash
# Focused NCU comparison for the reviewer-R2 global staging placement.
# Profiles SSB Q1.1 through the same SSBInGPU benchmark fixture as the timing
# sweep, inside Proteus's existing profiler-controlled query range.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO_ROOT/tools/workloads/common.sh"
OUT_BASE="${OUT_BASE:-$WORKLOAD_OUTPUT_ROOT/streaming_ncu_$(date +%Y%m%d_%H%M%S)}"
LOCK_FILE="${STREAMING_SWEEP_LOCK:-/tmp/proteus_streaming_sweep.lock}"
EXECUTABLE="${EXECUTABLE:-$BUILD/apps/benchmarks/olap/benchmark-ssb/proteus-benchmark-ssb}"
NCU="${NCU:-ncu}"
CASES="${CASES:-NONE RFOR_OP RFOR_STREAM_W1 RFOR_STREAM_W8 RFOR_STREAM_W16 RFOR_STREAM_W18 RFOR_PD FLBP_OP FLBP_STREAM_W1 FLBP_STREAM_W8 FLBP_STREAM_W16 FLBP_STREAM_W18 FLBP_PD}"
COMPRESSED_GPU_BUFFERS="${COMPRESSED_GPU_BUFFERS:-0.6}"
MAX_GRID_BLOCKS="${MAX_GRID_BLOCKS:-0}"

METRICS="${NCU_METRICS:-gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another streaming sweep holds $LOCK_FILE" >&2
  exit 3
fi

[[ -x "$EXECUTABLE" ]] || { echo "missing executable: $EXECUTABLE" >&2; exit 2; }
command -v "$NCU" >/dev/null 2>&1 || { echo "missing NCU: $NCU" >&2; exit 2; }
mkdir -p "$OUT_BASE" "$PELAGO/inputs"
ulimit -c 0
export LD_LIBRARY_PATH="$(proteus_library_path)"

cat > "$OUT_BASE/manifest.txt" <<EOF
commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
date=$(date --iso-8601=seconds)
query=SSB_Q1.1
cases=$CASES
metrics=$METRICS
max_grid_blocks=$MAX_GRID_BLOCKS
EOF

for label in $CASES; do
  codec=${label%%_*}
  if [[ "$label" == NONE ]]; then
    codec=NONE
    dataset="$SSB_NONE_DS"
    gpu_buffers=0.2
    mode_args=(--compression_type=NONE)
  else
    case "$codec" in
      RFOR)
        dataset="$SSB_RFOR_DS"
        geometry=(--tile_size=1024 --threads_per_block=256 --items_per_thread=4)
        ;;
      FLBP)
        dataset="$SSB_FLBP_DS"
        geometry=(--tile_size=1024 --threads_per_block=128 --items_per_thread=8)
        ;;
      *) echo "unsupported case: $label" >&2; exit 2 ;;
    esac
    gpu_buffers="$COMPRESSED_GPU_BUFFERS"
    case "$label" in
      *_OP) mode_args=(--compression_type="$codec" --decom_op=true "${geometry[@]}") ;;
      *_PD)
        gpu_buffers=0.2
        mode_args=(--compression_type="$codec" --pre_decom=true "${geometry[@]}")
        ;;
      *_STREAM_W1)
        mode_args=(--compression_type="$codec" --decom_global_stage=true --streaming_grid_rows=1 "${geometry[@]}")
        ;;
      *_STREAM_W8)
        mode_args=(--compression_type="$codec" --decom_global_stage=true --streaming_grid_rows=8 "${geometry[@]}")
        ;;
      *_STREAM_W16)
        mode_args=(--compression_type="$codec" --decom_global_stage=true --streaming_grid_rows=16 "${geometry[@]}")
        ;;
      *_STREAM_W18)
        mode_args=(--compression_type="$codec" --decom_global_stage=true --streaming_grid_rows=18 "${geometry[@]}")
        ;;
      *_STREAM_W24)
        mode_args=(--compression_type="$codec" --decom_global_stage=true --streaming_grid_rows=24 "${geometry[@]}")
        ;;
      *) echo "unsupported case: $label" >&2; exit 2 ;;
    esac
    mode_args+=(--max_grid_blocks="$MAX_GRID_BLOCKS" --use_metadata_in_block=true)
  fi

  [[ -f "$dataset/catalog.json" ]] || { echo "missing dataset: $dataset" >&2; exit 2; }
  ln -sfnT "$dataset" "$PELAGO/inputs/ssbm100"
  report_base="$OUT_BASE/$label"
  app_log="$OUT_BASE/$label.log"
  echo "=== NCU $label ===" | tee -a "$OUT_BASE/progress.log"

  set +e
  (
    cd "$PELAGO"
    "$NCU" \
      --metrics "$METRICS" \
      --force-overwrite \
      --replay-mode application \
      --app-replay-buffer file \
      --app-replay-mode relaxed \
      --nvtx \
      --profile-from-start 0 \
      --check-exit-code no \
      --export "$report_base" \
      "$EXECUTABLE" \
        --benchmark_filter=SSBInGPU/INGPU_Q1_1 \
        --benchmark_min_time=0 --benchmark_repetitions=1 \
        --benchmark_color=false --benchmark_format=console \
        --in_gpu=true --one_kernel=true --cpu_buffers=0.01 \
        --gpu_buffers="$gpu_buffers" --print_generated_code=false \
        "${mode_args[@]}"
  ) > "$app_log" 2>&1
  rc=$?
  set -e

  report="$report_base.ncu-rep"
  if [[ ! -f "$report" ]]; then
    echo "$label FAILED rc=$rc (no report)" | tee -a "$OUT_BASE/progress.log"
    continue
  fi
  "$NCU" --import "$report" --csv --page raw > "$report_base.raw.csv"
  echo "$label OK rc=$rc report=$report" | tee -a "$OUT_BASE/progress.log"
done

shopt -s nullglob
raw_csvs=("$OUT_BASE"/*.raw.csv)
if [[ ${#raw_csvs[@]} -eq 0 ]]; then
  echo "no NCU reports were exported" >&2
  exit 4
fi
python3 "$REPO_ROOT/tools/workloads/summarize_streaming_ncu.py" \
  "${raw_csvs[@]}" > "$OUT_BASE/summary.csv"
echo "ALL_DONE summary=$OUT_BASE/summary.csv" | tee -a "$OUT_BASE/progress.log"
