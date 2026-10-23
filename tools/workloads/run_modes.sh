#!/usr/bin/env bash
# Uniform timing run for the revision workloads (SSB / TPC-H / Stack) on this
# branch. Runs Uncom first as the reference, then full-RFOR and full-FLBP in the
# chosen decode mode. Result dumping is deliberately disabled here:
# PROTEUS_RESULT_DUMP_DIR changes the measured benchmark behavior and must not be
# present in matrix timing runs.
#
#   WORKLOAD=tpch bash tools/workloads/run_modes.sh
#   WORKLOAD=ssb  MODE=shared QUERIES="Q1_1" bash tools/workloads/run_modes.sh
#   WORKLOAD=stack TPB=128 IPT=8 bash tools/workloads/run_modes.sh
#
# MODE: shared (--decom_shared, default) | register (--decom_register) |
#       global-stage (--decom_global_stage) | pre-decom (--pre_decom).
# Decode geometry selectable via TPB / IPT / TILE (default 256 / 4 / 1024; TPB*IPT==TILE).
# Global-stage row-group concurrency is selectable via GLOBAL_STAGE_GRID_ROWS
# (default 16).
# Set INCLUDE_UNCOM=0 when a surrounding sweep has already measured the
# uncompressed reference and only compressed rows are required.
#
# Dataset selection is by repointing the fixture's inputs/<ds> symlink per
# codec. Dataset paths come from PROTEUS_DATA_ROOT or the per-dataset override
# variables defined in common.sh.
set -uo pipefail
unset PROTEUS_RESULT_DUMP_DIR
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/a100_gpu_buffer_policy.sh"
acquire_workload_fixture_lock || exit $?
# Do not leave multi-GB core files behind if a benchmark process fails.
ulimit -c 0
LOG_DIR="${LOG_DIR:-$WORKLOAD_OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKLOAD="${WORKLOAD:-tpch}"
case "$WORKLOAD" in
  ssb)
    FIX=SSBInGPU; LN=ssbm100; BIN="$BUILD/apps/benchmarks/olap/benchmark-ssb/proteus-benchmark-ssb"
    DEF_Q="Q1_1 Q2_1 Q3_1 Q4_1"
    NONE_DS="$SSB_NONE_DS"; RFOR_DS="$SSB_RFOR_DS"
    RFOR512_DS="$SSB_RFOR512_DS"; FLBP_DS="$SSB_FLBP_DS" ;;
  tpch)
    FIX=TPCHInGPU; LN=tpch100; BIN="$BUILD/apps/benchmarks/olap/benchmark-tpch/proteus-benchmark-tpch"
    DEF_Q="Q5 Q12"
    NONE_DS="$TPCH_NONE_DS"; RFOR_DS="$TPCH_RFOR_DS"
    RFOR512_DS="$TPCH_RFOR512_DS"; FLBP_DS="$TPCH_FLBP_DS" ;;
  stack)
    FIX=StackInGPU; LN=stack; BIN="$BUILD/apps/benchmarks/olap/benchmark-stack/proteus-benchmark-stack"
    DEF_Q="Q11 Q14"
    NONE_DS="$STACK_NONE_DS"; RFOR_DS="$STACK_RFOR_DS"
    RFOR512_DS="$STACK_RFOR512_DS"; FLBP_DS="$STACK_FLBP_DS" ;;
  *) echo "WORKLOAD must be ssb|tpch|stack" >&2; exit 2 ;;
esac
MODE="${MODE:-shared}"; CODECS="${CODECS:-RFOR FLBP}"; QUERIES="${QUERIES:-$DEF_Q}"
for query in $QUERIES; do
  case "$WORKLOAD:$query" in
    ssb:Q1_1|ssb:Q2_1|ssb:Q3_1|ssb:Q4_1|tpch:Q5|tpch:Q12|stack:Q11|stack:Q14) ;;
    *) echo "Query $query is outside the published $WORKLOAD workload." >&2; exit 2 ;;
  esac
done
TPB="${TPB:-256}"; IPT="${IPT:-4}"; TILE="${TILE:-1024}"
MAX_GRID_BLOCKS="${MAX_GRID_BLOCKS:-0}"
CPU_BUFFERS="${CPU_BUFFERS:-0.01}"
GPU_BUFFERS="${GPU_BUFFERS:-auto}"
PRE_DECOM_GPU_BUFFERS="${PRE_DECOM_GPU_BUFFERS:-0.2}"
UNCOM_GPU_BUFFERS="${UNCOM_GPU_BUFFERS:-0.2}"
VERIFY_RESULTS="${VERIFY_RESULTS:-0}"
GLOBAL_STAGE_GRID_ROWS="${GLOBAL_STAGE_GRID_ROWS:-16}"
INCLUDE_UNCOM="${INCLUDE_UNCOM:-1}"
case "$MODE" in
  shared) MFLAG="--decom_shared=true" ;;
  register) MFLAG="--decom_register=true" ;;
  global-stage) MFLAG="--decom_global_stage=true --global_stage_grid_rows=$GLOBAL_STAGE_GRID_ROWS" ;;
  pre-decom) MFLAG="--pre_decom=true" ;;
  *) echo "MODE must be shared|register|global-stage|pre-decom" >&2; exit 2 ;;
esac
GEOM="--tile_size=$TILE --threads_per_block=$TPB --items_per_thread=$IPT --max_grid_blocks=$MAX_GRID_BLOCKS --use_metadata_in_block=true"
OUT="${OUT:-$LOG_DIR/${WORKLOAD}_${MODE}_${TPB}x${IPT}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT" "$PELAGO/inputs"; cd "$PELAGO"; export LD_LIBRARY_PATH="$(proteus_library_path)"
bytes_in_dir(){ find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
# RFOR runs at tile 512 or 1024 (different on-disk datasets); FLBP vector is fixed 1024.
ds_for(){ case "$1" in
  NONE) echo "$NONE_DS";;
  RFOR) [ "$TILE" = 512 ] && echo "${RFOR512_DS:-}" || echo "$RFOR_DS";;
  FLBP) echo "$FLBP_DS";;
esac; }

echo "=== run_modes: $WORKLOAD MODE=$MODE geom=${TPB}x${IPT} codecs=[$CODECS] queries=[$QUERIES] ==="
: > "$OUT/progress.log"
failures=0
RUN_CODECS="$CODECS"
[ "$INCLUDE_UNCOM" = 1 ] && RUN_CODECS="NONE $RUN_CODECS"
for q in $QUERIES; do
  query_gpu_buffers="$GPU_BUFFERS"
  if [ "$query_gpu_buffers" = auto ]; then
    query_gpu_buffers=$(a100_gpu_buffers_for_query "$WORKLOAD" "$q")
  fi
  for codec in $RUN_CODECS; do
    [ "$codec" = NONE ] && { fl="--compression_type=NONE"; tag="uncom_$q"; codec_gpu_buffers="$UNCOM_GPU_BUFFERS"; } || { fl="--in_gpu=true --one_kernel=true --compression_type=$codec $MFLAG $GEOM"; tag="$(echo $codec|tr A-Z a-z)_${MODE}_$q"; [ "$MODE" = pre-decom ] && codec_gpu_buffers="$PRE_DECOM_GPU_BUFFERS" || codec_gpu_buffers="$query_gpu_buffers"; }
    ds="$(ds_for "$codec")"
    if [ -z "$ds" ] || [ ! -f "$ds/catalog.json" ]; then
      echo "$tag FAILED (no dataset: ${ds:-none})" | tee -a "$OUT/progress.log"
      failures=$((failures + 1))
      continue
    fi
    ln -sfnT "$ds" "$PELAGO/inputs/$LN"
    dataset_flag=""
    [ "$WORKLOAD" = stack ] && dataset_flag="--stack_dataset_path=$ds"
    dump="$OUT/$tag"; rm -rf "$dump"; mkdir -p "$dump"
    set +e
    numactl --interleave=all "$BIN" \
      --cpu_buffers="$CPU_BUFFERS" --gpu_buffers="$codec_gpu_buffers" $fl \
      $dataset_flag \
      --benchmark_filter="$FIX/INGPU_$q/" \
      --benchmark_color=false --print_generated_code=false > "$OUT/$tag.log" 2>&1
    rc=$?
    t=$(awk -v pat="INGPU_${q}/" \
      'index($0, pat) && index($0, "real_time") && $3 == "ms" { print $2; exit }' \
      "$OUT/$tag.log" || true)
    if [ "$VERIFY_RESULTS" = 1 ]; then
      echo "WARN: VERIFY_RESULTS=1 requested, but PROTEUS_RESULT_DUMP_DIR is disabled in this timing runner" >> "$OUT/$tag.log"
    fi
    if [ "$rc" -ne 0 ]; then
      id=FAILED_RC${rc}
      failures=$((failures + 1))
    elif [ -z "$t" ]; then
      id=FAILED_NO_TIMING
      failures=$((failures + 1))
    else
      id=NOT_CHECKED
    fi
    echo "$tag time=${t:-NA} ms result_vs_none=$id" | tee -a "$OUT/progress.log"
  done
done
if [ "$failures" -ne 0 ]; then
  echo "RUN_FAILED failures=$failures (log: $OUT/progress.log)" | tee -a "$OUT/progress.log" >&2
  exit 1
fi
echo "ALL_DONE  (log: $OUT/progress.log)" | tee -a "$OUT/progress.log"
