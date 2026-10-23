#!/usr/bin/env bash
# Sweep for the bounded global-memory staging placement.
# Each Google Benchmark process performs ten timed iterations. The sweep keeps
# result dumping disabled and records one process per (query, codec, window).
#
# A100 example:
#   OUT_BASE=/data/proteus-logs/a100-global-stage \
#   bash tools/workloads/run_global_stage_sweep.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO_ROOT/tools/workloads/common.sh"
source "$REPO_ROOT/tools/workloads/a100_gpu_buffer_policy.sh"
OUT_BASE="${OUT_BASE:-$WORKLOAD_OUTPUT_ROOT/global_stage_sweep_$(date +%Y%m%d_%H%M%S)}"
WINDOWS="${WINDOWS:-18}"
WORKLOADS="${WORKLOADS:-ssb tpch stack}"
CODECS="${CODECS:-RFOR FLBP}"
TILE="${TILE:-1024}"
CONFIGS="${CONFIGS:-}"
GPU_BUFFERS="${GPU_BUFFERS:-auto}"
acquire_workload_fixture_lock || exit $?

queries_for() {
  case "$1" in
    ssb) echo "${SSB_QUERIES:-Q1_1 Q2_1 Q3_1 Q4_1}" ;;
    tpch) echo "${TPCH_QUERIES:-Q5 Q12}" ;;
    stack) echo "${STACK_QUERIES:-Q11 Q14}" ;;
    *) echo "unsupported workload: $1" >&2; return 2 ;;
  esac
}

geometry_for() {
  case "$1" in
    RFOR) echo "${RFOR_TPB:-256} ${RFOR_IPT:-4}" ;;
    FLBP) echo "${FLBP_TPB:-128} ${FLBP_IPT:-8}" ;;
    *) echo "unsupported codec: $1" >&2; return 2 ;;
  esac
}

if [[ -z "$CONFIGS" ]]; then
  for codec in $CODECS; do
    read -r tpb ipt <<< "$(geometry_for "$codec")"
    CONFIGS+="$codec:$TILE:$tpb:$ipt "
  done
fi

validate_config() {
  local codec=$1 tile=$2 tpb=$3 ipt=$4
  [[ "$codec" == RFOR || "$codec" == FLBP ]] || {
    echo "unsupported codec in CONFIGS: $codec" >&2
    return 2
  }
  (( tpb * ipt == tile )) || {
    echo "invalid geometry in CONFIGS: $codec:$tile:$tpb:$ipt" >&2
    return 2
  }
  [[ "$codec" != FLBP || "$tile" == 1024 ]] || {
    echo "FLBP requires tile size 1024" >&2
    return 2
  }
}

mkdir -p "$OUT_BASE/runs"
commit="${GIT_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)}"
cat > "$OUT_BASE/manifest.txt" <<EOF
commit=$commit
date=$(date --iso-8601=seconds)
workloads=$WORKLOADS
windows=$WINDOWS
codecs=$CODECS
configs=$CONFIGS
gpu_buffers=$GPU_BUFFERS
ssb_queries=$(queries_for ssb)
tpch_queries=$(queries_for tpch)
stack_queries=$(queries_for stack)
rfor_geometry=$(geometry_for RFOR)
flbp_geometry=$(geometry_for FLBP)
max_grid_blocks=${MAX_GRID_BLOCKS:-0}
EOF
if [[ "$GPU_BUFFERS" == auto ]]; then
  for workload in $WORKLOADS; do
    for query in $(queries_for "$workload"); do
      printf 'gpu_buffers_%s_%s=%s\n' \
        "$workload" "$query" \
        "$(a100_gpu_buffers_for_query "$workload" "$query")" \
        >> "$OUT_BASE/manifest.txt"
    done
  done
fi

for workload in $WORKLOADS; do
  for query in $(queries_for "$workload"); do
    query_gpu_buffers="$GPU_BUFFERS"
    if [[ "$query_gpu_buffers" == auto ]]; then
      query_gpu_buffers=$(a100_gpu_buffers_for_query "$workload" "$query")
    fi
    for config in $CONFIGS; do
      IFS=: read -r codec tile tpb ipt <<< "$config"
      validate_config "$codec" "$tile" "$tpb" "$ipt"
      for window in $WINDOWS; do
        out="$OUT_BASE/runs/$workload/$query/$codec/t$tile/${tpb}x${ipt}/w$window"
        mkdir -p "$out"
        echo "=== $workload $query $codec T=$tile W=$window ${tpb}x${ipt} ===" | tee -a "$OUT_BASE/progress.log"
        BUILD="$BUILD" PELAGO="$PELAGO" LLVM_LIB="$LLVM_LIB" \
          WORKLOAD="$workload" QUERIES="$query" CODECS="$codec" \
          MODE=global-stage INCLUDE_UNCOM=0 GLOBAL_STAGE_GRID_ROWS="$window" \
          TPB="$tpb" IPT="$ipt" TILE="$tile" GPU_BUFFERS="$query_gpu_buffers" \
          OUT="$out" \
          bash "$REPO_ROOT/tools/workloads/run_modes.sh"
        status_line=$(tail -n 2 "$out/progress.log" | head -n 1)
        echo "$status_line" | tee -a "$OUT_BASE/progress.log"
        if [[ "$status_line" == *"time=NA"* || "$status_line" == *"FAILED_RC"* ]]; then
          echo "FAILED: $workload $query $codec T=$tile ${tpb}x${ipt} W=$window" \
            | tee -a "$OUT_BASE/progress.log" >&2
          exit 4
        fi
      done
    done
  done
done

QDIR="$QDIR" python3 "$REPO_ROOT/tools/workloads/footprint.py" \
  > "$OUT_BASE/footprint.csv"
python3 "$REPO_ROOT/tools/workloads/assemble_global_stage_csv.py" \
  "$OUT_BASE" "$OUT_BASE/footprint.csv" > "$OUT_BASE/global-stage.csv"
echo "ALL_DONE csv=$OUT_BASE/global-stage.csv" | tee -a "$OUT_BASE/progress.log"
