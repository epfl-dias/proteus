#!/usr/bin/env bash
# Compare resident, uncompressed execution with one CpuToGpu kernel over all
# row groups against the original one-kernel-per-row-group path.
# Run one query per process because the InGPU fixture owns resident buffers
# until process teardown.
set -euo pipefail

ulimit -c 0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"
acquire_workload_fixture_lock || exit $?
WORKLOAD="${WORKLOAD:-ssb}"
case "$WORKLOAD" in
  ssb)
    FIXTURE=SSBInGPU
    INPUT_NAME=ssbm100
    DEFAULT_BIN="$PELAGO/proteus-benchmark-ssb"
    DEFAULT_DATASET="$SSB_NONE_DS"
    DEFAULT_QUERIES="Q1_1 Q2_1 Q3_1 Q4_1"
    ;;
  tpch)
    FIXTURE=TPCHInGPU
    INPUT_NAME=tpch100
    DEFAULT_BIN="$PELAGO/proteus-benchmark-tpch"
    DEFAULT_DATASET="$TPCH_NONE_DS"
    DEFAULT_QUERIES="Q5 Q12"
    ;;
  stack)
    FIXTURE=StackInGPU
    INPUT_NAME=stack
    DEFAULT_BIN="$PELAGO/proteus-benchmark-stack"
    DEFAULT_DATASET="$STACK_NONE_DS"
    DEFAULT_QUERIES="Q11 Q14"
    ;;
  *)
    echo "WORKLOAD must be ssb|tpch|stack" >&2
    exit 2
    ;;
esac
BIN="${BIN:-$DEFAULT_BIN}"
DATASET="${DATASET:-$DEFAULT_DATASET}"
OUT="${OUT:-$WORKLOAD_OUTPUT_ROOT/one_kernel_sensitivity_a100_${WORKLOAD}_$(date +%Y%m%d_%H%M%S)}"
QUERIES="${QUERIES:-$DEFAULT_QUERIES}"
for query in $QUERIES; do
  case "$WORKLOAD:$query" in
    ssb:Q1_1|ssb:Q2_1|ssb:Q3_1|ssb:Q4_1|tpch:Q5|tpch:Q12|stack:Q11|stack:Q14) ;;
    *) echo "Query $query is outside the published $WORKLOAD workload." >&2; exit 2 ;;
  esac
done
CPU_BUFFERS="${CPU_BUFFERS:-0.01}"
GPU_BUFFERS="${GPU_BUFFERS:-0.2}"
MODE_ORDER="${MODE_ORDER:-false true}"
GIT_COMMIT="${GIT_COMMIT:-unknown}"

[[ -x "$BIN" ]] || { echo "Missing benchmark binary: $BIN" >&2; exit 2; }
[[ -f "$DATASET/catalog.json" ]] || {
  echo "Missing uncompressed $WORKLOAD dataset/catalog: $DATASET" >&2
  exit 2
}

mkdir -p "$OUT" "$PELAGO/inputs"
ln -sfnT "$DATASET" "$PELAGO/inputs/$INPUT_NAME"
cd "$PELAGO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$(proteus_library_path)"
unset PROTEUS_RESULT_DUMP_DIR

{
  echo "git_commit=$GIT_COMMIT"
  echo "workload=$WORKLOAD"
  echo "fixture=$FIXTURE"
  echo "dataset=$(readlink -f "inputs/$INPUT_NAME")"
  echo "queries=$QUERIES"
  echo "compression_type=NONE"
  echo "in_gpu=true"
  echo "cpu_buffers=$CPU_BUFFERS"
  echo "gpu_buffers=$GPU_BUFFERS"
  echo "mode_order=$MODE_ORDER"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null || true
} > "$OUT/metadata.txt"

printf 'query,one_kernel,time_ms,return_code,log\n' > "$OUT/timings.csv"

for query in $QUERIES; do
  # Pair the two modes per query to keep machine conditions close. Benchmark
  # setup and the fixture warmup are outside the reported timed iterations.
  for one_kernel in $MODE_ORDER; do
    tag="${query}_one_kernel_${one_kernel}"
    log="$OUT/$tag.log"
    echo "RUN $query one_kernel=$one_kernel"
    set +e
    dataset_args=()
    if [[ "$WORKLOAD" == stack ]]; then
      dataset_args+=(--stack_dataset_path="$DATASET")
    fi
    numactl --interleave=all "$BIN" \
      --compression_type=NONE \
      --in_gpu=true \
      --one_kernel="$one_kernel" \
      --cpu_buffers="$CPU_BUFFERS" \
      --gpu_buffers="$GPU_BUFFERS" \
      "${dataset_args[@]}" \
      --benchmark_filter="$FIXTURE/INGPU_${query}/" \
      --benchmark_color=false \
      --print_generated_code=false > "$log" 2>&1
    rc=$?
    set -e

    timing=$(awk -v pattern="INGPU_${query}/" \
      'index($0, pattern) && index($0, "real_time") { print $2 "," $3; exit }' \
      "$log")
    time_ms=${timing%,*}
    unit=${timing#*,}
    if [[ -z "$timing" || "$unit" != "ms" ]]; then
      echo "No millisecond timing for $tag (rc=$rc); see $log" >&2
      exit 1
    fi
    if [[ "$rc" -ne 0 ]]; then
      echo "Benchmark failed for $tag (rc=$rc); see $log" >&2
      exit "$rc"
    fi
    printf '%s,%s,%s,%s,%s\n' \
      "$query" "$one_kernel" "$time_ms" "$rc" "$(basename "$log")" \
      >> "$OUT/timings.csv"
  done
done

awk -F, -v queries="$QUERIES" '
  BEGIN {
    OFS=",";
    count=split(queries, order, " ");
    print "query,one_kernel_false_ms,one_kernel_true_ms,reduction_pct";
  }
  NR > 1 {
    time[$1 SUBSEP $2]=$3;
  }
  END {
    for (i=1; i<=count; i++) {
      q=order[i];
      off=time[q SUBSEP "false"];
      on=time[q SUBSEP "true"];
      reduction=100.0*(off-on)/off;
      print q, off, on, sprintf("%.6f", reduction);
      sum_off+=off;
      sum_on+=on;
      sum_reduction+=reduction;
    }
    print "ARITHMETIC_MEAN", "", "", sprintf("%.6f", sum_reduction/count);
    print "SUMMED_RUNTIME", sprintf("%.6f", sum_off), sprintf("%.6f", sum_on), \
      sprintf("%.6f", 100.0*(sum_off-sum_on)/sum_off);
  }
' "$OUT/timings.csv" > "$OUT/summary.csv"

cat "$OUT/summary.csv"
echo "ALL_DONE output=$OUT"
