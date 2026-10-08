#!/usr/bin/env bash
set -uo pipefail
ulimit -c 0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"
LOG_DIR="${LOG_DIR:-$WORKLOAD_OUTPUT_ROOT/rfor512_result_gate}"
WORKLOAD="${WORKLOAD:-all}"
CPU_BUFFERS="${CPU_BUFFERS:-0.01}"
NONE_GPU_BUFFERS="${NONE_GPU_BUFFERS:-0.2}"
RFOR_GPU_BUFFERS="${RFOR_GPU_BUFFERS:-0.6}"
TPB="${TPB:-128}"
IPT="${IPT:-4}"

mkdir -p "$LOG_DIR" "$PELAGO/inputs"
cd "$PELAGO"
export LD_LIBRARY_PATH="$(proteus_library_path)"

failures=0

run_query() {
  local binary=$1 fixture=$2 query=$3 dataset=$4 dump_dir=$5
  local gpu_buffers=$6
  shift 6
  mkdir -p "$dump_dir"
  set +e
  PROTEUS_RESULT_DUMP_DIR="$dump_dir" numactl --interleave=all "$binary" \
    --cpu_buffers="$CPU_BUFFERS" --gpu_buffers="$gpu_buffers" \
    --benchmark_filter="$fixture/INGPU_$query/" \
    --benchmark_color=false --print_generated_code=false \
    "$@" > "$dump_dir.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 134 ]; then
    echo "$query failed with rc=$rc (log: $dump_dir.log)" >&2
    return 1
  fi
  local result_count
  result_count=$(find "$dump_dir" -maxdepth 1 -type f | wc -l)
  if [ "$result_count" -ne 1 ]; then
    echo "$query produced $result_count result files in $dump_dir" >&2
    return 1
  fi
}

gate_workload() {
  local name=$1 fixture=$2 link_name=$3 binary=$4 none_ds=$5 rfor_ds=$6
  local queries=$7 dataset_flag_name=$8

  if [ ! -f "$none_ds/catalog.json" ] || [ ! -f "$rfor_ds/catalog.json" ]; then
    echo "$name dataset missing: NONE=$none_ds RFOR512=$rfor_ds" >&2
    return 1
  fi

  for query in $queries; do
    local query_dir="$LOG_DIR/$name/$query"
    local none_dump="$query_dir/none"
    local rfor_dump="$query_dir/rfor512"
    local dataset_args=()

    ln -sfnT "$none_ds" "$PELAGO/inputs/$link_name"
    if [ -n "$dataset_flag_name" ]; then
      dataset_args=("$dataset_flag_name=$none_ds")
    fi
    run_query "$binary" "$fixture" "$query" "$none_ds" "$none_dump" \
      "$NONE_GPU_BUFFERS" --compression_type=NONE "${dataset_args[@]}" || {
        failures=$((failures + 1)); continue;
      }

    ln -sfnT "$rfor_ds" "$PELAGO/inputs/$link_name"
    if [ -n "$dataset_flag_name" ]; then
      dataset_args=("$dataset_flag_name=$rfor_ds")
    fi
    run_query "$binary" "$fixture" "$query" "$rfor_ds" "$rfor_dump" \
      "$RFOR_GPU_BUFFERS" --in_gpu=true --one_kernel=true \
      --compression_type=RFOR --decom_op=true --tile_size=512 \
      --threads_per_block="$TPB" --items_per_thread="$IPT" \
      --use_metadata_in_block=true "${dataset_args[@]}" || {
        failures=$((failures + 1)); continue;
      }

    local none_file rfor_file
    none_file=$(find "$none_dump" -maxdepth 1 -type f -print -quit)
    rfor_file=$(find "$rfor_dump" -maxdepth 1 -type f -print -quit)
    if cmp -s "$none_file" "$rfor_file"; then
      echo "$name $query IDENTICAL"
    elif cmp -s <(LC_ALL=C sort "$none_file") <(LC_ALL=C sort "$rfor_file"); then
      echo "$name $query IDENTICAL_UNORDERED"
    else
      echo "$name $query MISMATCH" >&2
      failures=$((failures + 1))
    fi
  done
}

if [ "$WORKLOAD" = all ] || [ "$WORKLOAD" = tpch ]; then
  gate_workload tpch TPCHInGPU tpch100 \
    "$BUILD/apps/benchmarks/olap/benchmark-tpch/proteus-benchmark-tpch" \
    "$TPCH_NONE_DS" \
    "$TPCH_RFOR512_DS" \
    "Q5 Q12" ""
fi

if [ "$WORKLOAD" = all ] || [ "$WORKLOAD" = stack ]; then
  gate_workload stack StackInGPU stack \
    "$BUILD/apps/benchmarks/olap/benchmark-stack/proteus-benchmark-stack" \
    "$STACK_NONE_DS" \
    "$STACK_RFOR512_DS" \
    "Q11 Q14" "--stack_dataset_path"
fi

if [ "$failures" -ne 0 ]; then
  echo "RFOR512 result gate: FAIL ($failures queries)" >&2
  exit 1
fi
echo "RFOR512 result gate: PASS"
