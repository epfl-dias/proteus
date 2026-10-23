#!/usr/bin/env bash
set -euo pipefail

# Generate the RFOR and FLBP datasets used by the CB2T benchmark matrix from
# existing uncompressed Proteus datasets. Run this on the bare-metal host.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"
export LD_LIBRARY_PATH="$(proteus_library_path)"

GENERATOR=${GENERATOR:-$BUILD/tools/dataset-generation/proteus-generate-cb2t-dataset}
WORKLOADS=${WORKLOADS:-ssb tpch stack}
CODECS=${CODECS:-RFOR FLBP}
RFOR_TILES=${RFOR_TILES:-1024 512}
FORCE=${FORCE:-0}
STAGE_ROOT=${CB2T_STAGE_ROOT:-$WORKLOAD_OUTPUT_ROOT/logical-inputs}

dataset_path() {
  local workload=$1 codec=$2 tile=${3:-1024}
  case "$workload:$codec:$tile" in
    ssb:NONE:*) echo "$SSB_NONE_DS" ;;
    ssb:RFOR:1024) echo "$SSB_RFOR_DS" ;;
    ssb:RFOR:512) echo "$SSB_RFOR512_DS" ;;
    ssb:FLBP:1024) echo "$SSB_FLBP_DS" ;;
    tpch:NONE:*) echo "$TPCH_NONE_DS" ;;
    tpch:RFOR:1024) echo "$TPCH_RFOR_DS" ;;
    tpch:RFOR:512) echo "$TPCH_RFOR512_DS" ;;
    tpch:FLBP:1024) echo "$TPCH_FLBP_DS" ;;
    stack:NONE:*) echo "$STACK_NONE_DS" ;;
    stack:RFOR:1024) echo "$STACK_RFOR_DS" ;;
    stack:RFOR:512) echo "$STACK_RFOR512_DS" ;;
    stack:FLBP:1024) echo "$STACK_FLBP_DS" ;;
    *) echo "Unsupported dataset: $workload $codec tile=$tile" >&2; return 2 ;;
  esac
}

columns_file() {
  case "$1" in
    ssb) echo "" ;;
    tpch) echo "$SCRIPT_DIR/columns/tpch_q5_q12.txt" ;;
    stack) echo "$SCRIPT_DIR/columns/stack_q11_q14.txt" ;;
    *) return 2 ;;
  esac
}

generate_one() {
  local workload=$1 codec=$2 tile=$3
  local input_dir output_dir stage_dir columns algorithm
  input_dir=$(dataset_path "$workload" NONE)
  output_dir=$(dataset_path "$workload" "$codec" "$tile")
  stage_dir="$STAGE_ROOT/${workload}-${codec,,}-t${tile}"
  columns=$(columns_file "$workload")
  algorithm=${codec,,}

  local validate_args=(
    --reference-directory="$input_dir"
    --candidate-directory="$output_dir"
    --compression-algorithm="$codec"
  )
  [[ -n "$columns" ]] && validate_args+=(--columns-file="$columns")
  [[ "$codec" == RFOR ]] && validate_args+=(--tile-size="$tile")

  [[ -f "$input_dir/catalog.json" ]] || {
    echo "Missing $workload input catalog: $input_dir/catalog.json" >&2
    exit 1
  }
  if [[ "$FORCE" != 1 && -f "$output_dir/catalog.json" ]]; then
    if python3 "$SCRIPT_DIR/validate_partitioned_cardinality.py" "${validate_args[@]}"; then
      echo "[skip] $workload $codec tile=$tile: $output_dir"
      return
    fi
    echo "Existing output is incomplete or stale; set FORCE=1: $output_dir" >&2
    exit 1
  fi
  if [[ -e "$output_dir" || -e "$stage_dir" ]]; then
    [[ "$FORCE" == 1 ]] || {
      echo "Refusing to replace partial output; set FORCE=1: $output_dir" >&2
      exit 1
    }
    rm -rf "$output_dir" "$stage_dir"
  fi

  local column_args=()
  [[ -n "$columns" ]] && column_args+=(--columns-file="$columns")
  python3 "$SCRIPT_DIR/materialize_logical_columns.py" \
    --input-directory="$input_dir" \
    --output-directory="$stage_dir" \
    "${column_args[@]}"

  mkdir -p "$output_dir"
  cp "$input_dir/catalog.json" "$output_dir/catalog.json"
  local partition_args=(
    --input_directory="$stage_dir"
    --output_directories="$output_dir"
    --compression_algorithm="$algorithm"
    --compress_data
    --metadata_in_block=true
  )
  [[ -n "$columns" ]] && partition_args+=(--columns_file="$columns")
  [[ "$codec" == RFOR ]] && partition_args+=(--rfor_tile_size="$tile")
  "$GENERATOR" "${partition_args[@]}"

  python3 "$SCRIPT_DIR/validate_partitioned_cardinality.py" "${validate_args[@]}"
  rm -rf "$stage_dir"
}

[[ -x "$GENERATOR" ]] || {
  echo "Missing dataset generator: $GENERATOR" >&2
  exit 1
}
for workload in $WORKLOADS; do
  for codec in $CODECS; do
    if [[ "$codec" == RFOR ]]; then
      for tile in $RFOR_TILES; do
        generate_one "$workload" "$codec" "$tile"
      done
    else
      generate_one "$workload" "$codec" 1024
    fi
  done
done
