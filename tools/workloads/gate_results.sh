#!/usr/bin/env bash
# Separate correctness matrix; never use captured runs as timings.
set -uo pipefail
ulimit -c 0

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
source "$HERE/a100_gpu_buffer_policy.sh"

WORKLOADS=${WORKLOADS:-ssb tpch stack}
WINDOWS=${WINDOWS:-16 18}
LAYOUTS=${LAYOUTS:-UNCOM RFOR1024 FLBP1024 RFOR512}
MODES=${MODES:-shared register pre-decom global-stage}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

acquire_workload_fixture_lock || exit $?

mkdir -p "$WORKLOAD_OUTPUT_ROOT" "$PELAGO/inputs"
log_parent=${LOG_DIR:-$WORKLOAD_OUTPUT_ROOT}
mkdir -p "$log_parent"
LOG_DIR=$(mktemp -d "$log_parent/correctness-XXXXXX")
export LD_LIBRARY_PATH="$(proteus_library_path)"

cd "$PELAGO"
failures=0
query_mismatches=0
infrastructure_failures=0
count=0
printf 'workload\tquery\tcodec\ttile\tgeometry\tmode\twindow\tstatus\tfailure_class\tprocess_rc\tcomparison\tdetail\tresult_file\tlog_file\n' > "$LOG_DIR/results.tsv"

cat > "$LOG_DIR/manifest.txt" <<EOF
commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
date=$(date --iso-8601=seconds)
workloads=$WORKLOADS
windows=$WINDOWS
layouts=$LAYOUTS
modes=$MODES
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
build=$BUILD
pelago=$PELAGO
lock_file=$WORKLOAD_FIXTURE_LOCK
EOF

record_result() {
  local workload=$1 query=$2 codec=$3 tile=$4 geometry=$5 mode=$6 window=$7
  local status=$8 failure_class=$9 process_rc=${10} comparison=${11}
  local detail=${12} result_file=${13} log_file=${14}
  detail=${detail//$'\t'/ }
  detail=${detail//$'\n'/ }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workload" "$query" "$codec" "$tile" "$geometry" "$mode" "$window" \
    "$status" "$failure_class" "$process_rc" "$comparison" "$detail" \
    "$result_file" "$log_file" | tee -a "$LOG_DIR/results.tsv"
}

for wl in $WORKLOADS; do
  case "$wl" in
    ssb) fixture=SSBInGPU; link=ssbm100; queries=${SSB_QUERIES:-'Q1_1 Q2_1 Q3_1 Q4_1'} ;;
    tpch) fixture=TPCHInGPU; link=tpch100; queries=${TPCH_QUERIES:-'Q5 Q12'} ;;
    stack) fixture=StackInGPU; link=stack; queries=${STACK_QUERIES:-'Q11 Q14'} ;;
    *) echo "Invalid workload: $wl" >&2; exit 2 ;;
  esac

  for query in $queries; do
    case "$wl:$query" in
      ssb:Q1_1|ssb:Q2_1|ssb:Q3_1|ssb:Q4_1|tpch:Q5|tpch:Q12|stack:Q11|stack:Q14) ;;
      *) echo "Query $query is outside the published $wl workload." >&2; exit 2 ;;
    esac
  done

  [[ ! -e inputs/$link || -L inputs/$link ]] || {
    echo "Refusing to replace inputs/$link" >&2
    exit 2
  }
  original=$(readlink "inputs/$link" || true)
  restore() {
    [[ ! -L inputs/$link ]] || rm "inputs/$link"
    [[ -z $original ]] || ln -s "$original" "inputs/$link"
  }
  trap restore EXIT
  trap 'restore; exit 130' INT
  trap 'restore; exit 143' TERM

  binary="$BUILD/apps/benchmarks/olap/benchmark-$wl/proteus-benchmark-$wl"
  for query in $queries; do
    reference=''
    for layout in $LAYOUTS; do
      case "$layout" in
        UNCOM) codec=NONE; tile=0; geoms=0x0; modes=uncom; suffix=NONE ;;
        RFOR1024) codec=RFOR; tile=1024; geoms='256x4 128x8'; modes=$MODES; suffix=RFOR ;;
        FLBP1024) codec=FLBP; tile=1024; geoms='256x4 128x8'; modes=$MODES; suffix=FLBP ;;
        RFOR512) codec=RFOR; tile=512; geoms='128x4 256x2'; modes=$MODES; suffix=RFOR512 ;;
        *) echo "Invalid layout: $layout" >&2; exit 2 ;;
      esac
      ds_var="${wl^^}_${suffix}_DS"
      ds=$(realpath -m "${!ds_var}")

      for geom in $geoms; do
        for mode in $modes; do
          windows=0
          [[ $mode != global-stage ]] || windows=$WINDOWS
          [[ -n $windows ]] || { echo 'Empty WINDOWS' >&2; exit 2; }

          for window in $windows; do
            case "$mode" in uncom|shared|register|pre-decom|global-stage) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
            [[ $window =~ ^[0-9]+$ ]] || { echo "Invalid window: $window" >&2; exit 2; }
            [[ $mode != global-stage || $window -gt 0 ]] || { echo 'Global-stage window must be positive' >&2; exit 2; }

            dump="$LOG_DIR/${wl}_${query}_${layout}_${geom}_${mode}_w${window}"
            log_file="$dump.log"
            command_file="$dump.command"
            mkdir "$dump"
            count=$((count + 1))

            status=FAIL
            failure_class=infrastructure
            process_rc=not_run
            comparison=not_run
            detail=''
            result_file=''
            args=(--compression_type="$codec")
            buffers=${UNCOM_GPU_BUFFERS:-0.2}

            if [[ $codec != NONE ]]; then
              case "$mode" in
                shared) mode_flags=(--decom_shared=true --decom_register=false --pre_decom=false --decom_global_stage=false) ;;
                register) mode_flags=(--decom_shared=false --decom_register=true --pre_decom=false --decom_global_stage=false) ;;
                pre-decom) mode_flags=(--decom_shared=false --decom_register=false --pre_decom=true --decom_global_stage=false) ;;
                global-stage) mode_flags=(--decom_shared=false --decom_register=false --pre_decom=false --decom_global_stage=true) ;;
              esac
              args+=(--in_gpu=true --one_kernel=true "${mode_flags[@]}"
                --precompute_rl=false --tile_size="$tile"
                --threads_per_block="${geom%x*}" --items_per_thread="${geom#*x}"
                --use_metadata_in_block=true --max_grid_blocks="${MAX_GRID_BLOCKS:-0}")
              [[ $mode != global-stage ]] || args+=(--global_stage_grid_rows="$window")
              buffers=${GPU_BUFFERS:-auto}
              [[ $buffers != auto ]] || buffers=$(a100_gpu_buffers_for_query "$wl" "$query")
              [[ $mode != pre-decom ]] || buffers=${PRE_DECOM_GPU_BUFFERS:-0.2}
            fi
            [[ $wl != stack ]] || args+=(--stack_dataset_path="$ds")

            command=(numactl --interleave=all "$binary"
              --cpu_buffers="${CPU_BUFFERS:-0.01}" --gpu_buffers="$buffers"
              --benchmark_filter="^$fixture/INGPU_$query/"
              --benchmark_color=false --print_generated_code=false "${args[@]}")
            printf '%q ' "${command[@]}" > "$command_file"
            printf '\ndataset=%q\n' "$ds" >> "$command_file"

            if [[ ! -x $binary ]]; then
              echo "Missing benchmark binary: $binary" > "$log_file"
              detail=missing_binary
            elif [[ ! -f $ds/catalog.json ]]; then
              echo "Missing dataset: $ds" > "$log_file"
              detail=missing_dataset
            else
              ln -sfnT "$ds" "inputs/$link"
              if PROTEUS_RESULT_DUMP_DIR="$dump" "${command[@]}" > "$log_file" 2>&1; then
                process_rc=0
              else
                process_rc=$?
              fi

              mapfile -t files < <(find "$dump" -mindepth 1 -maxdepth 1 -type f -print)
              output_detail=''
              if [[ ${#files[@]} -eq 0 ]]; then
                output_detail=no_output
              elif [[ ${#files[@]} -gt 1 ]]; then
                output_detail=multiple_outputs_${#files[@]}
              elif [[ ! -s ${files[0]} ]]; then
                output_detail=empty_output
              else
                result_file=${files[0]}
                if python3 "$HERE/compare_results.py" --validate "$wl" "$query" "$result_file" >> "$log_file" 2>&1; then
                  validate_rc=0
                else
                  validate_rc=$?
                  output_detail=invalid_schema_rc${validate_rc}
                fi
              fi

              # A benchmark may fail after emitting a complete result (for example,
              # during process teardown). Keep the run failed, but retain a valid
              # uncompressed result so candidate outputs can still be diagnosed.
              if [[ $codec == NONE && -n $result_file && -z $output_detail ]]; then
                reference=$result_file
                comparison=reference
              fi

              if [[ $codec != NONE && -n $result_file && -z $output_detail && -n $reference ]]; then
                if python3 "$HERE/compare_results.py" "$wl" "$query" "$reference" "$result_file" >> "$log_file" 2>&1; then
                  compare_rc=0
                  comparison=match
                else
                  compare_rc=$?
                  if [[ $compare_rc -eq 1 ]]; then
                    comparison=mismatch
                  elif [[ $compare_rc -eq 3 ]]; then
                    comparison=schema_mismatch
                  else
                    comparison=error_rc${compare_rc}
                  fi
                fi
              fi

              if [[ $process_rc != 0 ]]; then
                detail=process_rc_${process_rc}
                [[ -z $output_detail ]] || detail+="+$output_detail"
              elif [[ -n $output_detail ]]; then
                detail=$output_detail
                if [[ $output_detail == invalid_schema_rc3 ]]; then
                  failure_class=query_mismatch
                fi
              elif [[ $codec == NONE ]]; then
                status=REFERENCE
                failure_class=none
                detail=ok
              elif [[ -z $reference ]]; then
                detail=no_valid_reference
              elif [[ $comparison == match ]]; then
                status=PASS
                failure_class=none
                detail=ok
              elif [[ $comparison == mismatch || $comparison == schema_mismatch ]]; then
                failure_class=query_mismatch
                detail=$comparison
              else
                detail=compare_${comparison}
              fi
            fi

            if [[ $status == FAIL ]]; then
              failures=$((failures + 1))
              if [[ $failure_class == query_mismatch ]]; then
                query_mismatches=$((query_mismatches + 1))
              else
                infrastructure_failures=$((infrastructure_failures + 1))
              fi
              echo "FAIL [$failure_class/$detail]: $wl $query $codec t$tile $geom $mode w$window; see $log_file" >&2
            fi
            record_result "$wl" "$query" "$codec" "$tile" "$geom" "$mode" "$window" \
              "$status" "$failure_class" "$process_rc" "$comparison" "$detail" \
              "$result_file" "$log_file"
          done
        done
      done
    done
  done
  restore
  trap - EXIT INT TERM
done

[[ $count -gt 0 ]] || exit 2
echo "Correctness gate: $count runs, $failures failures ($query_mismatches query mismatches, $infrastructure_failures infrastructure); logs: $LOG_DIR"
[[ $failures == 0 ]]
