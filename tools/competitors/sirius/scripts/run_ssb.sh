#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-${TMPDIR:-/tmp}/proteus-competitors/sirius}"
SIRIUS_BUILD_DIR="${SIRIUS_BUILD_DIR:-${TMPDIR:-/tmp}/sirius-legacy-release}"
DUCKDB_BIN="${DUCKDB_BIN:-$SIRIUS_BUILD_DIR/duckdb}"
DUCKDB_OPEN_FLAGS="${DUCKDB_OPEN_FLAGS:--readonly}"
DB_PATH="${DB_PATH:-}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/ssb-runs/$(date +%Y%m%d_%H%M%S)}"
MODE="time"
QUERIES=()
ITERATIONS=11
GPU_CACHE_SIZE="${GPU_CACHE_SIZE:-auto}"
GPU_PROCESSING_SIZE="${GPU_PROCESSING_SIZE:-auto}"
PINNED_MEMORY_SIZE="${PINNED_MEMORY_SIZE:-32 GB}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
NSYS="${NSYS:-$(command -v nsys || true)}"
NCU="${NCU:-$(command -v ncu || true)}"
NCU_METRICS="${NCU_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block}"
NSYS_GPU_METRICS_DEVICE="${NSYS_GPU_METRICS_DEVICE:-1}"

usage() {
  cat <<EOF
Usage:
  $0 --db <ssb_reduced_integer.duckdb> [options] [q1.1 q2.1 q3.1 q4.1]

Options:
  --duckdb <path>       DuckDB binary. Default: $DUCKDB_BIN
  --out-dir <path>      Output directory. Default: timestamped ssb_integer_runs dir
  --mode <time|nsys|ncu|profile>
  --iterations <N>      Total iterations per query in time mode. Default: 11 (1 warmup + 10 measured)
  --gpu <N>             CUDA_VISIBLE_DEVICES value. Default: 0
  --gpu-cache <SIZE>    gpu_buffer_init cache size. Default: auto per query
  --gpu-proc <SIZE>     gpu_buffer_init processing size. Default: auto per query
  --pinned <SIZE>       gpu_buffer_init pinned memory size. Default: 32 GB

Auto Sirius pool sizes:
  q1.1, q2.1: --gpu-cache "14 GB" --gpu-proc "18 GB"
  q3.1:       --gpu-cache "10 GB" --gpu-proc "18 GB"
  q4.1:       --gpu-cache "14 GB" --gpu-proc "18 GB"

The per-query cache and processing-pool defaults match the paper runs.

Environment overrides:
  DUCKDB_OPEN_FLAGS, NSYS, NCU, NCU_METRICS, NSYS_GPU_METRICS_DEVICE

Examples:
  $0 --db /data/bench/ssb.duckdb
  $0 --mode nsys --db /data/bench/ssb.duckdb q2.1
  $0 --mode ncu --db /data/bench/ssb.duckdb --iterations 1 q1.1
  $0 --mode profile --db /data/bench/ssb.duckdb q1.1

Expected DB schema:
  Created by load_ssb.sh.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --duckdb)
      DUCKDB_BIN="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --gpu)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --gpu-cache)
      GPU_CACHE_SIZE="$2"
      shift 2
      ;;
    --gpu-proc)
      GPU_PROCESSING_SIZE="$2"
      shift 2
      ;;
    --pinned)
      PINNED_MEMORY_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    q1.1|Q1.1|1|1.1)
      QUERIES+=("q1.1")
      shift
      ;;
    q2.1|Q2.1|2|2.1)
      QUERIES+=("q2.1")
      shift
      ;;
    q3.1|Q3.1|3|3.1)
      QUERIES+=("q3.1")
      shift
      ;;
    q4.1|Q4.1|4|4.1)
      QUERIES+=("q4.1")
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "$DB_PATH" ]; then
  echo "ERROR: --db is required"
  usage
  exit 1
fi

if [ ${#QUERIES[@]} -eq 0 ]; then
  QUERIES=("q1.1" "q2.1" "q3.1" "q4.1")
fi

if [ "$MODE" != "time" ] && [ "$MODE" != "nsys" ] && [ "$MODE" != "ncu" ] && [ "$MODE" != "profile" ]; then
  echo "ERROR: --mode must be one of: time, nsys, ncu, profile"
  exit 1
fi

if [ ! -x "$DUCKDB_BIN" ]; then
  echo "ERROR: DuckDB binary not found or not executable: $DUCKDB_BIN"
  exit 1
fi

if [ ! -f "$DB_PATH" ]; then
  echo "ERROR: database not found: $DB_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"

export SIRIUS_DISABLE=1
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"

duckdb_scalar() {
  local sql="$1"
  "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" -csv -noheader -c "$sql" | tr -d '\r' | tail -n 1
}

require_table() {
  local table="$1"
  local exists
  exists="$(duckdb_scalar "SELECT count(*) FROM information_schema.tables WHERE table_name='${table}';")"
  if [ "$exists" != "1" ]; then
    echo "ERROR: required table '$table' not found in $DB_PATH"
    echo "       Build the DB with: scripts/load_ssb_reduced_integer.sh <db> <ssb_tbl_dir> <duckdb>"
    exit 1
  fi
}

require_table lineorder
require_table customer
require_table date_dim
require_table part
require_table supplier
require_table category_dict
require_table mfgr_dict
require_table region_dict

LINEORDER_ROWS="$(duckdb_scalar "SELECT count(*) FROM lineorder;")"
CUSTOMER_ROWS="$(duckdb_scalar "SELECT count(*) FROM customer;")"
DATE_ROWS="$(duckdb_scalar "SELECT count(*) FROM date_dim;")"
PART_ROWS="$(duckdb_scalar "SELECT count(*) FROM part;")"
SUPPLIER_ROWS="$(duckdb_scalar "SELECT count(*) FROM supplier;")"

Q21_CATEGORY_ID="$(duckdb_scalar "SELECT category_id FROM category_dict WHERE p_category='MFGR#12';")"
Q21_REGION_ID="$(duckdb_scalar "SELECT region_id FROM region_dict WHERE region='AMERICA';")"
Q31_REGION_ID="$(duckdb_scalar "SELECT region_id FROM region_dict WHERE region='ASIA';")"
Q41_MFGR1_ID="$(duckdb_scalar "SELECT mfgr_id FROM mfgr_dict WHERE p_mfgr='MFGR#1';")"
Q41_MFGR2_ID="$(duckdb_scalar "SELECT mfgr_id FROM mfgr_dict WHERE p_mfgr='MFGR#2';")"

if [ -z "$Q21_CATEGORY_ID" ] || [ -z "$Q21_REGION_ID" ] || [ -z "$Q31_REGION_ID" ] || [ -z "$Q41_MFGR1_ID" ] || [ -z "$Q41_MFGR2_ID" ]; then
  echo "ERROR: could not resolve required dictionary IDs"
  echo "  MFGR#12 category_id: ${Q21_CATEGORY_ID:-<empty>}"
  echo "  AMERICA region_id: ${Q21_REGION_ID:-<empty>}"
  echo "  ASIA region_id: ${Q31_REGION_ID:-<empty>}"
  echo "  MFGR#1 mfgr_id: ${Q41_MFGR1_ID:-<empty>}"
  echo "  MFGR#2 mfgr_id: ${Q41_MFGR2_ID:-<empty>}"
  exit 1
fi

query_sql() {
  local query="$1"
  case "$query" in
    q1.1)
      cat <<'SQL'
SELECT sum(lo_extendedprice * lo_discount) AS revenue
FROM lineorder
WHERE lo_orderdate > 19930000
  AND lo_orderdate < 19940000
  AND lo_discount BETWEEN 1 AND 3
  AND lo_quantity < 25
SQL
      ;;
    q2.1)
      cat <<SQL
SELECT sum(lo_revenue) AS revenue,
       d_year,
       p_brand1_id AS p_brand_id
FROM lineorder,
     date_dim,
     part,
     supplier
WHERE lo_orderdate = d_datekey
  AND lo_partkey = p_partkey
  AND lo_suppkey = s_suppkey
  AND p_category_id = ${Q21_CATEGORY_ID}
  AND s_region_id = ${Q21_REGION_ID}
GROUP BY d_year,
         p_brand1_id
ORDER BY d_year,
         p_brand1_id
SQL
      ;;
    q3.1)
      cat <<SQL
SELECT c_nation_id,
       s_nation_id,
       d_year,
       sum(lo_revenue) AS revenue
FROM customer,
     lineorder,
     supplier,
     date_dim
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_orderdate = d_datekey
  AND c_region_id = ${Q31_REGION_ID}
  AND s_region_id = ${Q31_REGION_ID}
  AND d_year >= 1992
  AND d_year <= 1997
GROUP BY c_nation_id,
         s_nation_id,
         d_year
ORDER BY d_year ASC,
         revenue DESC
SQL
      ;;
    q4.1)
      cat <<SQL
SELECT d_year,
       c_nation_id,
       sum(lo_revenue - lo_supplycost) AS profit
FROM date_dim,
     customer,
     supplier,
     part,
     lineorder
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_partkey = p_partkey
  AND lo_orderdate = d_datekey
  AND c_region_id = ${Q21_REGION_ID}
  AND s_region_id = ${Q21_REGION_ID}
  AND (p_mfgr_id = ${Q41_MFGR1_ID} OR p_mfgr_id = ${Q41_MFGR2_ID})
GROUP BY d_year,
         c_nation_id
ORDER BY d_year,
         c_nation_id
SQL
      ;;
    *)
      echo "ERROR: unsupported query: $query" >&2
      return 1
      ;;
  esac
}

mask_bytes_for_rows() {
  local rows="$1"
  local necessary=$(((rows + 7) / 8))
  echo $((64 * ((necessary + 63) / 64)))
}

bytes_to_gib() {
  awk -v bytes="$1" 'BEGIN { printf "%.6f", bytes / 1024 / 1024 / 1024 }'
}

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

query_base_tables() {
  local query="$1"
  case "$query" in
    q1.1) echo "lineorder" ;;
    q2.1) echo "lineorder;date_dim;part;supplier" ;;
    q3.1) echo "customer;lineorder;supplier;date_dim" ;;
    q4.1) echo "date_dim;customer;supplier;part;lineorder" ;;
  esac
}

query_base_columns() {
  local query="$1"
  case "$query" in
    q1.1)
      echo "lineorder.lo_orderdate:INT32;lineorder.lo_quantity:INT16;lineorder.lo_extendedprice:INT32;lineorder.lo_discount:INT16"
      ;;
    q2.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_orderdate:INT32;lineorder.lo_partkey:INT32;lineorder.lo_suppkey:INT32;date_dim.d_datekey:INT32;date_dim.d_year:INT16;part.p_partkey:INT32;part.p_category_id:INT16;part.p_brand1_id:INT16;supplier.s_suppkey:INT32;supplier.s_region_id:INT16"
      ;;
    q3.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_orderdate:INT32;lineorder.lo_custkey:INT32;lineorder.lo_suppkey:INT32;customer.c_custkey:INT32;customer.c_region_id:INT16;customer.c_nation_id:INT16;supplier.s_suppkey:INT32;supplier.s_region_id:INT16;supplier.s_nation_id:INT16;date_dim.d_datekey:INT32;date_dim.d_year:INT16"
      ;;
    q4.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_supplycost:INT32;lineorder.lo_custkey:INT32;lineorder.lo_suppkey:INT32;lineorder.lo_partkey:INT32;lineorder.lo_orderdate:INT32;customer.c_custkey:INT32;customer.c_region_id:INT16;customer.c_nation_id:INT16;supplier.s_suppkey:INT32;supplier.s_region_id:INT16;part.p_partkey:INT32;part.p_mfgr_id:INT16;date_dim.d_datekey:INT32;date_dim.d_year:INT16"
      ;;
  esac
}

query_fact_columns() {
  local query="$1"
  case "$query" in
    q1.1)
      echo "lineorder.lo_orderdate:INT32;lineorder.lo_quantity:INT16;lineorder.lo_extendedprice:INT32;lineorder.lo_discount:INT16"
      ;;
    q2.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_orderdate:INT32;lineorder.lo_partkey:INT32;lineorder.lo_suppkey:INT32"
      ;;
    q3.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_orderdate:INT32;lineorder.lo_custkey:INT32;lineorder.lo_suppkey:INT32"
      ;;
    q4.1)
      echo "lineorder.lo_revenue:INT32;lineorder.lo_supplycost:INT32;lineorder.lo_custkey:INT32;lineorder.lo_suppkey:INT32;lineorder.lo_partkey:INT32;lineorder.lo_orderdate:INT32"
      ;;
  esac
}

query_fact_bytes_per_row() {
  local query="$1"
  case "$query" in
    q1.1)
      echo $((4 + 2 + 4 + 2))
      ;;
    q2.1|q3.1)
      echo $((4 + 4 + 4 + 4))
      ;;
    q4.1)
      echo $((4 + 4 + 4 + 4 + 4 + 4))
      ;;
  esac
}

query_fact_logical_bytes() {
  local query="$1"
  echo $((LINEORDER_ROWS * $(query_fact_bytes_per_row "$query")))
}

query_logical_bytes() {
  local query="$1"
  case "$query" in
    q1.1)
      echo $((LINEORDER_ROWS * (4 + 2 + 4 + 2)))
      ;;
    q2.1)
      echo $((LINEORDER_ROWS * (4 + 4 + 4 + 4) + DATE_ROWS * (4 + 2) + PART_ROWS * (4 + 2 + 2) + SUPPLIER_ROWS * (4 + 2)))
      ;;
    q3.1)
      echo $((LINEORDER_ROWS * (4 + 4 + 4 + 4) + CUSTOMER_ROWS * (4 + 2 + 2) + SUPPLIER_ROWS * (4 + 2 + 2) + DATE_ROWS * (4 + 2)))
      ;;
    q4.1)
      echo $((LINEORDER_ROWS * (4 + 4 + 4 + 4 + 4 + 4) + CUSTOMER_ROWS * (4 + 2 + 2) + SUPPLIER_ROWS * (4 + 2) + PART_ROWS * (4 + 2) + DATE_ROWS * (4 + 2)))
      ;;
  esac
}

query_internal_bytes() {
  local query="$1"
  local logical
  logical="$(query_logical_bytes "$query")"
  case "$query" in
    q1.1)
      echo $((logical + 4 * $(mask_bytes_for_rows "$LINEORDER_ROWS")))
      ;;
    q2.1)
      echo $((logical + 4 * $(mask_bytes_for_rows "$LINEORDER_ROWS") + 2 * $(mask_bytes_for_rows "$DATE_ROWS") + 3 * $(mask_bytes_for_rows "$PART_ROWS") + 2 * $(mask_bytes_for_rows "$SUPPLIER_ROWS")))
      ;;
    q3.1)
      echo $((logical + 4 * $(mask_bytes_for_rows "$LINEORDER_ROWS") + 3 * $(mask_bytes_for_rows "$CUSTOMER_ROWS") + 3 * $(mask_bytes_for_rows "$SUPPLIER_ROWS") + 2 * $(mask_bytes_for_rows "$DATE_ROWS")))
      ;;
    q4.1)
      echo $((logical + 6 * $(mask_bytes_for_rows "$LINEORDER_ROWS") + 3 * $(mask_bytes_for_rows "$CUSTOMER_ROWS") + 2 * $(mask_bytes_for_rows "$SUPPLIER_ROWS") + 2 * $(mask_bytes_for_rows "$PART_ROWS") + 2 * $(mask_bytes_for_rows "$DATE_ROWS")))
      ;;
  esac
}

query_cache_size() {
  local query="$1"
  if [ "$GPU_CACHE_SIZE" != "auto" ]; then
    echo "$GPU_CACHE_SIZE"
    return
  fi
  case "$query" in
    q3.1) echo "10 GB" ;;
    q1.1|q2.1|q4.1) echo "14 GB" ;;
  esac
}

query_processing_size() {
  local query="$1"
  if [ "$GPU_PROCESSING_SIZE" != "auto" ]; then
    echo "$GPU_PROCESSING_SIZE"
    return
  fi
  case "$query" in
    q1.1|q2.1|q3.1|q4.1) echo "18 GB" ;;
  esac
}

SUMMARY_CSV="$OUT_DIR/summary.csv"
write_summary_header() {
  printf 'query,iteration,mode,status,runtime_s,execute_ms,logical_base_table_bytes,logical_base_table_gib,sirius_internal_base_table_bytes,sirius_internal_base_table_gib,lineorder_logical_bytes,lineorder_logical_gib,lineorder_bytes_per_row,lineorder_columns,base_tables,base_columns,gpu_cache_size,gpu_processing_size,log_file,report_file\n' > "$SUMMARY_CSV"
}

append_summary_rows() {
  local query="$1"
  local log_file="$2"
  local report_file="${3:-}"
  local cache_size="$4"
  local processing_size="$5"
  local logical_bytes internal_bytes logical_gib internal_gib
  local fact_bytes fact_gib fact_bytes_per_row fact_columns
  logical_bytes="$(query_logical_bytes "$query")"
  internal_bytes="$(query_internal_bytes "$query")"
  logical_gib="$(bytes_to_gib "$logical_bytes")"
  internal_gib="$(bytes_to_gib "$internal_bytes")"
  fact_bytes="$(query_fact_logical_bytes "$query")"
  fact_gib="$(bytes_to_gib "$fact_bytes")"
  fact_bytes_per_row="$(query_fact_bytes_per_row "$query")"
  fact_columns="$(query_fact_columns "$query")"

  awk -v query="$query" \
      -v mode="$MODE" \
      -v logical_bytes="$logical_bytes" \
      -v logical_gib="$logical_gib" \
      -v internal_bytes="$internal_bytes" \
      -v internal_gib="$internal_gib" \
      -v fact_bytes="$fact_bytes" \
      -v fact_gib="$fact_gib" \
      -v fact_bytes_per_row="$fact_bytes_per_row" \
      -v fact_columns="$fact_columns" \
      -v base_tables="$(query_base_tables "$query")" \
      -v base_columns="$(query_base_columns "$query")" \
      -v cache_size="$cache_size" \
      -v processing_size="$processing_size" \
      -v log_file="$log_file" \
      -v report_file="$report_file" '
    function csv(s) { gsub(/"/, "\"\"", s); return "\"" s "\"" }
    $0 ~ "__SIRIUS_" query "_ITER_" && $0 ~ "_BEGIN__" {
      marker = $0
      sub(/^.*_ITER_/, "", marker)
      sub(/_BEGIN__.*$/, "", marker)
      current = marker + 0
      if (current > max_iter) max_iter = current
      next
    }
    $0 ~ "__SIRIUS_" query "_ITER_" && $0 ~ "_END__" {
      current = 0
      next
    }
    current > 0 && /fallback to DuckDB/ {
      fallback[current] = 1
    }
    current > 0 && /Catalog Error|Parser Error|Binder Error|Invalid Error|Segmentation fault|ERROR:|Error:/ {
      error[current] = 1
    }
    /Execute query time:/ {
      if (current > 0) {
        execute[current] = $(NF - 1)
        if (current > max_iter) max_iter = current
      } else {
        execute[++execute_count] = $(NF - 1)
      }
    }
    /Run Time \(s\): real/ {
      if (current > 0) {
        runtime[current] = $5
        if (current > max_iter) max_iter = current
      } else {
        runtime[++runtime_count] = $5
      }
    }
    END {
      max_count = max_iter
      if (runtime_count > max_count) max_count = runtime_count
      if (execute_count > max_count) max_count = execute_count
      if (max_count == 0) max_count = 1
      for (i = 1; i <= max_count; i++) {
        status = "ok"
        if (error[i]) status = "error"
        if (fallback[i]) status = "fallback"
        printf "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
          csv(query),
          i,
          csv(mode),
          csv(status),
          csv((runtime[i] == "" ? "NA" : runtime[i])),
          csv((execute[i] == "" ? "NA" : execute[i])),
          logical_bytes,
          logical_gib,
          internal_bytes,
          internal_gib,
          fact_bytes,
          fact_gib,
          fact_bytes_per_row,
          csv(fact_columns),
          csv(base_tables),
          csv(base_columns),
          csv(cache_size),
          csv(processing_size),
          csv(log_file),
          csv(report_file)
      }
    }
  ' "$log_file" >> "$SUMMARY_CSV"
}

write_sql_file() {
  local query="$1"
  local include_profiler="$2"
  local sql_file="$3"
  local iterations="$4"
  local cache_size="$5"
  local processing_size="$6"

  {
    echo "CALL gpu_buffer_init('${cache_size}', '${processing_size}', pinned_memory_size='${PINNED_MEMORY_SIZE}');"
    if [ "$include_profiler" = true ]; then
      echo "CALL profiler_start();"
    fi
    echo ".timer on"
    local i
    for i in $(seq 1 "$iterations"); do
      echo ".print __SIRIUS_${query}_ITER_${i}_BEGIN__"
      printf 'CALL gpu_processing("%s");\n' "$(query_sql "$query" | tr '\n' ' ' | sed 's/"/""/g')"
      echo ".print __SIRIUS_${query}_ITER_${i}_END__"
    done
    if [ "$include_profiler" = true ]; then
      echo "CALL profiler_stop();"
    fi
  } > "$sql_file"
}

write_profile_sql_file() {
  local query="$1"
  local sql_file="$2"
  local cache_size="$3"
  local processing_size="$4"

  {
    echo "CALL gpu_buffer_init('${cache_size}', '${processing_size}', pinned_memory_size='${PINNED_MEMORY_SIZE}');"
    echo ".timer on"
    echo ".print __SIRIUS_${query}_ITER_1_BEGIN__"
    printf 'CALL gpu_processing("%s");\n' "$(query_sql "$query" | tr '\n' ' ' | sed 's/"/""/g')"
    echo ".print __SIRIUS_${query}_ITER_1_END__"
    echo "CALL profiler_start();"
    echo ".print __SIRIUS_${query}_ITER_2_BEGIN__"
    printf 'CALL gpu_processing("%s");\n' "$(query_sql "$query" | tr '\n' ' ' | sed 's/"/""/g')"
    echo ".print __SIRIUS_${query}_ITER_2_END__"
    echo "CALL profiler_stop();"
  } > "$sql_file"
}

run_time_query() {
  local query="$1"
  local cache_size processing_size
  cache_size="$(query_cache_size "$query")"
  processing_size="$(query_processing_size "$query")"
  local query_dir="$OUT_DIR/$query"
  mkdir -p "$query_dir"
  local sql_file="$query_dir/query.sql"
  local log_file="$query_dir/time.log"

  write_sql_file "$query" false "$sql_file" "$ITERATIONS" "$cache_size" "$processing_size"
  "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" < "$sql_file" > "$log_file" 2>&1
  append_summary_rows "$query" "$log_file" "" "$cache_size" "$processing_size"
  echo "Wrote $log_file"
}

run_nsys_query() {
  local query="$1"
  local cache_size processing_size
  cache_size="$(query_cache_size "$query")"
  processing_size="$(query_processing_size "$query")"
  local query_dir="$OUT_DIR/$query"
  mkdir -p "$query_dir"
  local sql_file="$query_dir/query_nsys.sql"
  local log_file="$query_dir/nsys.log"
  local report_prefix="$query_dir/${query}"

  if [ ! -x "$NSYS" ]; then
    echo "ERROR: nsys not found or not executable: $NSYS"
    exit 1
  fi

  write_sql_file "$query" true "$sql_file" 1 "$cache_size" "$processing_size"
  "$NSYS" \
    profile \
    --trace=cuda,nvtx \
    --stats=false \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-device="$NSYS_GPU_METRICS_DEVICE" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=none \
    -f true \
    -o "$report_prefix" \
    "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" < "$sql_file" > "$log_file" 2>&1
  append_summary_rows "$query" "$log_file" "${report_prefix}.nsys-rep" "$cache_size" "$processing_size"
  echo "Wrote $log_file and ${report_prefix}.nsys-rep"
}

run_ncu_query() {
  local query="$1"
  local cache_size processing_size
  cache_size="$(query_cache_size "$query")"
  processing_size="$(query_processing_size "$query")"
  local query_dir="$OUT_DIR/$query"
  mkdir -p "$query_dir"
  local sql_file="$query_dir/query_ncu.sql"
  local log_file="$query_dir/ncu.log"
  local report_prefix="$query_dir/${query}"

  if [ ! -x "$NCU" ]; then
    echo "ERROR: ncu not found or not executable: $NCU"
    exit 1
  fi

  write_sql_file "$query" false "$sql_file" 1 "$cache_size" "$processing_size"
  "$NCU" \
    --metrics "$NCU_METRICS" \
    --target-processes all \
    -f \
    -o "$report_prefix" \
    "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" < "$sql_file" > "$log_file" 2>&1
  append_summary_rows "$query" "$log_file" "${report_prefix}.ncu-rep" "$cache_size" "$processing_size"
  echo "Wrote $log_file and ${report_prefix}.ncu-rep"
}

run_profile_query() {
  local query="$1"
  local cache_size processing_size
  cache_size="$(query_cache_size "$query")"
  processing_size="$(query_processing_size "$query")"
  local query_dir="$OUT_DIR/$query"
  mkdir -p "$query_dir"

  if [ ! -x "$NSYS" ]; then
    echo "ERROR: nsys not found or not executable: $NSYS"
    exit 1
  fi
  if [ ! -x "$NCU" ]; then
    echo "ERROR: ncu not found or not executable: $NCU"
    exit 1
  fi

  local nsys_sql_file="$query_dir/query_profile_nsys.sql"
  local nsys_log_file="$query_dir/profile_nsys.log"
  local nsys_report_prefix="$query_dir/${query}_profile_nsys"

  write_profile_sql_file "$query" "$nsys_sql_file" "$cache_size" "$processing_size"
  "$NSYS" \
    profile \
    --trace=cuda,nvtx \
    --stats=false \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-device="$NSYS_GPU_METRICS_DEVICE" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=none \
    -f true \
    -o "$nsys_report_prefix" \
    "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" < "$nsys_sql_file" > "$nsys_log_file" 2>&1
  append_summary_rows "$query" "$nsys_log_file" "${nsys_report_prefix}.nsys-rep" "$cache_size" "$processing_size"
  echo "Wrote $nsys_log_file and ${nsys_report_prefix}.nsys-rep"

  local ncu_sql_file="$query_dir/query_profile_ncu.sql"
  local ncu_log_file="$query_dir/profile_ncu.log"
  local ncu_report_prefix="$query_dir/${query}_profile_ncu"

  write_profile_sql_file "$query" "$ncu_sql_file" "$cache_size" "$processing_size"
  "$NCU" \
    --metrics "$NCU_METRICS" \
    --target-processes all \
    --profile-from-start 0 \
    -f \
    -o "$ncu_report_prefix" \
    "$DUCKDB_BIN" $DUCKDB_OPEN_FLAGS "$DB_PATH" < "$ncu_sql_file" > "$ncu_log_file" 2>&1
  append_summary_rows "$query" "$ncu_log_file" "${ncu_report_prefix}.ncu-rep" "$cache_size" "$processing_size"
  echo "Wrote $ncu_log_file and ${ncu_report_prefix}.ncu-rep"
}

{
  echo "mode=$MODE"
  echo "db=$DB_PATH"
  echo "duckdb=$DUCKDB_BIN"
  echo "duckdb_open_flags=$DUCKDB_OPEN_FLAGS"
  echo "out_dir=$OUT_DIR"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_cache_size=$GPU_CACHE_SIZE"
  echo "gpu_processing_size=$GPU_PROCESSING_SIZE"
  echo "pinned_memory_size=$PINNED_MEMORY_SIZE"
  echo "q2_1_category_MFGR12_id=$Q21_CATEGORY_ID"
  echo "q2_1_region_AMERICA_id=$Q21_REGION_ID"
  echo "q3_1_region_ASIA_id=$Q31_REGION_ID"
  echo "q4_1_region_AMERICA_id=$Q21_REGION_ID"
  echo "q4_1_mfgr1_id=$Q41_MFGR1_ID"
  echo "q4_1_mfgr2_id=$Q41_MFGR2_ID"
  echo "lineorder_rows=$LINEORDER_ROWS"
  echo "customer_rows=$CUSTOMER_ROWS"
  echo "date_rows=$DATE_ROWS"
  echo "part_rows=$PART_ROWS"
  echo "supplier_rows=$SUPPLIER_ROWS"
  echo "queries=${QUERIES[*]}"
} > "$OUT_DIR/run_config.txt"

write_summary_header

echo "Running SSB integer queries"
echo "  mode:    $MODE"
echo "  DB:      $DB_PATH"
echo "  DuckDB:  $DUCKDB_BIN"
echo "  out dir: $OUT_DIR"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"
echo "  Q2.1 IDs: p_category MFGR#12 -> $Q21_CATEGORY_ID, s_region AMERICA -> $Q21_REGION_ID"
echo "  Q3.1 IDs: c/s_region ASIA -> $Q31_REGION_ID"
echo "  Q4.1 IDs: c/s_region AMERICA -> $Q21_REGION_ID, p_mfgr MFGR#1/MFGR#2 -> $Q41_MFGR1_ID/$Q41_MFGR2_ID"

for query in "${QUERIES[@]}"; do
  echo "  $query pools: cache=$(query_cache_size "$query"), processing=$(query_processing_size "$query")"
  case "$MODE" in
	    time) run_time_query "$query" ;;
	    nsys) run_nsys_query "$query" ;;
	    ncu) run_ncu_query "$query" ;;
	    profile) run_profile_query "$query" ;;
	  esac
done

echo "Summary CSV: $SUMMARY_CSV"
echo "Done."
