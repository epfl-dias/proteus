#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${HEAVYDB_ROOT:-${ROOT_DIR:-}}
WORK_ROOT=${WORK_ROOT:-${TMPDIR:-/tmp}/proteus-competitors/heavydb}
BUILD_DIR=${BUILD_DIR:-${ROOT_DIR:+$ROOT_DIR/build-release}}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/ssb-data}
SSB_DIR=${SSB_DIR:-}
PREP_DIR=${PREP_DIR:-$WORK_ROOT/ssb-prepared}
DUCKDB_BIN=${DUCKDB_BIN:-$(command -v duckdb || true)}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/ssb-runs}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6274}
# HeavyDB fragment (rowgroup) size, rows per fragment. Default 32M
# (production). Set this above every table's row count, e.g.
# FRAG_SIZE=2000000000, to force one fragment per table.
FRAG_SIZE=${FRAG_SIZE:-32000000}
PASSWORD=${PASSWORD:-HyperInteractive}
EXTRA_PREFIX=${EXTRA_PREFIX:-}
PIXI_ENV=${PIXI_ENV:-${ROOT_DIR:+$ROOT_DIR/.pixi/envs/default}}
MAVEN_TMP=${MAVEN_TMP:-$WORK_ROOT/maven-tmp}
LOAD_MODE=${LOAD_MODE:-if_missing}
PREPROCESS_MODE=${PREPROCESS_MODE:-if_missing}
WARM_RUNS=${WARM_RUNS:-10}
PROFILE_QUERY_MARKERS=${PROFILE_QUERY_MARKERS:-1}
QUERIES=()

usage() {
  cat <<'USAGE'
Usage: run_ssb.sh --ssb-dir DIR [options] [q1.1 q2.1 q3.1 q4.1]

Runs HeavyDB SSB Q1.1, Q2.1, Q3.1, and Q4.1 on externally
dictionary-encoded integer tables. A DuckDB preprocessing step creates
deterministic ordered dictionaries, rewrites SSB string columns to SMALLINT IDs,
and writes TSV files that HeavyDB imports.

Options:
  --load                Drop/recreate and reload HeavyDB SSB tables.
  --skip-load           Do not load; fail later if tables are absent.
  --load-if-missing     Load only if one or more HeavyDB SSB tables are absent. Default.
  --preprocess          Rebuild integer TSVs and dictionary metadata with DuckDB.
  --skip-preprocess     Reuse existing integer TSVs and dictionary metadata.
  --preprocess-if-missing
                        Rebuild only if integer TSVs are absent. Default.
  --data-dir DIR        HeavyDB data directory. Default: WORK_ROOT/ssb-data
  --ssb-dir DIR         Raw SSB .tbl directory. Required.
  --prep-dir DIR        Integer TSV/dictionary directory. Default: WORK_ROOT/ssb-prepared
  --duckdb-bin PATH     DuckDB binary for preprocessing. Default: duckdb from PATH
  --run-root DIR        Parent directory for run reports. Default: WORK_ROOT/ssb-runs
  --gpu ID              CUDA_VISIBLE_DEVICES value. Default: 0
  --warm-runs N         Extra repeated runs per query after the first run. Default: 10
  -h, --help            Show this help.

Environment overrides are also supported: HEAVYDB_ROOT, WORK_ROOT, BUILD_DIR,
DATA_DIR, SSB_DIR, PREP_DIR,
DUCKDB_BIN, RUN_ROOT, GPU_ID, PORT, FRAG_SIZE, PASSWORD, EXTRA_PREFIX, PIXI_ENV,
MAVEN_TMP, LOAD_MODE, PREPROCESS_MODE, WARM_RUNS, PROFILE_QUERY_MARKERS.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --load) LOAD_MODE=always; shift ;;
    --skip-load) LOAD_MODE=never; shift ;;
    --load-if-missing) LOAD_MODE=if_missing; shift ;;
    --preprocess) PREPROCESS_MODE=always; shift ;;
    --skip-preprocess) PREPROCESS_MODE=never; shift ;;
    --preprocess-if-missing) PREPROCESS_MODE=if_missing; shift ;;
    --data-dir) DATA_DIR=$2; shift 2 ;;
    --ssb-dir) SSB_DIR=$2; shift 2 ;;
    --prep-dir) PREP_DIR=$2; shift 2 ;;
    --duckdb-bin) DUCKDB_BIN=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --gpu) GPU_ID=$2; shift 2 ;;
    --warm-runs) WARM_RUNS=$2; shift 2 ;;
    q1.1|Q1.1|1|1.1) QUERIES+=(q1_1); shift ;;
    q2.1|Q2.1|2|2.1) QUERIES+=(q2_1); shift ;;
    q3.1|Q3.1|3|3.1) QUERIES+=(q3_1); shift ;;
    q4.1|Q4.1|4|4.1) QUERIES+=(q4_1); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${#QUERIES[@]} -eq 0 ]]; then
  QUERIES=(q1_1 q2_1 q3_1 q4_1)
fi

[[ -n "$ROOT_DIR" ]] || { echo "Set HEAVYDB_ROOT to the HeavyDB checkout." >&2; exit 1; }
[[ -n "$SSB_DIR" ]] || { echo "Pass --ssb-dir or set SSB_DIR." >&2; exit 1; }

HEAVYDB_BIN="$BUILD_DIR/bin/heavydb"
HEAVYSQL_BIN="$BUILD_DIR/bin/heavysql"
INITHEAVY_BIN="$BUILD_DIR/bin/initheavy"
DICT_DB="$PREP_DIR/ssb_integer.duckdb"
LD_PATH="$EXTRA_PREFIX/lib:$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}"
RUN_DIR="$RUN_ROOT/$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$RUN_DIR/logs/heavydb.out"

mkdir -p "$RUN_DIR"/{sql,query_outputs,logs,stats} "$PREP_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$RUN_DIR/run.log"
}

require_file() {
  [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

for p in "$HEAVYDB_BIN" "$HEAVYSQL_BIN" "$INITHEAVY_BIN" "$DUCKDB_BIN" \
         "$SSB_DIR/lineorder.tbl" "$SSB_DIR/customer.tbl" "$SSB_DIR/part.tbl" \
         "$SSB_DIR/supplier.tbl" "$SSB_DIR/date.tbl"; do
  require_file "$p"
done

run_sql_file() {
  local sql_file=$1
  local out_file=$2
  /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
    -p "$PASSWORD" --port "$PORT" -t < "$sql_file" > "$out_file" 2>&1
}

run_heavysql_commands() {
  local out_file=$1
  shift
  printf '%s\n' "$@" | /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
    -p "$PASSWORD" --port "$PORT" > "$out_file" 2>&1
}

server_is_up() {
  local tmp="$RUN_DIR/logs/connection_check.txt"
  printf 'SELECT 1;\n' | /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
    -p "$PASSWORD" --port "$PORT" > "$tmp" 2>&1
}

start_server() {
  if server_is_up; then
    log "HeavyDB already running on port $PORT; using existing server."
    SERVER_STARTED=0
    return
  fi

  mkdir -p "$DATA_DIR" "$MAVEN_TMP"
  if [[ ! -d "$DATA_DIR/catalogs" ]]; then
    log "Initializing HeavyDB data directory: $DATA_DIR"
    /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$INITHEAVY_BIN" \
      --data "$DATA_DIR" --skip-geo > "$RUN_DIR/logs/initheavy.out" 2>&1
  fi

  log "Starting HeavyDB on GPU $GPU_ID."
  /usr/bin/env TMPDIR="${TMPDIR:-/tmp}" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HEAVYDB_BENCH_PROFILE_QUERY="$PROFILE_QUERY_MARKERS" \
    MAVEN_OPTS="-Djava.io.tmpdir=$MAVEN_TMP" \
    LD_LIBRARY_PATH="$LD_PATH" \
    "$HEAVYDB_BIN" "$DATA_DIR" \
    "--port=$PORT" \
    "--allowed-import-paths=[\"$PREP_DIR\"]" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED=1

  for _ in $(seq 1 300); do
    if server_is_up; then
      log "HeavyDB is accepting SQL connections."
      return
    fi
    sleep 1
  done

  echo "HeavyDB did not become ready. See $RUN_DIR/logs/heavydb.out" >&2
  exit 1
}

stop_server() {
  if [[ ${SERVER_STARTED:-0} -eq 1 ]] && [[ -n ${SERVER_PID:-} ]]; then
    log "Stopping HeavyDB server PID $SERVER_PID."
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

write_preprocess_sql() {
  cat > "$RUN_DIR/sql/preprocess_integer_ssb.sql" <<SQL
PRAGMA threads=16;

DROP VIEW IF EXISTS lineorder_src;
DROP VIEW IF EXISTS customer_src;
DROP VIEW IF EXISTS supplier_src;
DROP VIEW IF EXISTS part_src;
DROP VIEW IF EXISTS date_src;

DROP TABLE IF EXISTS lineorder;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS date_dim;
DROP TABLE IF EXISTS region_dict;
DROP TABLE IF EXISTS nation_dict;
DROP TABLE IF EXISTS city_dict;
DROP TABLE IF EXISTS mfgr_dict;
DROP TABLE IF EXISTS category_dict;
DROP TABLE IF EXISTS brand_dict;
DROP TABLE IF EXISTS query_constants;

CREATE OR REPLACE TEMP VIEW lineorder_src AS
SELECT * FROM read_csv('$SSB_DIR/lineorder.tbl', auto_detect=false, header=false, delim='|', columns={
  'lo_orderkey':'INTEGER', 'lo_linenumber':'SMALLINT', 'lo_custkey':'INTEGER',
  'lo_partkey':'INTEGER', 'lo_suppkey':'INTEGER', 'lo_orderdate':'INTEGER',
  'lo_orderpriority':'VARCHAR', 'lo_shippriority':'VARCHAR', 'lo_quantity':'TINYINT',
  'lo_extendedprice':'INTEGER', 'lo_ordtotalprice':'INTEGER', 'lo_discount':'TINYINT',
  'lo_revenue':'INTEGER', 'lo_supplycost':'INTEGER', 'lo_tax':'SMALLINT',
  'lo_commitdate':'INTEGER', 'lo_shipmode':'VARCHAR', '__end':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW customer_src AS
SELECT * FROM read_csv('$SSB_DIR/customer.tbl', auto_detect=false, header=false, delim='|', columns={
  'c_custkey':'INTEGER', 'c_name':'VARCHAR', 'c_address':'VARCHAR', 'c_city':'VARCHAR',
  'c_nation':'VARCHAR', 'c_region':'VARCHAR', 'c_phone':'VARCHAR', 'c_mktsegment':'VARCHAR',
  '__end':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW supplier_src AS
SELECT * FROM read_csv('$SSB_DIR/supplier.tbl', auto_detect=false, header=false, delim='|', columns={
  's_suppkey':'INTEGER', 's_name':'VARCHAR', 's_address':'VARCHAR', 's_city':'VARCHAR',
  's_nation':'VARCHAR', 's_region':'VARCHAR', 's_phone':'VARCHAR', '__end':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW part_src AS
SELECT * FROM read_csv('$SSB_DIR/part.tbl', auto_detect=false, header=false, delim='|', columns={
  'p_partkey':'INTEGER', 'p_name':'VARCHAR', 'p_mfgr':'VARCHAR', 'p_category':'VARCHAR',
  'p_brand':'VARCHAR', 'p_color':'VARCHAR', 'p_type':'VARCHAR', 'p_size':'SMALLINT',
  'p_container':'VARCHAR', '__end':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW date_src AS
SELECT * FROM read_csv('$SSB_DIR/date.tbl', auto_detect=false, header=false, delim='|', columns={
  'd_datekey':'INTEGER', 'd_date':'VARCHAR', 'd_dayofweek':'VARCHAR', 'd_month':'VARCHAR',
  'd_year':'SMALLINT', 'd_yearmonthnum':'INTEGER', 'd_yearmonth':'VARCHAR',
  'd_daynuminweek':'SMALLINT', 'd_daynuminmonth':'SMALLINT', 'd_daynuminyear':'SMALLINT',
  'd_monthnuminyear':'SMALLINT', 'd_weeknuminyear':'TINYINT', 'd_sellingseason':'VARCHAR',
  'd_lastdayinweekfl':'VARCHAR', 'd_lastdayinmonthfl':'VARCHAR', 'd_holidayfl':'VARCHAR',
  'd_weekdayfl':'VARCHAR', '__end':'VARCHAR'
});

CREATE TABLE region_dict AS
SELECT CAST(row_number() OVER (ORDER BY region) AS TINYINT) AS region_id, region
FROM (SELECT DISTINCT c_region AS region FROM customer_src UNION SELECT DISTINCT s_region AS region FROM supplier_src);

CREATE TABLE nation_dict AS
SELECT CAST(row_number() OVER (ORDER BY nation) AS TINYINT) AS nation_id, nation
FROM (SELECT DISTINCT c_nation AS nation FROM customer_src UNION SELECT DISTINCT s_nation AS nation FROM supplier_src);

CREATE TABLE city_dict AS
SELECT CAST(row_number() OVER (ORDER BY city) AS SMALLINT) AS city_id, city
FROM (SELECT DISTINCT c_city AS city FROM customer_src UNION SELECT DISTINCT s_city AS city FROM supplier_src);

CREATE TABLE mfgr_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_mfgr) AS TINYINT) AS mfgr_id, p_mfgr
FROM (SELECT DISTINCT p_mfgr FROM part_src);

CREATE TABLE category_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_category) AS TINYINT) AS category_id, p_category
FROM (SELECT DISTINCT p_category FROM part_src);

CREATE TABLE brand_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_brand) AS SMALLINT) AS brand_id, p_brand
FROM (SELECT DISTINCT p_brand FROM part_src);

CREATE TABLE lineorder AS
SELECT lo_orderdate, lo_quantity, lo_extendedprice, lo_discount, lo_partkey,
       lo_suppkey, lo_revenue, lo_custkey, lo_supplycost
FROM lineorder_src;

CREATE TABLE customer AS
SELECT c.c_custkey, r.region_id AS c_region_id, n.nation_id AS c_nation_id,
       ci.city_id AS c_city_id
FROM customer_src c
JOIN region_dict r ON c.c_region = r.region
JOIN nation_dict n ON c.c_nation = n.nation
JOIN city_dict ci ON c.c_city = ci.city;

CREATE TABLE supplier AS
SELECT s.s_suppkey, r.region_id AS s_region_id, n.nation_id AS s_nation_id,
       ci.city_id AS s_city_id
FROM supplier_src s
JOIN region_dict r ON s.s_region = r.region
JOIN nation_dict n ON s.s_nation = n.nation
JOIN city_dict ci ON s.s_city = ci.city;

CREATE TABLE part AS
SELECT p.p_partkey, m.mfgr_id AS p_mfgr_id, c.category_id AS p_category_id,
       b.brand_id AS p_brand_id
FROM part_src p
JOIN mfgr_dict m ON p.p_mfgr = m.p_mfgr
JOIN category_dict c ON p.p_category = c.p_category
JOIN brand_dict b ON p.p_brand = b.p_brand;

CREATE TABLE date_dim AS
SELECT d_datekey, d_year, d_yearmonthnum, d_weeknuminyear
FROM date_src;

CREATE TABLE query_constants AS
SELECT 'region_AMERICA' AS name, CAST(region_id AS TINYINT) AS value FROM region_dict WHERE region = 'AMERICA'
UNION ALL SELECT 'region_ASIA', CAST(region_id AS TINYINT) FROM region_dict WHERE region = 'ASIA'
UNION ALL SELECT 'category_MFGR_12', CAST(category_id AS TINYINT) FROM category_dict WHERE p_category = 'MFGR#12'
UNION ALL SELECT 'mfgr_MFGR_1', CAST(mfgr_id AS TINYINT) FROM mfgr_dict WHERE p_mfgr = 'MFGR#1'
UNION ALL SELECT 'mfgr_MFGR_2', CAST(mfgr_id AS TINYINT) FROM mfgr_dict WHERE p_mfgr = 'MFGR#2';

COPY lineorder TO '$PREP_DIR/lineorder.tsv' (DELIMITER '\t', HEADER false);
COPY customer TO '$PREP_DIR/customer.tsv' (DELIMITER '\t', HEADER false);
COPY supplier TO '$PREP_DIR/supplier.tsv' (DELIMITER '\t', HEADER false);
COPY part TO '$PREP_DIR/part.tsv' (DELIMITER '\t', HEADER false);
COPY date_dim TO '$PREP_DIR/date_dim.tsv' (DELIMITER '\t', HEADER false);
COPY region_dict TO '$PREP_DIR/region_dict.tsv' (DELIMITER '\t', HEADER true);
COPY nation_dict TO '$PREP_DIR/nation_dict.tsv' (DELIMITER '\t', HEADER true);
COPY city_dict TO '$PREP_DIR/city_dict.tsv' (DELIMITER '\t', HEADER true);
COPY mfgr_dict TO '$PREP_DIR/mfgr_dict.tsv' (DELIMITER '\t', HEADER true);
COPY category_dict TO '$PREP_DIR/category_dict.tsv' (DELIMITER '\t', HEADER true);
COPY brand_dict TO '$PREP_DIR/brand_dict.tsv' (DELIMITER '\t', HEADER true);
COPY query_constants TO '$PREP_DIR/query_constants.tsv' (DELIMITER '\t', HEADER false);
SQL
}

preprocessed_files_exist() {
  for p in "$PREP_DIR/lineorder.tsv" "$PREP_DIR/customer.tsv" "$PREP_DIR/supplier.tsv" \
           "$PREP_DIR/part.tsv" "$PREP_DIR/date_dim.tsv" "$PREP_DIR/query_constants.tsv"; do
    [[ -s "$p" ]] || return 1
  done
}

need_preprocess() {
  [[ "$PREPROCESS_MODE" == "always" ]] && return 0
  [[ "$PREPROCESS_MODE" == "never" ]] && return 1
  preprocessed_files_exist || return 0
  return 1
}

run_preprocess_if_needed() {
  if need_preprocess; then
    log "Preprocessing raw SSB into integer TSVs and ordered dictionaries with DuckDB."
    write_preprocess_sql
    "$DUCKDB_BIN" "$DICT_DB" < "$RUN_DIR/sql/preprocess_integer_ssb.sql" \
      > "$RUN_DIR/logs/preprocess.out" 2>&1
  else
    log "Skipping preprocessing; integer TSVs and query constants already exist."
  fi
  preprocessed_files_exist || {
    echo "Preprocessed integer files are missing under $PREP_DIR" >&2
    exit 1
  }
}

constant_id() {
  local name=$1
  awk -F '\t' -v name="$name" '$1 == name {print $2; found=1} END {if (!found) exit 1}' \
    "$PREP_DIR/query_constants.tsv"
}

write_schema_sql() {
  cat > "$RUN_DIR/sql/schema.sql" <<'SQL'
DROP TABLE IF EXISTS lineorder;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS date_dim;

CREATE TABLE lineorder (
  lo_orderdate INT,
  lo_quantity TINYINT,
  lo_extendedprice INT,
  lo_discount TINYINT,
  lo_partkey INT,
  lo_suppkey INT,
  lo_revenue INT,
  lo_custkey INT,
  lo_supplycost INT
) WITH (fragment_size=32000000);

CREATE TABLE customer (
  c_custkey INT,
  c_region_id TINYINT,
  c_nation_id TINYINT,
  c_city_id SMALLINT
) WITH (fragment_size=32000000);

CREATE TABLE supplier (
  s_suppkey INT,
  s_region_id TINYINT,
  s_nation_id TINYINT,
  s_city_id SMALLINT
) WITH (fragment_size=32000000);

CREATE TABLE part (
  p_partkey INT,
  p_mfgr_id TINYINT,
  p_category_id TINYINT,
  p_brand_id SMALLINT
) WITH (fragment_size=32000000);

CREATE TABLE date_dim (
  d_datekey INT,
  d_year SMALLINT,
  d_yearmonthnum INT,
  d_weeknuminyear TINYINT
) WITH (fragment_size=32000000);
SQL
  sed -i "s/fragment_size=32000000/fragment_size=${FRAG_SIZE}/g" "$RUN_DIR/sql/schema.sql"
}

write_load_sql() {
  cat > "$RUN_DIR/sql/load.sql" <<SQL
COPY customer FROM '$PREP_DIR/customer.tsv' WITH (delimiter='\t', header='false');
COPY part FROM '$PREP_DIR/part.tsv' WITH (delimiter='\t', header='false');
COPY supplier FROM '$PREP_DIR/supplier.tsv' WITH (delimiter='\t', header='false');
COPY date_dim FROM '$PREP_DIR/date_dim.tsv' WITH (delimiter='\t', header='false');
COPY lineorder FROM '$PREP_DIR/lineorder.tsv' WITH (delimiter='\t', header='false');
SQL
}

write_query_sql() {
  local region_america=$1
  local region_asia=$2
  local category_mfgr_12=$3
  local mfgr_1=$4
  local mfgr_2=$5

  cat > "$RUN_DIR/sql/q1_1.sql" <<'SQL'
SELECT sum(lo_extendedprice * lo_discount) AS revenue
FROM lineorder
WHERE lo_orderdate > 19930000
  AND lo_orderdate < 19940000
  AND lo_discount BETWEEN 1 AND 3
  AND lo_quantity < 25;
SQL

  cat > "$RUN_DIR/sql/q2_1.sql" <<SQL
SELECT sum(lo_revenue) AS revenue, d_year, p_brand_id
FROM lineorder, date_dim, part, supplier
WHERE lo_orderdate = d_datekey
  AND lo_partkey = p_partkey
  AND lo_suppkey = s_suppkey
  AND p_category_id = $category_mfgr_12
  AND s_region_id = $region_america
GROUP BY d_year, p_brand_id
ORDER BY d_year, p_brand_id;
SQL

  cat > "$RUN_DIR/sql/q3_1.sql" <<SQL
SELECT c_nation_id, s_nation_id, d_year, sum(lo_revenue) AS revenue
FROM customer, lineorder, supplier, date_dim
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_orderdate = d_datekey
  AND c_region_id = $region_asia
  AND s_region_id = $region_asia
  AND d_year >= 1992 AND d_year <= 1997
GROUP BY c_nation_id, s_nation_id, d_year
ORDER BY d_year ASC, revenue DESC;
SQL

  cat > "$RUN_DIR/sql/q4_1.sql" <<SQL
SELECT d_year, c_nation_id, sum(lo_revenue - lo_supplycost) AS profit
FROM date_dim, customer, supplier, part, lineorder
WHERE lo_custkey = c_custkey
  AND lo_suppkey = s_suppkey
  AND lo_partkey = p_partkey
  AND lo_orderdate = d_datekey
  AND c_region_id = $region_america
  AND s_region_id = $region_america
  AND (p_mfgr_id = $mfgr_1 OR p_mfgr_id = $mfgr_2)
GROUP BY d_year, c_nation_id
ORDER BY d_year, c_nation_id;
SQL
}

table_count() {
  local table=$1
  local tmp="$RUN_DIR/logs/count_${table}.txt"
  printf 'SELECT COUNT(*) AS n FROM %s;\n' "$table" | \
    /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
      -p "$PASSWORD" --port "$PORT" > "$tmp" 2>&1 || return 1
  awk '/^[0-9]+$/ {print; found=1} END {if (!found) exit 1}' "$tmp"
}

need_load() {
  [[ "$LOAD_MODE" == "always" ]] && return 0
  [[ "$LOAD_MODE" == "never" ]] && return 1
  for t in lineorder customer part supplier date_dim; do
    if ! table_count "$t" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

extract_timing_ms() {
  awk -F'Execution time: | ms' '/Execution time:/ {print $2; exit}' "$1"
}

server_log_line_count() {
  local log_file
  log_file=$(active_server_info_log)
  if [[ -f "$log_file" ]]; then
    wc -l < "$log_file"
  else
    echo 0
  fi
}

active_server_info_log() {
  local latest=""
  latest=$(ls -t "$DATA_DIR"/log/heavydb.INFO.*.log 2>/dev/null | head -1 || true)
  if [[ -n "$latest" ]]; then
    echo "$latest"
  else
    echo "$SERVER_LOG"
  fi
}

extract_profile_marker_summary() {
  local query=$1
  local run=$2
  local start_line=$3
  local end_line=$4
  local slice_file="$RUN_DIR/logs/profile_${query}_run${run}.log"
  local log_file
  log_file=$(active_server_info_log)

  if [[ ! -f "$log_file" ]] || [[ "$end_line" -le "$start_line" ]]; then
    : > "$slice_file"
  else
    sed -n "$((start_line + 1)),${end_line}p" "$log_file" > "$slice_file"
  fi

  awk -v query="$query" -v run="$run" -v source="$slice_file" '
    BEGIN {
      OFS="\t"
      marker[1]="HEAVYDB_BENCH_QUERY_KERNEL_MS"
      marker[2]="HEAVYDB_BENCH_GPU_EXEC_ENVELOPE_MS"
      marker[3]="HEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS"
      marker[4]="HEAVYDB_BENCH_THRIFT_RESULT_CONVERT_MS"
    }
    {
      for (i=1; i<=4; i++) {
        m=marker[i]
        if (index($0, m)) {
          n=split($0, parts, "elapsed_ms=")
          if (n > 1) {
            value=parts[2]
            sub(/[[:space:]].*/, "", value)
            sum[m] += value + 0
            count[m]++
          }
        }
      }
    }
    END {
      for (i=1; i<=4; i++) {
        m=marker[i]
        printf "%s\t%s\t%s\t%d\t%.6f\t%s\n", query, run, m, count[m] + 0, sum[m] + 0, source
      }
    }
  ' "$slice_file" >> "$RUN_DIR/profile_timings.tsv"
}

capture_stats() {
  local label=$1
  nvidia-smi -i "$GPU_ID" > "$RUN_DIR/stats/nvidia_${label}.txt" 2>&1 || true
  run_heavysql_commands "$RUN_DIR/stats/memory_${label}.txt" \
    '\memory_summary' '\memory_gpu' '\memory_cpu' || true
}

write_table_details_sql() {
  cat > "$RUN_DIR/sql/table_details.sql" <<'SQL'
SHOW TABLE DETAILS lineorder;
SHOW TABLE DETAILS customer;
SHOW TABLE DETAILS part;
SHOW TABLE DETAILS supplier;
SHOW TABLE DETAILS date_dim;
SQL
}

parse_gpu_by_table() {
  local memory_file=$1
  local details_file=$2
  local out_file=$3
  awk -F'|' '
    /^[0-9]+\|/ { id=$1; name=$2; table[id]=name }
    END { for (id in table) print id "\t" table[id] }
  ' "$details_file" > "$RUN_DIR/stats/table_id_map.tsv" || true

  awk '
    BEGIN { page=512 }
    / USED[[:space:]]+[0-9]+,[0-9]+,[0-9]+,[0-9]+,/ {
      num_page=$3
      key=$6
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      split(key, k, ",")
      table_id=k[2]
      bytes[table_id] += num_page * page
    }
    END {
      print "table_id\tgpu_bytes\tgpu_mib"
      for (id in bytes) printf "%s\t%.0f\t%.2f\n", id, bytes[id], bytes[id]/1024/1024
    }
  ' "$memory_file" > "$RUN_DIR/stats/gpu_memory_by_table.raw.tsv"

  awk 'BEGIN{FS=OFS="\t"} NR==FNR {name[$1]=$2; next}
       FNR==1 {print "table_id","table_name","gpu_bytes","gpu_mib"; next}
       {print $1,(name[$1] ? name[$1] : "unknown"),$2,$3}' \
    "$RUN_DIR/stats/table_id_map.tsv" \
    "$RUN_DIR/stats/gpu_memory_by_table.raw.tsv" > "$out_file"
}

table_row_width_bytes() {
  case "$1" in
    lineorder) echo 30 ;;
    customer) echo 8 ;;
    supplier) echo 8 ;;
    part) echo 8 ;;
    date_dim) echo 11 ;;
    *) echo 0 ;;
  esac
}

write_table_space_tsv() {
  local out="$RUN_DIR/stats/table_space_summary.tsv"
  {
    printf 'table_name\trow_count\trow_width_bytes\tformula_bytes\tformula_mib\tformula_gib\theavydb_data_file_bytes\theavydb_data_file_mib\tgpu_bytes\tgpu_mib\n'
    for table in lineorder customer supplier part date_dim; do
      rows=$(table_count "$table" || echo NA)
      width=$(table_row_width_bytes "$table")
      awk -v table="$table" -v rows="$rows" -v width="$width" '
        BEGIN { FS=OFS="\t"; gpu_bytes="NA"; gpu_mib="NA" }
        NR == FNR {
          if (FNR > 1 && $2 == table) {
            gpu_bytes=$3
            gpu_mib=$4
          }
          next
        }
        /^[0-9]+\|/ {
          split($0, d, "|")
          if (d[2] == table) {
            heavydb_data_file_bytes=d[18]
          }
        }
        END {
          if (heavydb_data_file_bytes == "") heavydb_data_file_bytes="NA"
          if (rows == "NA") {
            formula_bytes="NA"; formula_mib="NA"; formula_gib="NA"
          } else {
            formula_bytes=rows * width
            formula_mib=formula_bytes / 1024 / 1024
            formula_gib=formula_bytes / 1024 / 1024 / 1024
          }
          if (heavydb_data_file_bytes == "NA") {
            heavydb_data_file_mib="NA"
          } else {
            heavydb_data_file_mib=heavydb_data_file_bytes / 1024 / 1024
          }
          printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                 table, rows, width,
                 formula_bytes == "NA" ? "NA" : sprintf("%.0f", formula_bytes),
                 formula_mib == "NA" ? "NA" : sprintf("%.2f", formula_mib),
                 formula_gib == "NA" ? "NA" : sprintf("%.4f", formula_gib),
                 heavydb_data_file_bytes,
                 heavydb_data_file_mib == "NA" ? "NA" : sprintf("%.2f", heavydb_data_file_mib),
                 gpu_bytes, gpu_mib
        }
      ' "$RUN_DIR/stats/gpu_memory_by_table.tsv" "$RUN_DIR/stats/table_details_after_queries.txt"
    done
  } > "$out"
}

write_query_timing_summary_tsv() {
  awk '
    BEGIN {
      FS="\t"; OFS="\t"
      order[1]="q1_1"; order[2]="q2_1"; order[3]="q3_1"; order[4]="q4_1"
      print "query","cold_ms","warm_runs","warm_min_ms","warm_avg_ms","warm_max_ms","all_runs_avg_ms"
    }
    NR == 1 { next }
    {
      q=$1
      run=$2 + 0
      ms=$3 + 0
      count[q]++
      total[q] += ms
      if (run == 1) {
        cold[q] = ms
      } else {
        warm_count[q]++
        warm_total[q] += ms
        if (!(q in warm_min) || ms < warm_min[q]) warm_min[q] = ms
        if (!(q in warm_max) || ms > warm_max[q]) warm_max[q] = ms
      }
    }
    END {
      for (i=1; i<=4; i++) {
        q=order[i]
        if (!(q in count)) continue
        if (warm_count[q] > 0) {
          printf "%s\t%.0f\t%d\t%.0f\t%.2f\t%.0f\t%.2f\n",
                 q, cold[q], warm_count[q], warm_min[q], warm_total[q] / warm_count[q],
                 warm_max[q], total[q] / count[q]
        } else {
          printf "%s\t%.0f\t0\tNA\tNA\tNA\t%.2f\n", q, cold[q], total[q] / count[q]
        }
      }
    }
  ' "$RUN_DIR/timings.tsv" > "$RUN_DIR/stats/query_timing_summary.tsv"
}

write_profile_timing_summary_tsv() {
  awk -F"\t" '
    BEGIN {
      OFS="\t"
      print "query","marker","cold_ms","warm_runs","warm_min_ms","warm_avg_ms","warm_max_ms","all_runs_avg_ms","cold_count","warm_count_sum","all_count_sum"
    }
    NR == 1 { next }
    {
      q=$1; run=$2 + 0; marker=$3; count=$4 + 0; ms=$5 + 0
      key=q SUBSEP marker
      seen[key]=1
      query[key]=q
      marker_name[key]=marker
      total[key] += ms
      total_count[key] += count
      all_runs[key]++
      if (run == 1) {
        cold[key]=ms
        cold_count[key]=count
      } else {
        warm_runs[key]++
        warm_total[key] += ms
        warm_count_sum[key] += count
        if (!(key in warm_min) || ms < warm_min[key]) warm_min[key]=ms
        if (!(key in warm_max) || ms > warm_max[key]) warm_max[key]=ms
      }
    }
    END {
      for (key in seen) {
        if (warm_runs[key] > 0) {
          printf "%s\t%s\t%.6f\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\n",
                 query[key], marker_name[key], cold[key], warm_runs[key],
                 warm_min[key], warm_total[key] / warm_runs[key], warm_max[key],
                 total[key] / all_runs[key], cold_count[key] + 0,
                 warm_count_sum[key] + 0, total_count[key] + 0
        } else {
          printf "%s\t%s\t%.6f\t0\tNA\tNA\tNA\t%.6f\t%d\t0\t%d\n",
                 query[key], marker_name[key], cold[key], total[key] / all_runs[key],
                 cold_count[key] + 0, total_count[key] + 0
        }
      }
    }
  ' "$RUN_DIR/profile_timings.tsv" > "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv"
  {
    head -1 "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv"
    tail -n +2 "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv" |
      sort -t "$(printf '\t')" -k1,1 -k2,2
  } > "$RUN_DIR/stats/profile_timing_summary.tsv"
}

write_query_space_tsv() {
  local lineorder_rows customer_rows supplier_rows part_rows date_rows
  lineorder_rows=$(table_count lineorder || echo NA)
  customer_rows=$(table_count customer || echo NA)
  supplier_rows=$(table_count supplier || echo NA)
  part_rows=$(table_count part || echo NA)
  date_rows=$(table_count date_dim || echo NA)

  awk -v lineorder_rows="$lineorder_rows" -v customer_rows="$customer_rows" \
      -v supplier_rows="$supplier_rows" -v part_rows="$part_rows" -v date_rows="$date_rows" '
    BEGIN {
      OFS="\t"
      print "query","formula_bytes","formula_mib","formula_gib","lineorder_formula_bytes","lineorder_formula_mib","lineorder_formula_gib","formula_detail"
      emit("q1_1",
           lineorder_rows * 10,
           lineorder_rows * 10,
           "lineorder_rows*(lo_extendedprice:4+lo_discount:1+lo_quantity:1+lo_orderdate:4)")
      emit("q2_1",
           lineorder_rows * 16 + date_rows * 6 + part_rows * 7 + supplier_rows * 5,
           lineorder_rows * 16,
           "lineorder_rows*(lo_revenue:4+lo_orderdate:4+lo_partkey:4+lo_suppkey:4);date_dim_rows*(d_datekey:4+d_year:2);part_rows*(p_partkey:4+p_category_id:1+p_brand_id:2);supplier_rows*(s_suppkey:4+s_region_id:1)")
      emit("q3_1",
           customer_rows * 6 + lineorder_rows * 16 + supplier_rows * 6 + date_rows * 6,
           lineorder_rows * 16,
           "customer_rows*(c_custkey:4+c_region_id:1+c_nation_id:1);lineorder_rows*(lo_custkey:4+lo_suppkey:4+lo_orderdate:4+lo_revenue:4);supplier_rows*(s_suppkey:4+s_region_id:1+s_nation_id:1);date_dim_rows*(d_datekey:4+d_year:2)")
      emit("q4_1",
           date_rows * 6 + customer_rows * 6 + supplier_rows * 5 + part_rows * 5 + lineorder_rows * 24,
           lineorder_rows * 24,
           "date_dim_rows*(d_datekey:4+d_year:2);customer_rows*(c_custkey:4+c_region_id:1+c_nation_id:1);supplier_rows*(s_suppkey:4+s_region_id:1);part_rows*(p_partkey:4+p_mfgr_id:1);lineorder_rows*(lo_custkey:4+lo_suppkey:4+lo_partkey:4+lo_orderdate:4+lo_revenue:4+lo_supplycost:4)")
    }
    function emit(query, bytes, lineorder_bytes, detail) {
      printf "%s\t%.0f\t%.2f\t%.4f\t%.0f\t%.2f\t%.4f\t%s\n",
             query,
             bytes, bytes / 1024 / 1024, bytes / 1024 / 1024 / 1024,
             lineorder_bytes, lineorder_bytes / 1024 / 1024, lineorder_bytes / 1024 / 1024 / 1024,
             detail
    }
  ' > "$RUN_DIR/stats/query_space_summary.tsv"

  awk -v lineorder_rows="$lineorder_rows" -v customer_rows="$customer_rows" \
      -v supplier_rows="$supplier_rows" -v part_rows="$part_rows" -v date_rows="$date_rows" '
    BEGIN {
      OFS="\t"
      print "query","table_name","row_count","query_table_width_bytes","formula_bytes","formula_mib","formula_gib","formula_detail"
      emit("q1_1", "lineorder", lineorder_rows, 10, "lo_extendedprice:4+lo_discount:1+lo_quantity:1+lo_orderdate:4")
      emit("q2_1", "lineorder", lineorder_rows, 16, "lo_revenue:4+lo_orderdate:4+lo_partkey:4+lo_suppkey:4")
      emit("q2_1", "date_dim", date_rows, 6, "d_datekey:4+d_year:2")
      emit("q2_1", "part", part_rows, 7, "p_partkey:4+p_category_id:1+p_brand_id:2")
      emit("q2_1", "supplier", supplier_rows, 5, "s_suppkey:4+s_region_id:1")
      emit("q3_1", "customer", customer_rows, 6, "c_custkey:4+c_region_id:1+c_nation_id:1")
      emit("q3_1", "lineorder", lineorder_rows, 16, "lo_custkey:4+lo_suppkey:4+lo_orderdate:4+lo_revenue:4")
      emit("q3_1", "supplier", supplier_rows, 6, "s_suppkey:4+s_region_id:1+s_nation_id:1")
      emit("q3_1", "date_dim", date_rows, 6, "d_datekey:4+d_year:2")
      emit("q4_1", "date_dim", date_rows, 6, "d_datekey:4+d_year:2")
      emit("q4_1", "customer", customer_rows, 6, "c_custkey:4+c_region_id:1+c_nation_id:1")
      emit("q4_1", "supplier", supplier_rows, 5, "s_suppkey:4+s_region_id:1")
      emit("q4_1", "part", part_rows, 5, "p_partkey:4+p_mfgr_id:1")
      emit("q4_1", "lineorder", lineorder_rows, 24, "lo_custkey:4+lo_suppkey:4+lo_partkey:4+lo_orderdate:4+lo_revenue:4+lo_supplycost:4")
    }
    function emit(query, table_name, rows, width, detail, bytes) {
      bytes = rows * width
      printf "%s\t%s\t%s\t%d\t%.0f\t%.2f\t%.4f\t%s\n",
             query, table_name, rows, width, bytes, bytes / 1024 / 1024, bytes / 1024 / 1024 / 1024, detail
    }
  ' > "$RUN_DIR/stats/query_space_components.tsv"
}

write_benchmark_summary_csv() {
  write_query_timing_summary_tsv
  write_profile_timing_summary_tsv
  write_table_space_tsv
  write_query_space_tsv

  local out="$RUN_DIR/stats/benchmark_summary.csv"
  {
    printf 'record_type,name,query,run,execution_ms,cold_ms,warm_runs,warm_min_ms,warm_avg_ms,warm_max_ms,all_runs_avg_ms,table_name,row_count,row_width_bytes,formula_bytes,formula_mib,formula_gib,lineorder_formula_bytes,lineorder_formula_mib,lineorder_formula_gib,heavydb_data_file_bytes,heavydb_data_file_mib,gpu_bytes,gpu_mib,disk_bytes,path,value,unit,source,formula_detail,marker_count,cold_marker_count,warm_marker_count_sum,all_marker_count_sum\n'
    awk -v run_dir="$RUN_DIR" -v data_dir="$DATA_DIR" -v prep_dir="$PREP_DIR" \
        -v gpu_id="$GPU_ID" -v warm_runs="$WARM_RUNS" '
      BEGIN {
        OFS=","
        emit_metric("run_dir", run_dir, "path")
        emit_metric("data_dir", data_dir, "path")
        emit_metric("prep_dir", prep_dir, "path")
        emit_metric("gpu_id", gpu_id, "id")
        emit_metric("warm_runs", warm_runs, "count")
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
      function emit_metric(name, value, unit) {
        f[1]="run_metric"; f[2]=name; f[27]=value; f[28]=unit; f[29]="run_info"; emit()
      }
    '
    awk '
      BEGIN { OFS="," }
      {
        f[1]="disk_usage"; f[2]="du_bytes"; f[25]=$1; f[26]=$2; f[27]=$1; f[28]="bytes"; f[29]="du_data.txt"; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/du_data.txt"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_run"; f[2]="execution_ms"; f[3]=$1; f[4]=$2; f[5]=$3; f[28]="ms"; f[29]="timings.tsv"; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/timings.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_run_profile"; f[2]=$3; f[3]=$1; f[4]=$2; f[5]=$5; f[28]="ms"; f[29]=$6; f[31]=$4; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/profile_timings.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_summary"; f[2]="timing_summary"; f[3]=$1; f[6]=$2; f[7]=$3
        f[8]=$4; f[9]=$5; f[10]=$6; f[11]=$7; f[28]="ms"; f[29]="query_timing_summary.tsv"; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/query_timing_summary.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_profile_summary"; f[2]=$2; f[3]=$1; f[6]=$3; f[7]=$4
        f[8]=$5; f[9]=$6; f[10]=$7; f[11]=$8; f[28]="ms"
        f[29]="profile_timing_summary.tsv"; f[32]=$9; f[33]=$10; f[34]=$11; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/profile_timing_summary.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_space"; f[2]="logical_formula"; f[3]=$1; f[15]=$2; f[16]=$3; f[17]=$4
        f[18]=$5; f[19]=$6; f[20]=$7; f[28]="bytes"; f[29]="query_space_summary.tsv"; f[30]=$8; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/query_space_summary.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="query_space_table"; f[2]="logical_formula_component"; f[3]=$1; f[12]=$2; f[13]=$3
        f[14]=$4; f[15]=$5; f[16]=$6; f[17]=$7; f[28]="bytes"; f[29]="query_space_components.tsv"; f[30]=$8; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/query_space_components.tsv"
    awk -F"\t" '
      BEGIN { OFS="," }
      NR > 1 {
        f[1]="table_space"; f[2]="space_summary"; f[12]=$1; f[13]=$2; f[14]=$3; f[15]=$4
        f[16]=$5; f[17]=$6; f[21]=$7; f[22]=$8; f[23]=$9; f[24]=$10
        f[29]="table_space_summary.tsv"; emit()
      }
      function emit(  i) {
        for (i=1; i<=34; i++) printf "%s%s", f[i], (i == 34 ? ORS : OFS)
        delete f
      }
    ' "$RUN_DIR/stats/table_space_summary.tsv"
  } > "$out"
}

write_schema_sql
write_load_sql
write_table_details_sql

{
  echo "run_dir=$RUN_DIR"
  echo "date=$(date -Is)"
  echo "root_dir=$ROOT_DIR"
  echo "build_dir=$BUILD_DIR"
  echo "data_dir=$DATA_DIR"
  echo "ssb_dir=$SSB_DIR"
  echo "prep_dir=$PREP_DIR"
  echo "duckdb_bin=$DUCKDB_BIN"
  echo "dict_db=$DICT_DB"
  echo "gpu_id=$GPU_ID"
  echo "load_mode=$LOAD_MODE"
  echo "preprocess_mode=$PREPROCESS_MODE"
  echo "warm_runs=$WARM_RUNS"
  echo "profile_query_markers=$PROFILE_QUERY_MARKERS"
  echo "queries=${QUERIES[*]}"
} > "$RUN_DIR/run_info.txt"

run_preprocess_if_needed

REGION_AMERICA_ID=$(constant_id region_AMERICA)
REGION_ASIA_ID=$(constant_id region_ASIA)
CATEGORY_MFGR_12_ID=$(constant_id category_MFGR_12)
MFGR_1_ID=$(constant_id mfgr_MFGR_1)
MFGR_2_ID=$(constant_id mfgr_MFGR_2)
write_query_sql "$REGION_AMERICA_ID" "$REGION_ASIA_ID" "$CATEGORY_MFGR_12_ID" "$MFGR_1_ID" "$MFGR_2_ID"

start_server

log "Capturing baseline GPU/memory state."
capture_stats before

if need_load; then
  log "Loading integer SSB schema/data into HeavyDB."
  run_sql_file "$RUN_DIR/sql/schema.sql" "$RUN_DIR/query_outputs/schema.out"
  run_sql_file "$RUN_DIR/sql/load.sql" "$RUN_DIR/query_outputs/load.out"
else
  log "Skipping HeavyDB load because SSB tables already exist or LOAD_MODE=never."
fi

log "Capturing table storage details after load."
run_sql_file "$RUN_DIR/sql/table_details.sql" "$RUN_DIR/stats/table_details.txt" || true
du -sb "$DATA_DIR/data" "$DATA_DIR" "$PREP_DIR" > "$RUN_DIR/stats/du_data.txt" 2>&1 || true
capture_stats after_load

printf 'query\trun\texecution_ms\toutput_file\n' > "$RUN_DIR/timings.tsv"
printf 'query\trun\tmarker\tmarker_count\telapsed_ms_sum\tsource_file\n' > "$RUN_DIR/profile_timings.tsv"
for q in "${QUERIES[@]}"; do
  total_runs=$((1 + WARM_RUNS))
  for run in $(seq 1 "$total_runs"); do
    out="$RUN_DIR/query_outputs/${q}_run${run}.out"
    log "Running $q run $run/$total_runs."
    profile_start_line=$(server_log_line_count)
    run_sql_file "$RUN_DIR/sql/${q}.sql" "$out"
    profile_end_line=$(server_log_line_count)
    extract_profile_marker_summary "$q" "$run" "$profile_start_line" "$profile_end_line"
    ms=$(extract_timing_ms "$out" || true)
    printf '%s\t%s\t%s\t%s\n' "$q" "$run" "${ms:-NA}" "$out" >> "$RUN_DIR/timings.tsv"
  done
done

log "Capturing final storage and memory stats."
run_sql_file "$RUN_DIR/sql/table_details.sql" "$RUN_DIR/stats/table_details_after_queries.txt" || true
capture_stats after_queries
parse_gpu_by_table "$RUN_DIR/stats/memory_after_queries.txt" \
  "$RUN_DIR/stats/table_details_after_queries.txt" \
  "$RUN_DIR/stats/gpu_memory_by_table.tsv" || true
write_benchmark_summary_csv

log "Run complete."
cat <<EOF

HeavyDB integer SSB run complete.
Run directory: $RUN_DIR

Important files:
  $RUN_DIR/timings.tsv
  $RUN_DIR/stats/table_details_after_queries.txt
  $RUN_DIR/stats/memory_after_queries.txt
  $RUN_DIR/stats/gpu_memory_by_table.tsv
  $RUN_DIR/stats/benchmark_summary.csv
  $RUN_DIR/stats/nvidia_after_queries.txt
  $RUN_DIR/stats/du_data.txt
  $RUN_DIR/query_outputs/

Dictionary artifacts:
  $DICT_DB
  $PREP_DIR/*_dict.tsv
  $PREP_DIR/query_constants.tsv

EOF
