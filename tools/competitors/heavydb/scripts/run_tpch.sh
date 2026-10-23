#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMING_CHECKER="$SCRIPT_DIR/../../check_timings.py"
ROOT_DIR=${HEAVYDB_ROOT:-${ROOT_DIR:-}}
WORK_ROOT=${WORK_ROOT:-${TMPDIR:-/tmp}/proteus-competitors/heavydb}
BUILD_DIR=${BUILD_DIR:-${ROOT_DIR:+$ROOT_DIR/build-release}}
# v2 dirs: the resubmission schema widens lineitem/orders and adds
# supplier/part/nation/region, so old v1 data dirs must not be reused.
DATA_DIR=${DATA_DIR:-$WORK_ROOT/tpch-data}
RAW_DIR=${RAW_DIR:-}
PREP_DIR=${PREP_DIR:-$WORK_ROOT/tpch-prepared}
DUCKDB_BIN=${DUCKDB_BIN:-$(command -v duckdb || true)}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/tpch-runs}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6274}
# HeavyDB fragment (rowgroup) size, rows per fragment. Default 32M (production).
# A value larger than every table's row count => one fragment per table (whole
# column GPU-resident, closest to Proteus's resident model). E.g. FRAG_SIZE=2000000000.
FRAG_SIZE=${FRAG_SIZE:-32000000}
PASSWORD=${PASSWORD:-HyperInteractive}
EXTRA_PREFIX=${EXTRA_PREFIX:-}
PIXI_ENV=${PIXI_ENV:-${ROOT_DIR:+$ROOT_DIR/.pixi/envs/default}}
MAVEN_TMP=${MAVEN_TMP:-$WORK_ROOT/maven-tmp}
LOAD_MODE=${LOAD_MODE:-if_missing}
PREPROCESS_MODE=${PREPROCESS_MODE:-if_missing}
WARM_RUNS=${WARM_RUNS:-10}
PROFILE_QUERY_MARKERS=${PROFILE_QUERY_MARKERS:-1}
QUERIES=${QUERIES:-"q5 q12"}

usage() {
  cat <<'USAGE'
Usage: run_tpch.sh --raw-dir DIR [options]

Runs the paper's HeavyDB TPC-H Q5 and Q12 on integer-encoded SF100 data.
The loader preprocesses raw TPC-H .tbl files into narrow integer TSVs:
scaled money, YYYYMMDD dates, and lexicographic dictionary IDs for strings.

Options:
  --load                Drop/recreate and reload HeavyDB TPC-H tables.
  --skip-load           Do not load; fail later if tables are absent.
  --load-if-missing     Load only if one or more tables are absent. Default.
  --preprocess          Rebuild integer TSVs and dictionary metadata with DuckDB.
  --skip-preprocess     Reuse existing integer TSVs and dictionary metadata.
  --preprocess-if-missing
                        Rebuild only if integer TSVs are absent. Default.
  --data-dir DIR        HeavyDB data directory. Default: WORK_ROOT/tpch-data
  --raw-dir DIR         Raw TPC-H .tbl directory. Required unless reusing data.
  --prep-dir DIR        Integer TSV/dictionary directory. Default: WORK_ROOT/tpch-prepared
  --duckdb-bin PATH     DuckDB binary for preprocessing. Default: duckdb from PATH
  --run-root DIR        Parent directory for run reports. Default: WORK_ROOT/tpch-runs
  --gpu ID              CUDA_VISIBLE_DEVICES value. Default: 0
  --warm-runs N         Extra repeated runs per query after the cold run. Default: 10
  --queries "q5 q12"    Space-separated subset of the paper queries. Default: both.
  -h, --help            Show this help.

Environment overrides are also supported: HEAVYDB_ROOT, WORK_ROOT, BUILD_DIR,
DATA_DIR, RAW_DIR,
PREP_DIR, DUCKDB_BIN, RUN_ROOT, GPU_ID, PORT, PASSWORD, EXTRA_PREFIX, PIXI_ENV,
MAVEN_TMP, LOAD_MODE, PREPROCESS_MODE, WARM_RUNS, PROFILE_QUERY_MARKERS, QUERIES.
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
    --raw-dir) RAW_DIR=$2; shift 2 ;;
    --prep-dir) PREP_DIR=$2; shift 2 ;;
    --duckdb-bin) DUCKDB_BIN=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --gpu) GPU_ID=$2; shift 2 ;;
    --warm-runs) WARM_RUNS=$2; shift 2 ;;
    --queries) QUERIES=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$WARM_RUNS" =~ ^[0-9]+$ ]] || { echo "--warm-runs must be a nonnegative integer." >&2; exit 2; }
[[ -n "$ROOT_DIR" ]] || { echo "Set HEAVYDB_ROOT to the HeavyDB checkout." >&2; exit 1; }
if [[ "$PREPROCESS_MODE" != "never" ]]; then
  [[ -n "$RAW_DIR" ]] || { echo "Pass --raw-dir or set RAW_DIR." >&2; exit 1; }
fi
for query in $QUERIES; do
  case "$query" in q5|q12) ;; *) echo "Only paper queries q5 and q12 are supported." >&2; exit 2 ;; esac
done

HEAVYDB_BIN="$BUILD_DIR/bin/heavydb"
HEAVYSQL_BIN="$BUILD_DIR/bin/heavysql"
INITHEAVY_BIN="$BUILD_DIR/bin/initheavy"
DICT_DB="$PREP_DIR/tpch_integer.duckdb"
LD_PATH="$EXTRA_PREFIX/lib:$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}"
RUN_DIR="$RUN_ROOT/$(date +%Y%m%d-%H%M%S)"
SERVER_STDOUT_LOG="$RUN_DIR/logs/heavydb.out"

mkdir -p "$RUN_DIR"/{sql,query_outputs,logs,stats} "$PREP_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$RUN_DIR/run.log"
}

require_file() {
  [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

for p in "$HEAVYDB_BIN" "$HEAVYSQL_BIN" "$INITHEAVY_BIN" "$DUCKDB_BIN" \
         "$TIMING_CHECKER"; do
  require_file "$p"
done
if [[ "$PREPROCESS_MODE" != "never" ]]; then
  for p in "$RAW_DIR/lineitem.tbl" "$RAW_DIR/orders.tbl" "$RAW_DIR/customer.tbl" \
           "$RAW_DIR/supplier.tbl" "$RAW_DIR/part.tbl" "$RAW_DIR/nation.tbl" \
           "$RAW_DIR/region.tbl"; do
    require_file "$p"
  done
fi

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

active_server_info_log() {
  local latest=""
  latest=$(ls -t "$DATA_DIR"/log/heavydb.INFO.*.log 2>/dev/null | head -1 || true)
  if [[ -n "$latest" ]]; then
    echo "$latest"
  else
    echo "$SERVER_STDOUT_LOG"
  fi
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
    > "$SERVER_STDOUT_LOG" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED=1

  for _ in $(seq 1 300); do
    if server_is_up; then
      log "HeavyDB is accepting SQL connections."
      return
    fi
    sleep 1
  done

  echo "HeavyDB did not become ready. See $SERVER_STDOUT_LOG" >&2
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
  cat > "$RUN_DIR/sql/preprocess_integer_tpch.sql" <<SQL
PRAGMA threads=16;

DROP VIEW IF EXISTS lineitem_src;
DROP VIEW IF EXISTS orders_src;
DROP VIEW IF EXISTS customer_src;
DROP VIEW IF EXISTS supplier_src;
DROP VIEW IF EXISTS part_src;
DROP VIEW IF EXISTS nation_src;
DROP VIEW IF EXISTS region_src;
DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS orders_tbl;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS nation_tbl;
DROP TABLE IF EXISTS region_tbl;
DROP TABLE IF EXISTS returnflag_dict;
DROP TABLE IF EXISTS linestatus_dict;
DROP TABLE IF EXISTS mktsegment_dict;
DROP TABLE IF EXISTS shipmode_dict;
DROP TABLE IF EXISTS shipinstruct_dict;
DROP TABLE IF EXISTS orderpriority_dict;
DROP TABLE IF EXISTS brand_dict;
DROP TABLE IF EXISTS container_dict;
DROP TABLE IF EXISTS regionname_dict;
DROP TABLE IF EXISTS query_constants;

CREATE OR REPLACE TEMP VIEW lineitem_src AS
SELECT * FROM read_csv('$RAW_DIR/lineitem.tbl', auto_detect=false, header=false, delim='|', columns={
  'l_orderkey':'INTEGER', 'l_partkey':'INTEGER', 'l_suppkey':'INTEGER',
  'l_linenumber':'INTEGER', 'l_quantity':'DECIMAL(15,2)',
  'l_extendedprice':'DECIMAL(15,2)', 'l_discount':'DECIMAL(15,2)',
  'l_tax':'DECIMAL(15,2)', 'l_returnflag':'VARCHAR', 'l_linestatus':'VARCHAR',
  'l_shipdate':'DATE', 'l_commitdate':'DATE', 'l_receiptdate':'DATE',
  'l_shipinstruct':'VARCHAR', 'l_shipmode':'VARCHAR', 'l_comment':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW orders_src AS
SELECT * FROM read_csv('$RAW_DIR/orders.tbl', auto_detect=false, header=false, delim='|', columns={
  'o_orderkey':'INTEGER', 'o_custkey':'INTEGER', 'o_orderstatus':'VARCHAR',
  'o_totalprice':'DECIMAL(15,2)', 'o_orderdate':'DATE', 'o_orderpriority':'VARCHAR',
  'o_clerk':'VARCHAR', 'o_shippriority':'INTEGER', 'o_comment':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW customer_src AS
SELECT * FROM read_csv('$RAW_DIR/customer.tbl', auto_detect=false, header=false, delim='|', columns={
  'c_custkey':'INTEGER', 'c_name':'VARCHAR', 'c_address':'VARCHAR',
  'c_nationkey':'INTEGER', 'c_phone':'VARCHAR', 'c_acctbal':'DECIMAL(15,2)',
  'c_mktsegment':'VARCHAR', 'c_comment':'VARCHAR'
});

CREATE TABLE returnflag_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_returnflag) - 1 AS TINYINT) AS returnflag_id,
       l_returnflag
FROM (SELECT DISTINCT l_returnflag FROM lineitem_src);

CREATE TABLE linestatus_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_linestatus) - 1 AS TINYINT) AS linestatus_id,
       l_linestatus
FROM (SELECT DISTINCT l_linestatus FROM lineitem_src);

CREATE TABLE mktsegment_dict AS
SELECT CAST(row_number() OVER (ORDER BY c_mktsegment) - 1 AS TINYINT) AS mktsegment_id,
       c_mktsegment
FROM (SELECT DISTINCT c_mktsegment FROM customer_src);

CREATE OR REPLACE TEMP VIEW supplier_src AS
SELECT * FROM read_csv('$RAW_DIR/supplier.tbl', auto_detect=false, header=false, delim='|', columns={
  's_suppkey':'INTEGER', 's_name':'VARCHAR', 's_address':'VARCHAR',
  's_nationkey':'INTEGER', 's_phone':'VARCHAR', 's_acctbal':'DECIMAL(15,2)',
  's_comment':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW part_src AS
SELECT * FROM read_csv('$RAW_DIR/part.tbl', auto_detect=false, header=false, delim='|', columns={
  'p_partkey':'INTEGER', 'p_name':'VARCHAR', 'p_mfgr':'VARCHAR',
  'p_brand':'VARCHAR', 'p_type':'VARCHAR', 'p_size':'INTEGER',
  'p_container':'VARCHAR', 'p_retailprice':'DECIMAL(15,2)', 'p_comment':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW nation_src AS
SELECT * FROM read_csv('$RAW_DIR/nation.tbl', auto_detect=false, header=false, delim='|', columns={
  'n_nationkey':'INTEGER', 'n_name':'VARCHAR', 'n_regionkey':'INTEGER',
  'n_comment':'VARCHAR'
});

CREATE OR REPLACE TEMP VIEW region_src AS
SELECT * FROM read_csv('$RAW_DIR/region.tbl', auto_detect=false, header=false, delim='|', columns={
  'r_regionkey':'INTEGER', 'r_name':'VARCHAR', 'r_comment':'VARCHAR'
});

CREATE TABLE shipmode_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_shipmode) - 1 AS TINYINT) AS shipmode_id,
       l_shipmode
FROM (SELECT DISTINCT l_shipmode FROM lineitem_src);

CREATE TABLE shipinstruct_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_shipinstruct) - 1 AS TINYINT) AS shipinstruct_id,
       l_shipinstruct
FROM (SELECT DISTINCT l_shipinstruct FROM lineitem_src);

CREATE TABLE orderpriority_dict AS
SELECT CAST(row_number() OVER (ORDER BY o_orderpriority) - 1 AS TINYINT) AS orderpriority_id,
       o_orderpriority
FROM (SELECT DISTINCT o_orderpriority FROM orders_src);

CREATE TABLE brand_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_brand) - 1 AS SMALLINT) AS brand_id,
       p_brand
FROM (SELECT DISTINCT p_brand FROM part_src);

CREATE TABLE container_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_container) - 1 AS TINYINT) AS container_id,
       p_container
FROM (SELECT DISTINCT p_container FROM part_src);

CREATE TABLE regionname_dict AS
SELECT CAST(row_number() OVER (ORDER BY r_name) - 1 AS TINYINT) AS regionname_id,
       r_name
FROM (SELECT DISTINCT r_name FROM region_src);

CREATE TABLE lineitem AS
SELECT l.l_orderkey,
       CAST(l.l_quantity AS TINYINT) AS l_quantity,
       CAST(l.l_extendedprice * 100 AS INTEGER) AS l_extendedprice,
       CAST(l.l_discount * 100 AS TINYINT) AS l_discount,
       CAST(l.l_tax * 100 AS TINYINT) AS l_tax,
       rf.returnflag_id AS l_returnflag,
       ls.linestatus_id AS l_linestatus,
       CAST(strftime(l.l_shipdate, '%Y%m%d') AS INTEGER) AS l_shipdate,
       l.l_partkey,
       l.l_suppkey,
       CAST(strftime(l.l_commitdate, '%Y%m%d') AS INTEGER) AS l_commitdate,
       CAST(strftime(l.l_receiptdate, '%Y%m%d') AS INTEGER) AS l_receiptdate,
       sm.shipmode_id AS l_shipmode,
       si.shipinstruct_id AS l_shipinstruct
FROM lineitem_src l
JOIN returnflag_dict rf ON l.l_returnflag = rf.l_returnflag
JOIN linestatus_dict ls ON l.l_linestatus = ls.l_linestatus
JOIN shipmode_dict sm ON l.l_shipmode = sm.l_shipmode
JOIN shipinstruct_dict si ON l.l_shipinstruct = si.l_shipinstruct;

CREATE TABLE orders_tbl AS
SELECT o.o_orderkey,
       o.o_custkey,
       CAST(strftime(o.o_orderdate, '%Y%m%d') AS INTEGER) AS o_orderdate,
       CAST(o.o_shippriority AS TINYINT) AS o_shippriority,
       op.orderpriority_id AS o_orderpriority
FROM orders_src o
JOIN orderpriority_dict op ON o.o_orderpriority = op.o_orderpriority;

CREATE TABLE customer AS
SELECT c.c_custkey,
       m.mktsegment_id AS c_mktsegment,
       CAST(c.c_nationkey AS TINYINT) AS c_nationkey
FROM customer_src c
JOIN mktsegment_dict m ON c.c_mktsegment = m.c_mktsegment;

CREATE TABLE supplier AS
SELECT s_suppkey,
       CAST(s_nationkey AS TINYINT) AS s_nationkey
FROM supplier_src;

CREATE TABLE part AS
SELECT p.p_partkey,
       b.brand_id AS p_brand,
       ct.container_id AS p_container,
       CAST(p.p_size AS TINYINT) AS p_size,
       CAST(CASE WHEN p.p_type LIKE 'PROMO%' THEN 1 ELSE 0 END AS TINYINT) AS p_promo
FROM part_src p
JOIN brand_dict b ON p.p_brand = b.p_brand
JOIN container_dict ct ON p.p_container = ct.p_container;

CREATE TABLE nation_tbl AS
SELECT CAST(n_nationkey AS TINYINT) AS n_nationkey,
       CAST(n_regionkey AS TINYINT) AS n_regionkey
FROM nation_src;

CREATE TABLE region_tbl AS
SELECT CAST(r.r_regionkey AS TINYINT) AS r_regionkey,
       rn.regionname_id AS r_name
FROM region_src r
JOIN regionname_dict rn ON r.r_name = rn.r_name;

CREATE TABLE query_constants AS
SELECT 'regionname_ASIA' AS name, CAST(regionname_id AS SMALLINT) AS value
FROM regionname_dict WHERE r_name = 'ASIA'
UNION ALL SELECT 'shipmode_MAIL', CAST(shipmode_id AS SMALLINT) FROM shipmode_dict WHERE l_shipmode = 'MAIL'
UNION ALL SELECT 'shipmode_SHIP', CAST(shipmode_id AS SMALLINT) FROM shipmode_dict WHERE l_shipmode = 'SHIP'
UNION ALL SELECT 'orderpriority_1_URGENT', CAST(orderpriority_id AS SMALLINT) FROM orderpriority_dict WHERE o_orderpriority = '1-URGENT'
UNION ALL SELECT 'orderpriority_2_HIGH', CAST(orderpriority_id AS SMALLINT) FROM orderpriority_dict WHERE o_orderpriority = '2-HIGH';

COPY lineitem TO '$PREP_DIR/lineitem.tsv' (DELIMITER '\t', HEADER false);
COPY orders_tbl TO '$PREP_DIR/orders.tsv' (DELIMITER '\t', HEADER false);
COPY customer TO '$PREP_DIR/customer.tsv' (DELIMITER '\t', HEADER false);
COPY supplier TO '$PREP_DIR/supplier.tsv' (DELIMITER '\t', HEADER false);
COPY part TO '$PREP_DIR/part.tsv' (DELIMITER '\t', HEADER false);
COPY nation_tbl TO '$PREP_DIR/nation.tsv' (DELIMITER '\t', HEADER false);
COPY region_tbl TO '$PREP_DIR/region.tsv' (DELIMITER '\t', HEADER false);
COPY returnflag_dict TO '$PREP_DIR/returnflag_dict.tsv' (DELIMITER '\t', HEADER true);
COPY linestatus_dict TO '$PREP_DIR/linestatus_dict.tsv' (DELIMITER '\t', HEADER true);
COPY mktsegment_dict TO '$PREP_DIR/mktsegment_dict.tsv' (DELIMITER '\t', HEADER true);
COPY shipmode_dict TO '$PREP_DIR/shipmode_dict.tsv' (DELIMITER '\t', HEADER true);
COPY shipinstruct_dict TO '$PREP_DIR/shipinstruct_dict.tsv' (DELIMITER '\t', HEADER true);
COPY orderpriority_dict TO '$PREP_DIR/orderpriority_dict.tsv' (DELIMITER '\t', HEADER true);
COPY brand_dict TO '$PREP_DIR/brand_dict.tsv' (DELIMITER '\t', HEADER true);
COPY container_dict TO '$PREP_DIR/container_dict.tsv' (DELIMITER '\t', HEADER true);
COPY regionname_dict TO '$PREP_DIR/regionname_dict.tsv' (DELIMITER '\t', HEADER true);
COPY query_constants TO '$PREP_DIR/query_constants.tsv' (DELIMITER '\t', HEADER false);
SQL
}

preprocessed_files_exist() {
  if [[ "$PREPROCESS_MODE" == "never" && "$LOAD_MODE" == "never" ]]; then
    [[ -s "$PREP_DIR/query_constants.tsv" ]] || return 1
    return 0
  fi
  for p in "$PREP_DIR/lineitem.tsv" "$PREP_DIR/orders.tsv" \
           "$PREP_DIR/customer.tsv" "$PREP_DIR/supplier.tsv" \
           "$PREP_DIR/part.tsv" "$PREP_DIR/nation.tsv" \
           "$PREP_DIR/region.tsv" "$PREP_DIR/query_constants.tsv"; do
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
    log "Preprocessing raw TPC-H into integer TSVs and ordered dictionaries with DuckDB."
    write_preprocess_sql
    "$DUCKDB_BIN" "$DICT_DB" < "$RUN_DIR/sql/preprocess_integer_tpch.sql" \
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
DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS orders_tbl;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS nation_tbl;
DROP TABLE IF EXISTS region_tbl;

CREATE TABLE lineitem (
  l_orderkey INT,
  l_quantity TINYINT,
  l_extendedprice INT,
  l_discount TINYINT,
  l_tax TINYINT,
  l_returnflag TINYINT,
  l_linestatus TINYINT,
  l_shipdate INT,
  l_partkey INT,
  l_suppkey INT,
  l_commitdate INT,
  l_receiptdate INT,
  l_shipmode TINYINT,
  l_shipinstruct TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE orders_tbl (
  o_orderkey INT,
  o_custkey INT,
  o_orderdate INT,
  o_shippriority TINYINT,
  o_orderpriority TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE customer (
  c_custkey INT,
  c_mktsegment TINYINT,
  c_nationkey TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE supplier (
  s_suppkey INT,
  s_nationkey TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE part (
  p_partkey INT,
  p_brand SMALLINT,
  p_container TINYINT,
  p_size TINYINT,
  p_promo TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE nation_tbl (
  n_nationkey TINYINT,
  n_regionkey TINYINT
) WITH (fragment_size=32000000);

CREATE TABLE region_tbl (
  r_regionkey TINYINT,
  r_name TINYINT
) WITH (fragment_size=32000000);
SQL
  # apply the configurable fragment (rowgroup) size to all tables
  sed -i "s/fragment_size=32000000/fragment_size=${FRAG_SIZE}/g" "$RUN_DIR/sql/schema.sql"
}

write_load_sql() {
  cat > "$RUN_DIR/sql/load.sql" <<SQL
COPY customer FROM '$PREP_DIR/customer.tsv' WITH (delimiter='\t', header='false');
COPY orders_tbl FROM '$PREP_DIR/orders.tsv' WITH (delimiter='\t', header='false');
COPY supplier FROM '$PREP_DIR/supplier.tsv' WITH (delimiter='\t', header='false');
COPY part FROM '$PREP_DIR/part.tsv' WITH (delimiter='\t', header='false');
COPY nation_tbl FROM '$PREP_DIR/nation.tsv' WITH (delimiter='\t', header='false');
COPY region_tbl FROM '$PREP_DIR/region.tsv' WITH (delimiter='\t', header='false');
COPY lineitem FROM '$PREP_DIR/lineitem.tsv' WITH (delimiter='\t', header='false');
SQL
}

# Dictionary/query constants resolved by the preprocess step; loaded into
# the QC associative array before write_query_sql runs.
declare -A QC
load_query_constants() {
  while IFS=$'\t' read -r name value; do
    [[ -n "$name" ]] && QC[$name]=$value
  done < "$PREP_DIR/query_constants.tsv"
  for key in regionname_ASIA shipmode_MAIL shipmode_SHIP \
             orderpriority_1_URGENT orderpriority_2_HIGH; do
    [[ -n "${QC[$key]:-}" ]] || {
      echo "Missing query constant '$key' in $PREP_DIR/query_constants.tsv (rerun with --preprocess)" >&2
      exit 1
    }
  done
}

# Query semantics mirror the paper's simplified Proteus Q5 and Q12: money is
# scaled integer, dates are YYYYMMDD integers, and strings are dictionary IDs.
write_query_sql() {
  cat > "$RUN_DIR/sql/q5.sql" <<SQL
SELECT n_nationkey,
       SUM(CAST(l_extendedprice AS BIGINT) * l_discount) AS revenue
FROM customer, orders_tbl, lineitem, supplier, nation_tbl, region_tbl
WHERE c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey
  AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey
  AND n_regionkey = r_regionkey
  AND r_name = ${QC[regionname_ASIA]}
  AND o_orderdate >= 19940101
  AND o_orderdate < 19950101
GROUP BY n_nationkey
ORDER BY revenue DESC;
SQL

  cat > "$RUN_DIR/sql/q12.sql" <<SQL
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = ${QC[orderpriority_1_URGENT]}
                  OR o_orderpriority = ${QC[orderpriority_2_HIGH]}
                THEN 1 ELSE 0 END) AS high_line_count,
       SUM(CASE WHEN o_orderpriority <> ${QC[orderpriority_1_URGENT]}
                 AND o_orderpriority <> ${QC[orderpriority_2_HIGH]}
                THEN 1 ELSE 0 END) AS low_line_count
FROM orders_tbl, lineitem
WHERE o_orderkey = l_orderkey
  AND (l_shipmode = ${QC[shipmode_MAIL]} OR l_shipmode = ${QC[shipmode_SHIP]})
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= 19940101
  AND l_receiptdate < 19950101
GROUP BY l_shipmode
ORDER BY l_shipmode;
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
  for t in lineitem orders_tbl customer supplier part nation_tbl region_tbl; do
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
SHOW TABLE DETAILS lineitem;
SHOW TABLE DETAILS orders_tbl;
SHOW TABLE DETAILS customer;
SHOW TABLE DETAILS supplier;
SHOW TABLE DETAILS part;
SHOW TABLE DETAILS nation_tbl;
SHOW TABLE DETAILS region_tbl;
SQL
}

write_query_timing_summary_tsv() {
  awk -v query_list="$QUERIES" '
    BEGIN {
      FS="\t"; OFS="\t"
      norder=split(query_list, order, " ")
      print "query","cold_ms","warm_runs","warm_min_ms","warm_avg_ms","warm_max_ms","all_runs_avg_ms"
    }
    NR == 1 { next }
    {
      q=$1; run=$2 + 0; ms=$3 + 0
      count[q]++; total[q] += ms
      if (run == 1) cold[q] = ms
      else {
        warm_count[q]++; warm_total[q] += ms
        if (!(q in warm_min) || ms < warm_min[q]) warm_min[q] = ms
        if (!(q in warm_max) || ms > warm_max[q]) warm_max[q] = ms
      }
    }
    END {
      for (i=1; i<=norder; i++) {
        q=order[i]
        if (!(q in count)) continue
        printf "%s\t%.0f\t%d\t%.0f\t%.2f\t%.0f\t%.2f\n",
               q, cold[q], warm_count[q], warm_min[q],
               warm_total[q] / warm_count[q], warm_max[q], total[q] / count[q]
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
      seen[key]=1; query[key]=q; marker_name[key]=marker
      total[key] += ms; total_count[key] += count; all_runs[key]++
      if (run == 1) {
        cold[key]=ms; cold_count[key]=count
      } else {
        warm_runs[key]++; warm_total[key] += ms; warm_count_sum[key] += count
        if (!(key in warm_min) || ms < warm_min[key]) warm_min[key]=ms
        if (!(key in warm_max) || ms > warm_max[key]) warm_max[key]=ms
      }
    }
    END {
      for (key in seen) {
        printf "%s\t%s\t%.6f\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\n",
               query[key], marker_name[key], cold[key], warm_runs[key],
               warm_min[key], warm_total[key] / warm_runs[key], warm_max[key],
               total[key] / all_runs[key], cold_count[key] + 0,
               warm_count_sum[key] + 0, total_count[key] + 0
      }
    }
  ' "$RUN_DIR/profile_timings.tsv" > "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv"
  {
    head -1 "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv"
    tail -n +2 "$RUN_DIR/stats/profile_timing_summary.unsorted.tsv" |
      sort -t "$(printf '\t')" -k1,1 -k2,2
  } > "$RUN_DIR/stats/profile_timing_summary.tsv"
}

table_row_width_bytes() {
  case "$1" in
    lineitem) echo 35 ;;   # 4+1+4+1+1+1+1+4 + 4+4+4+4+1+1
    orders_tbl) echo 14 ;; # 4+4+4+1+1
    customer) echo 6 ;;    # 4+1+1
    supplier) echo 5 ;;    # 4+1
    part) echo 9 ;;        # 4+2+1+1+1
    nation_tbl) echo 2 ;;
    region_tbl) echo 2 ;;
    *) echo 0 ;;
  esac
}

write_query_space_tsv() {
  local lineitem_rows orders_rows customer_rows supplier_rows nation_rows region_rows
  lineitem_rows=$(table_count lineitem || echo NA)
  orders_rows=$(table_count orders_tbl || echo NA)
  customer_rows=$(table_count customer || echo NA)
  supplier_rows=$(table_count supplier || echo NA)
  nation_rows=$(table_count nation_tbl || echo NA)
  region_rows=$(table_count region_tbl || echo NA)

  awk -v lineitem_rows="$lineitem_rows" -v orders_rows="$orders_rows" \
      -v customer_rows="$customer_rows" -v supplier_rows="$supplier_rows" \
      -v nation_rows="$nation_rows" \
      -v region_rows="$region_rows" '
    BEGIN {
      OFS="\t"
      print "query","formula_bytes","formula_mib","formula_gib","lineitem_formula_bytes","lineitem_formula_mib","lineitem_formula_gib","formula_detail"
      emit("q5", lineitem_rows * 13 + orders_rows * 12 + customer_rows * 5 + supplier_rows * 5 + nation_rows * 2 + region_rows * 2,
           lineitem_rows * 13,
           "lineitem_rows*(l_orderkey:4+l_suppkey:4+l_extendedprice:4+l_discount:1);orders_rows*(o_orderkey:4+o_custkey:4+o_orderdate:4);customer_rows*(c_custkey:4+c_nationkey:1);supplier_rows*(s_suppkey:4+s_nationkey:1);nation_rows*2;region_rows*2")
      emit("q12", lineitem_rows * 17 + orders_rows * 5,
           lineitem_rows * 17,
           "lineitem_rows*(l_orderkey:4+l_shipmode:1+l_commitdate:4+l_receiptdate:4+l_shipdate:4);orders_rows*(o_orderkey:4+o_orderpriority:1)")
    }
    function emit(query, bytes, lineitem_bytes, detail) {
      printf "%s\t%.0f\t%.2f\t%.4f\t%.0f\t%.2f\t%.4f\t%s\n",
             query, bytes, bytes / 1024 / 1024, bytes / 1024 / 1024 / 1024,
             lineitem_bytes, lineitem_bytes / 1024 / 1024,
             lineitem_bytes / 1024 / 1024 / 1024, detail
    }
  ' > "$RUN_DIR/stats/query_space_summary.tsv"
}

write_table_space_tsv() {
  local out="$RUN_DIR/stats/table_space_summary.tsv"
  {
    printf 'table_name\trow_count\trow_width_bytes\tformula_bytes\tformula_mib\tformula_gib\n'
    for table in lineitem orders_tbl customer supplier part nation_tbl region_tbl; do
      rows=$(table_count "$table" || echo NA)
      width=$(table_row_width_bytes "$table")
      awk -v table="$table" -v rows="$rows" -v width="$width" 'BEGIN {
        bytes = rows * width
        printf "%s\t%s\t%d\t%.0f\t%.2f\t%.4f\n",
               table, rows, width, bytes, bytes / 1024 / 1024, bytes / 1024 / 1024 / 1024
      }'
    done
  } > "$out"
}

write_benchmark_summary_csv() {
  write_query_timing_summary_tsv
  write_profile_timing_summary_tsv
  write_table_space_tsv
  write_query_space_tsv

  local out="$RUN_DIR/stats/benchmark_summary.csv"
  {
    printf 'record_type,name,query,run,execution_ms,cold_ms,warm_runs,warm_min_ms,warm_avg_ms,warm_max_ms,all_runs_avg_ms,table_name,row_count,row_width_bytes,formula_bytes,formula_mib,formula_gib,lineitem_formula_bytes,lineitem_formula_mib,lineitem_formula_gib,disk_bytes,path,value,unit,source,formula_detail,marker_count,cold_marker_count,warm_marker_count_sum,all_marker_count_sum\n'
    awk -v run_dir="$RUN_DIR" -v data_dir="$DATA_DIR" -v prep_dir="$PREP_DIR" \
        -v raw_dir="$RAW_DIR" -v gpu_id="$GPU_ID" -v warm_runs="$WARM_RUNS" '
      BEGIN {
        OFS=","
        emit_metric("run_dir", run_dir, "path")
        emit_metric("data_dir", data_dir, "path")
        emit_metric("prep_dir", prep_dir, "path")
        emit_metric("raw_dir", raw_dir, "path")
        emit_metric("gpu_id", gpu_id, "id")
        emit_metric("warm_runs", warm_runs, "count")
      }
      function emit(  i) { for (i=1; i<=30; i++) printf "%s%s", f[i], (i == 30 ? ORS : OFS); delete f }
      function emit_metric(name, value, unit) { f[1]="run_metric"; f[2]=name; f[23]=value; f[24]=unit; f[25]="run_info"; emit() }
    '
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="query_run"; f[2]="execution_ms"; f[3]=$1; f[4]=$2; f[5]=$3; f[24]="ms"; f[25]="timings.tsv"; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/timings.tsv"
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="query_run_profile"; f[2]=$3; f[3]=$1; f[4]=$2; f[5]=$5; f[24]="ms"; f[25]=$6; f[27]=$4; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/profile_timings.tsv"
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="query_summary"; f[2]="timing_summary"; f[3]=$1; f[6]=$2; f[7]=$3; f[8]=$4; f[9]=$5; f[10]=$6; f[11]=$7; f[24]="ms"; f[25]="query_timing_summary.tsv"; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/stats/query_timing_summary.tsv"
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="query_profile_summary"; f[2]=$2; f[3]=$1; f[6]=$3; f[7]=$4; f[8]=$5; f[9]=$6; f[10]=$7; f[11]=$8; f[24]="ms"; f[25]="profile_timing_summary.tsv"; f[28]=$9; f[29]=$10; f[30]=$11; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/stats/profile_timing_summary.tsv"
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="query_space"; f[2]="logical_formula"; f[3]=$1; f[15]=$2; f[16]=$3; f[17]=$4; f[18]=$5; f[19]=$6; f[20]=$7; f[24]="bytes"; f[25]="query_space_summary.tsv"; f[26]=$8; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/stats/query_space_summary.tsv"
    awk -F"\t" 'BEGIN{OFS=","} NR>1 {f[1]="table_space"; f[2]="space_summary"; f[12]=$1; f[13]=$2; f[14]=$3; f[15]=$4; f[16]=$5; f[17]=$6; f[25]="table_space_summary.tsv"; emit()}
      function emit(  i){for(i=1;i<=30;i++) printf "%s%s", f[i], (i==30?ORS:OFS); delete f}' "$RUN_DIR/stats/table_space_summary.tsv"
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
  echo "raw_dir=$RAW_DIR"
  echo "prep_dir=$PREP_DIR"
  echo "duckdb_bin=$DUCKDB_BIN"
  echo "dict_db=$DICT_DB"
  echo "gpu_id=$GPU_ID"
  echo "load_mode=$LOAD_MODE"
  echo "preprocess_mode=$PREPROCESS_MODE"
  echo "warm_runs=$WARM_RUNS"
  echo "queries=$QUERIES"
  echo "profile_query_markers=$PROFILE_QUERY_MARKERS"
} > "$RUN_DIR/run_info.txt"

run_preprocess_if_needed
load_query_constants
write_query_sql

start_server

log "Capturing baseline GPU/memory state."
capture_stats before

if need_load; then
  log "Loading integer TPC-H schema/data into HeavyDB."
  run_sql_file "$RUN_DIR/sql/schema.sql" "$RUN_DIR/query_outputs/schema.out"
  run_sql_file "$RUN_DIR/sql/load.sql" "$RUN_DIR/query_outputs/load.out"
else
  log "Skipping HeavyDB load because TPC-H tables already exist or LOAD_MODE=never."
fi

log "Capturing table storage details after load."
run_sql_file "$RUN_DIR/sql/table_details.sql" "$RUN_DIR/stats/table_details.txt" || true
du -sb "$DATA_DIR/data" "$DATA_DIR" "$PREP_DIR" > "$RUN_DIR/stats/du_data.txt" 2>&1 || true
capture_stats after_load

printf 'query\trun\texecution_ms\toutput_file\n' > "$RUN_DIR/timings.tsv"
printf 'query\trun\tmarker\tmarker_count\telapsed_ms_sum\tsource_file\n' > "$RUN_DIR/profile_timings.tsv"
total_runs=$((1 + WARM_RUNS))
for q in $QUERIES; do
  for run in $(seq 1 "$total_runs"); do
    out="$RUN_DIR/query_outputs/${q}_run${run}.out"
    log "Running $q run $run/$total_runs."
    profile_start_line=$(server_log_line_count)
    query_rc=0
    run_sql_file "$RUN_DIR/sql/${q}.sql" "$out" || query_rc=$?
    profile_end_line=$(server_log_line_count)
    extract_profile_marker_summary "$q" "$run" "$profile_start_line" "$profile_end_line"
    if (( query_rc == 0 )); then
      ms=$(extract_timing_ms "$out" || true)
    else
      log "ERROR: $q run $run failed with heavysql status $query_rc; see $out"
      ms=""
    fi
    printf '%s\t%s\t%s\t%s\n' "$q" "$run" "${ms:-NA}" "$out" >> "$RUN_DIR/timings.tsv"
  done
done

python3 "$TIMING_CHECKER" heavydb \
  --file "$RUN_DIR/timings.tsv" \
  --profile-file "$RUN_DIR/profile_timings.tsv" \
  --queries "$QUERIES" \
  --iterations "$total_runs"

log "Capturing final storage and memory stats."
run_sql_file "$RUN_DIR/sql/table_details.sql" "$RUN_DIR/stats/table_details_after_queries.txt" || true
capture_stats after_queries
write_benchmark_summary_csv

log "Run complete."
cat <<EOF

HeavyDB integer TPC-H run complete.
Run directory: $RUN_DIR

Important files:
  $RUN_DIR/timings.tsv
  $RUN_DIR/profile_timings.tsv
  $RUN_DIR/stats/query_timing_summary.tsv
  $RUN_DIR/stats/profile_timing_summary.tsv
  $RUN_DIR/stats/benchmark_summary.csv
  $RUN_DIR/query_outputs/

Dictionary artifacts:
  $DICT_DB
  $PREP_DIR/*_dict.tsv
  $PREP_DIR/query_constants.tsv

EOF
