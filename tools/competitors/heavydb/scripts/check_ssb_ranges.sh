#!/usr/bin/env bash
set -euo pipefail

SSB_DIR=${1:-}
DUCKDB_BIN=${2:-$(command -v duckdb || true)}
OUT_CSV=${3:-${TMPDIR:-/tmp}/heavydb-ssb-range-check.csv}

usage() {
  cat <<'USAGE'
Usage: check_ssb_ranges.sh <ssb_tbl_dir> [duckdb_bin] [out_csv]

Scans the raw SSB .tbl files and writes a CSV with min/max values, the smallest
safe HeavyDB integer storage type, and the type selected by the benchmark
runner.

The default output is TMPDIR/heavydb-ssb-range-check.csv.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for required in lineorder.tbl customer.tbl supplier.tbl part.tbl date.tbl; do
  if [[ ! -f "$SSB_DIR/$required" ]]; then
    echo "Missing input file: $SSB_DIR/$required" >&2
    exit 1
  fi
done

if [[ ! -x "$DUCKDB_BIN" ]]; then
  echo "DuckDB binary is not executable: $DUCKDB_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"
TMP_SQL=$(mktemp "${TMPDIR:-/tmp}/heavydb_range_check_XXXXXX.sql")
trap 'rm -f "$TMP_SQL"' EXIT

cat > "$TMP_SQL" <<SQL
PRAGMA threads=16;

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

CREATE OR REPLACE TEMP VIEW raw_stats AS
SELECT 'lineorder' AS table_name, 'lo_orderdate' AS column_name, min(lo_orderdate)::BIGINT AS min_value, max(lo_orderdate)::BIGINT AS max_value, 'INT' AS selected_heavydb_type FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_quantity', min(lo_quantity)::BIGINT, max(lo_quantity)::BIGINT, 'TINYINT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_extendedprice', min(lo_extendedprice)::BIGINT, max(lo_extendedprice)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_discount', min(lo_discount)::BIGINT, max(lo_discount)::BIGINT, 'TINYINT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_partkey', min(lo_partkey)::BIGINT, max(lo_partkey)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_suppkey', min(lo_suppkey)::BIGINT, max(lo_suppkey)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_revenue', min(lo_revenue)::BIGINT, max(lo_revenue)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_custkey', min(lo_custkey)::BIGINT, max(lo_custkey)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'lineorder', 'lo_supplycost', min(lo_supplycost)::BIGINT, max(lo_supplycost)::BIGINT, 'INT' FROM lineorder_src
UNION ALL SELECT 'customer', 'c_custkey', min(c_custkey)::BIGINT, max(c_custkey)::BIGINT, 'INT' FROM customer_src
UNION ALL SELECT 'supplier', 's_suppkey', min(s_suppkey)::BIGINT, max(s_suppkey)::BIGINT, 'INT' FROM supplier_src
UNION ALL SELECT 'part', 'p_partkey', min(p_partkey)::BIGINT, max(p_partkey)::BIGINT, 'INT' FROM part_src
UNION ALL SELECT 'date_dim', 'd_datekey', min(d_datekey)::BIGINT, max(d_datekey)::BIGINT, 'INT' FROM date_src
UNION ALL SELECT 'date_dim', 'd_year', min(d_year)::BIGINT, max(d_year)::BIGINT, 'SMALLINT' FROM date_src
UNION ALL SELECT 'date_dim', 'd_yearmonthnum', min(d_yearmonthnum)::BIGINT, max(d_yearmonthnum)::BIGINT, 'INT' FROM date_src
UNION ALL SELECT 'date_dim', 'd_weeknuminyear', min(d_weeknuminyear)::BIGINT, max(d_weeknuminyear)::BIGINT, 'TINYINT' FROM date_src
UNION ALL SELECT 'customer/supplier', 'region_id', 1, count(*)::BIGINT, 'TINYINT' FROM (SELECT DISTINCT c_region AS v FROM customer_src UNION SELECT DISTINCT s_region AS v FROM supplier_src)
UNION ALL SELECT 'customer/supplier', 'nation_id', 1, count(*)::BIGINT, 'TINYINT' FROM (SELECT DISTINCT c_nation AS v FROM customer_src UNION SELECT DISTINCT s_nation AS v FROM supplier_src)
UNION ALL SELECT 'customer/supplier', 'city_id', 1, count(*)::BIGINT, 'SMALLINT' FROM (SELECT DISTINCT c_city AS v FROM customer_src UNION SELECT DISTINCT s_city AS v FROM supplier_src)
UNION ALL SELECT 'part', 'p_mfgr_id', 1, count(*)::BIGINT, 'TINYINT' FROM (SELECT DISTINCT p_mfgr AS v FROM part_src)
UNION ALL SELECT 'part', 'p_category_id', 1, count(*)::BIGINT, 'TINYINT' FROM (SELECT DISTINCT p_category AS v FROM part_src)
UNION ALL SELECT 'part', 'p_brand_id', 1, count(*)::BIGINT, 'SMALLINT' FROM (SELECT DISTINCT p_brand AS v FROM part_src);

COPY (
  SELECT table_name,
         column_name,
         min_value,
         max_value,
         CASE
           WHEN min_value >= -127 AND max_value <= 127 THEN 'TINYINT'
           WHEN min_value >= -32767 AND max_value <= 32767 THEN 'SMALLINT'
           WHEN min_value >= -2147483647 AND max_value <= 2147483647 THEN 'INT'
           ELSE 'BIGINT'
         END AS smallest_safe_heavydb_type,
         selected_heavydb_type,
         CASE selected_heavydb_type
           WHEN 'TINYINT' THEN 1
           WHEN 'SMALLINT' THEN 2
           WHEN 'INT' THEN 4
           WHEN 'BIGINT' THEN 8
         END AS selected_bytes
  FROM raw_stats
  ORDER BY table_name, column_name
) TO '$OUT_CSV' (HEADER true, DELIMITER ',');
SQL

"$DUCKDB_BIN" < "$TMP_SQL" >/dev/null
echo "$OUT_CSV"
