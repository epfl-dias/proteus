#!/usr/bin/env bash

set -euo pipefail

SIRIUS_BUILD_DIR="${SIRIUS_BUILD_DIR:-${TMPDIR:-/tmp}/sirius-legacy-release}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <duckdb_db_path> <ssb_tbl_dir> [duckdb_bin]"
  echo "Example: $0 /data/bench/ssb.duckdb /data/ssb/sf100"
  exit 1
fi

DB_PATH="$1"
TBL_DIR="$2"
DUCKDB_BIN="${3:-$SIRIUS_BUILD_DIR/duckdb}"

for required in lineorder.tbl customer.tbl supplier.tbl part.tbl date.tbl; do
  if [ ! -f "$TBL_DIR/$required" ]; then
    echo "ERROR: missing input file: $TBL_DIR/$required"
    exit 1
  fi
done

if [ ! -x "$DUCKDB_BIN" ]; then
  echo "ERROR: DuckDB binary not found or not executable: $DUCKDB_BIN"
  exit 1
fi

TMP_SQL="$(mktemp "${TMPDIR:-/tmp}/ssb_reduced_load_XXXXXX.sql")"
trap 'rm -f "$TMP_SQL"' EXIT

cat > "$TMP_SQL" <<EOF
-- Reduced integer-only SSB schema for legacy Sirius gpu_processing.
-- Type choices are conservative for the legacy path:
--   - INTEGER only where values clearly fit in signed 16-bit
--   - INTEGER elsewhere, even if a tighter bit-packed format would exist in another engine

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
DROP TABLE IF EXISTS brand1_dict;

CREATE OR REPLACE TEMP VIEW lineorder_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/lineorder.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  columns = {
    'lo_orderkey': 'INTEGER',
    'lo_linenumber': 'INTEGER',
    'lo_custkey': 'INTEGER',
    'lo_partkey': 'INTEGER',
    'lo_suppkey': 'INTEGER',
    'lo_orderdate': 'INTEGER',
    'lo_orderpriority': 'VARCHAR',
    'lo_shippriority': 'VARCHAR',
    'lo_quantity': 'INTEGER',
    'lo_extendedprice': 'INTEGER',
    'lo_ordertotalprice': 'INTEGER',
    'lo_discount': 'INTEGER',
    'lo_revenue': 'INTEGER',
    'lo_supplycost': 'INTEGER',
    'lo_tax': 'INTEGER',
    'lo_commitdate': 'INTEGER',
    'lo_shipmode': 'VARCHAR',
    '__end': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW customer_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/customer.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  columns = {
    'c_custkey': 'INTEGER',
    'c_name': 'VARCHAR',
    'c_address': 'VARCHAR',
    'c_city': 'VARCHAR',
    'c_nation': 'VARCHAR',
    'c_region': 'VARCHAR',
    'c_phone': 'VARCHAR',
    'c_mktsegment': 'VARCHAR',
    '__end': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW supplier_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/supplier.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  columns = {
    's_suppkey': 'INTEGER',
    's_name': 'VARCHAR',
    's_address': 'VARCHAR',
    's_city': 'VARCHAR',
    's_nation': 'VARCHAR',
    's_region': 'VARCHAR',
    's_phone': 'VARCHAR',
    '__end': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW part_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/part.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  columns = {
    'p_partkey': 'INTEGER',
    'p_name': 'VARCHAR',
    'p_mfgr': 'VARCHAR',
    'p_category': 'VARCHAR',
    'p_brand1': 'VARCHAR',
    'p_color': 'VARCHAR',
    'p_type': 'VARCHAR',
    'p_size': 'INTEGER',
    'p_container': 'VARCHAR',
    '__end': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW date_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/date.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  columns = {
    'd_datekey': 'INTEGER',
    'd_date': 'VARCHAR',
    'd_dayofweek': 'VARCHAR',
    'd_month': 'VARCHAR',
    'd_year': 'INTEGER',
    'd_yearmonthnum': 'INTEGER',
    'd_yearmonth': 'VARCHAR',
    'd_daynuminweek': 'INTEGER',
    'd_daynuminmonth': 'INTEGER',
    'd_daynuminyear': 'INTEGER',
    'd_monthnuminyear': 'INTEGER',
    'd_weeknuminyear': 'INTEGER',
    'd_sellingseason': 'VARCHAR',
    'd_lastdayinweekfl': 'VARCHAR',
    'd_lastdayinmonthfl': 'VARCHAR',
    'd_holidayfl': 'VARCHAR',
    'd_weekdayfl': 'VARCHAR',
    '__end': 'VARCHAR'
  }
);

CREATE TABLE region_dict AS
SELECT CAST(row_number() OVER (ORDER BY region) AS INTEGER) AS region_id,
       region
FROM (
  SELECT DISTINCT c_region AS region FROM customer_src
  UNION
  SELECT DISTINCT s_region AS region FROM supplier_src
);

CREATE TABLE nation_dict AS
SELECT CAST(row_number() OVER (ORDER BY nation) AS INTEGER) AS nation_id,
       nation
FROM (
  SELECT DISTINCT c_nation AS nation FROM customer_src
  UNION
  SELECT DISTINCT s_nation AS nation FROM supplier_src
);

CREATE TABLE city_dict AS
SELECT CAST(row_number() OVER (ORDER BY city) AS INTEGER) AS city_id,
       city
FROM (
  SELECT DISTINCT c_city AS city FROM customer_src
  UNION
  SELECT DISTINCT s_city AS city FROM supplier_src
);

CREATE TABLE mfgr_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_mfgr) AS INTEGER) AS mfgr_id,
       p_mfgr
FROM (
  SELECT DISTINCT p_mfgr FROM part_src
);

CREATE TABLE category_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_category) AS INTEGER) AS category_id,
       p_category
FROM (
  SELECT DISTINCT p_category FROM part_src
);

CREATE TABLE brand1_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_brand1) AS INTEGER) AS brand1_id,
       p_brand1
FROM (
  SELECT DISTINCT p_brand1 FROM part_src
);

CREATE TABLE lineorder (
  lo_orderdate INTEGER NOT NULL,
  lo_quantity INTEGER NOT NULL,
  lo_extendedprice INTEGER NOT NULL,
  lo_discount INTEGER NOT NULL,
  lo_partkey INTEGER NOT NULL,
  lo_suppkey INTEGER NOT NULL,
  lo_revenue INTEGER NOT NULL,
  lo_custkey INTEGER NOT NULL,
  lo_supplycost INTEGER NOT NULL
);

INSERT INTO lineorder
SELECT lo_orderdate,
       lo_quantity,
       lo_extendedprice,
       lo_discount,
       lo_partkey,
       lo_suppkey,
       lo_revenue,
       lo_custkey,
       lo_supplycost
FROM lineorder_src;

CREATE TABLE customer (
  c_custkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  c_region_id INTEGER NOT NULL,
  c_nation_id INTEGER NOT NULL,
  c_city_id INTEGER NOT NULL
);

INSERT INTO customer
SELECT c.c_custkey,
       r.region_id,
       n.nation_id,
       ci.city_id
FROM customer_src c
JOIN region_dict r ON c.c_region = r.region
JOIN nation_dict n ON c.c_nation = n.nation
JOIN city_dict ci ON c.c_city = ci.city;

CREATE TABLE supplier (
  s_suppkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  s_region_id INTEGER NOT NULL,
  s_nation_id INTEGER NOT NULL,
  s_city_id INTEGER NOT NULL
);

INSERT INTO supplier
SELECT s.s_suppkey,
       r.region_id,
       n.nation_id,
       ci.city_id
FROM supplier_src s
JOIN region_dict r ON s.s_region = r.region
JOIN nation_dict n ON s.s_nation = n.nation
JOIN city_dict ci ON s.s_city = ci.city;

CREATE TABLE part (
  p_partkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  p_mfgr_id INTEGER NOT NULL,
  p_category_id INTEGER NOT NULL,
  p_brand1_id INTEGER NOT NULL
);

INSERT INTO part
SELECT p.p_partkey,
       m.mfgr_id,
       c.category_id,
       b.brand1_id
FROM part_src p
JOIN mfgr_dict m ON p.p_mfgr = m.p_mfgr
JOIN category_dict c ON p.p_category = c.p_category
JOIN brand1_dict b ON p.p_brand1 = b.p_brand1;

CREATE TABLE date_dim (
  d_datekey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  d_year INTEGER NOT NULL,
  d_yearmonthnum INTEGER NOT NULL,
  d_weeknuminyear INTEGER NOT NULL
);

INSERT INTO date_dim
SELECT d_datekey,
       d_year,
       d_yearmonthnum,
       d_weeknuminyear
FROM date_src;

SELECT 'lineorder_rows', count(*) FROM lineorder;
SELECT 'customer_rows', count(*) FROM customer;
SELECT 'supplier_rows', count(*) FROM supplier;
SELECT 'part_rows', count(*) FROM part;
SELECT 'date_rows', count(*) FROM date_dim;
SELECT 'region_cardinality', count(*) FROM region_dict;
SELECT 'nation_cardinality', count(*) FROM nation_dict;
SELECT 'city_cardinality', count(*) FROM city_dict;
SELECT 'mfgr_cardinality', count(*) FROM mfgr_dict;
SELECT 'category_cardinality', count(*) FROM category_dict;
SELECT 'brand1_cardinality', count(*) FROM brand1_dict;
EOF

echo "Loading reduced integer-only SSB schema"
echo "  DB:      $DB_PATH"
echo "  TBL dir: $TBL_DIR"
echo "  DuckDB:  $DUCKDB_BIN"

"$DUCKDB_BIN" "$DB_PATH" < "$TMP_SQL"

echo ""
echo "Done."
echo "Tables created:"
echo "  lineorder(lo_orderdate INTEGER, lo_quantity INTEGER, lo_extendedprice INTEGER,"
echo "            lo_discount INTEGER, lo_partkey INTEGER, lo_suppkey INTEGER,"
echo "            lo_revenue INTEGER, lo_custkey INTEGER, lo_supplycost INTEGER)"
echo "  customer(c_custkey INTEGER, c_region_id INTEGER, c_nation_id INTEGER, c_city_id INTEGER)"
echo "  supplier(s_suppkey INTEGER, s_region_id INTEGER, s_nation_id INTEGER, s_city_id INTEGER)"
echo "  part(p_partkey INTEGER, p_mfgr_id INTEGER, p_category_id INTEGER, p_brand1_id INTEGER)"
echo "  date_dim(d_datekey INTEGER, d_year INTEGER, d_yearmonthnum INTEGER, d_weeknuminyear INTEGER)"
echo ""
echo "Dictionary tables created:"
echo "  region_dict, nation_dict, city_dict, mfgr_dict, category_dict, brand1_dict"
