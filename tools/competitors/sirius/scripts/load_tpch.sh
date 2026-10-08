#!/usr/bin/env bash

set -euo pipefail

SIRIUS_BUILD_DIR="${SIRIUS_BUILD_DIR:-${TMPDIR:-/tmp}/sirius-legacy-release}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <duckdb_db_path> <tpch_tbl_dir> [duckdb_bin]"
  echo "Example: $0 /data/bench/tpch.duckdb /data/tpch/sf100"
  exit 1
fi

DB_PATH="$1"
TBL_DIR="$2"
DUCKDB_BIN="${3:-$SIRIUS_BUILD_DIR/duckdb}"

for required in lineitem.tbl orders.tbl customer.tbl supplier.tbl part.tbl \
                nation.tbl region.tbl; do
  if [ ! -f "$TBL_DIR/$required" ]; then
    echo "ERROR: missing input file: $TBL_DIR/$required"
    exit 1
  fi
done

if [ ! -x "$DUCKDB_BIN" ]; then
  echo "ERROR: DuckDB binary not found or not executable: $DUCKDB_BIN"
  exit 1
fi

TMP_SQL="$(mktemp "${TMPDIR:-/tmp}/tpch_reduced_load_XXXXXX.sql")"
trap 'rm -f "$TMP_SQL"' EXIT

cat > "$TMP_SQL" <<EOF
-- Reduced integer-only TPC-H schema for legacy Sirius gpu_processing.
-- Type choices follow the TPC-H baseline notes:
--   - INTEGER where values fit but Sirius has no 8-bit execution type
--   - INTEGER for dates, keys, and scaled money values

PRAGMA threads=16;

DROP VIEW IF EXISTS lineitem_src;
DROP VIEW IF EXISTS orders_src;
DROP VIEW IF EXISTS customer_src;
DROP VIEW IF EXISTS supplier_src;
DROP VIEW IF EXISTS part_src;
DROP VIEW IF EXISTS nation_src;
DROP VIEW IF EXISTS region_src;

DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS returnflag_dict;
DROP TABLE IF EXISTS linestatus_dict;
DROP TABLE IF EXISTS mktsegment_dict;
DROP TABLE IF EXISTS shipmode_dict;
DROP TABLE IF EXISTS shipinstruct_dict;
DROP TABLE IF EXISTS orderpriority_dict;
DROP TABLE IF EXISTS brand_dict;
DROP TABLE IF EXISTS container_dict;
DROP TABLE IF EXISTS regionname_dict;

CREATE OR REPLACE TEMP VIEW lineitem_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/lineitem.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  quote = '"',
  columns = {
    'l_orderkey': 'INTEGER',
    'l_partkey': 'INTEGER',
    'l_suppkey': 'INTEGER',
    'l_linenumber': 'INTEGER',
    'l_quantity': 'DECIMAL(15,2)',
    'l_extendedprice': 'DECIMAL(15,2)',
    'l_discount': 'DECIMAL(15,2)',
    'l_tax': 'DECIMAL(15,2)',
    'l_returnflag': 'VARCHAR',
    'l_linestatus': 'VARCHAR',
    'l_shipdate': 'VARCHAR',
    'l_commitdate': 'VARCHAR',
    'l_receiptdate': 'VARCHAR',
    'l_shipinstruct': 'VARCHAR',
    'l_shipmode': 'VARCHAR',
    'l_comment': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW orders_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/orders.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  quote = '"',
  columns = {
    'o_orderkey': 'INTEGER',
    'o_custkey': 'INTEGER',
    'o_orderstatus': 'VARCHAR',
    'o_totalprice': 'DECIMAL(15,2)',
    'o_orderdate': 'VARCHAR',
    'o_orderpriority': 'VARCHAR',
    'o_clerk': 'VARCHAR',
    'o_shippriority': 'INTEGER',
    'o_comment': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW customer_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/customer.tbl',
  auto_detect = false,
  header = false,
  delim = '|',
  quote = '"',
  columns = {
    'c_custkey': 'INTEGER',
    'c_name': 'VARCHAR',
    'c_address': 'VARCHAR',
    'c_nationkey': 'INTEGER',
    'c_phone': 'VARCHAR',
    'c_acctbal': 'DECIMAL(15,2)',
    'c_mktsegment': 'VARCHAR',
    'c_comment': 'VARCHAR'
  }
);

CREATE TABLE returnflag_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_returnflag) - 1 AS INTEGER) AS returnflag_id,
       l_returnflag
FROM (SELECT DISTINCT l_returnflag FROM lineitem_src);

CREATE TABLE linestatus_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_linestatus) - 1 AS INTEGER) AS linestatus_id,
       l_linestatus
FROM (SELECT DISTINCT l_linestatus FROM lineitem_src);

CREATE TABLE mktsegment_dict AS
SELECT CAST(row_number() OVER (ORDER BY c_mktsegment) - 1 AS INTEGER) AS mktsegment_id,
       c_mktsegment
FROM (SELECT DISTINCT c_mktsegment FROM customer_src);

CREATE OR REPLACE TEMP VIEW supplier_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/supplier.tbl',
  auto_detect = false, header = false, delim = '|', quote = '"',
  columns = {
    's_suppkey': 'INTEGER', 's_name': 'VARCHAR', 's_address': 'VARCHAR',
    's_nationkey': 'INTEGER', 's_phone': 'VARCHAR',
    's_acctbal': 'DECIMAL(15,2)', 's_comment': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW part_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/part.tbl',
  auto_detect = false, header = false, delim = '|', quote = '"',
  columns = {
    'p_partkey': 'INTEGER', 'p_name': 'VARCHAR', 'p_mfgr': 'VARCHAR',
    'p_brand': 'VARCHAR', 'p_type': 'VARCHAR', 'p_size': 'INTEGER',
    'p_container': 'VARCHAR', 'p_retailprice': 'DECIMAL(15,2)',
    'p_comment': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW nation_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/nation.tbl',
  auto_detect = false, header = false, delim = '|', quote = '"',
  columns = {
    'n_nationkey': 'INTEGER', 'n_name': 'VARCHAR',
    'n_regionkey': 'INTEGER', 'n_comment': 'VARCHAR'
  }
);

CREATE OR REPLACE TEMP VIEW region_src AS
SELECT * FROM read_csv(
  '$TBL_DIR/region.tbl',
  auto_detect = false, header = false, delim = '|', quote = '"',
  columns = {
    'r_regionkey': 'INTEGER', 'r_name': 'VARCHAR', 'r_comment': 'VARCHAR'
  }
);

CREATE TABLE shipmode_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_shipmode) - 1 AS INTEGER) AS shipmode_id,
       l_shipmode
FROM (SELECT DISTINCT l_shipmode FROM lineitem_src);

CREATE TABLE shipinstruct_dict AS
SELECT CAST(row_number() OVER (ORDER BY l_shipinstruct) - 1 AS INTEGER) AS shipinstruct_id,
       l_shipinstruct
FROM (SELECT DISTINCT l_shipinstruct FROM lineitem_src);

CREATE TABLE orderpriority_dict AS
SELECT CAST(row_number() OVER (ORDER BY o_orderpriority) - 1 AS INTEGER) AS orderpriority_id,
       o_orderpriority
FROM (SELECT DISTINCT o_orderpriority FROM orders_src);

CREATE TABLE brand_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_brand) - 1 AS INTEGER) AS brand_id,
       p_brand
FROM (SELECT DISTINCT p_brand FROM part_src);

CREATE TABLE container_dict AS
SELECT CAST(row_number() OVER (ORDER BY p_container) - 1 AS INTEGER) AS container_id,
       p_container
FROM (SELECT DISTINCT p_container FROM part_src);

CREATE TABLE regionname_dict AS
SELECT CAST(row_number() OVER (ORDER BY r_name) - 1 AS INTEGER) AS regionname_id,
       r_name
FROM (SELECT DISTINCT r_name FROM region_src);

CREATE TABLE lineitem (
  l_orderkey INTEGER NOT NULL,
  l_shipdate INTEGER NOT NULL,
  l_extendedprice INTEGER NOT NULL,
  l_discount INTEGER NOT NULL,
  l_quantity INTEGER NOT NULL,
  l_tax INTEGER NOT NULL,
  l_returnflag_id INTEGER NOT NULL,
  l_linestatus_id INTEGER NOT NULL,
  l_partkey INTEGER NOT NULL,
  l_suppkey INTEGER NOT NULL,
  l_commitdate INTEGER NOT NULL,
  l_receiptdate INTEGER NOT NULL,
  l_shipmode_id INTEGER NOT NULL,
  l_shipinstruct_id INTEGER NOT NULL
);

INSERT INTO lineitem
SELECT l.l_orderkey,
       CAST(replace(l.l_shipdate, '-', '') AS INTEGER) AS l_shipdate,
       CAST(round(l.l_extendedprice * 100) AS INTEGER) AS l_extendedprice,
       CAST(round(l.l_discount * 100) AS INTEGER) AS l_discount,
       CAST(l.l_quantity AS INTEGER) AS l_quantity,
       CAST(round(l.l_tax * 100) AS INTEGER) AS l_tax,
       r.returnflag_id,
       s.linestatus_id,
       l.l_partkey,
       l.l_suppkey,
       CAST(replace(l.l_commitdate, '-', '') AS INTEGER) AS l_commitdate,
       CAST(replace(l.l_receiptdate, '-', '') AS INTEGER) AS l_receiptdate,
       sm.shipmode_id,
       si.shipinstruct_id
FROM lineitem_src l
JOIN returnflag_dict r ON l.l_returnflag = r.l_returnflag
JOIN linestatus_dict s ON l.l_linestatus = s.l_linestatus
JOIN shipmode_dict sm ON l.l_shipmode = sm.l_shipmode
JOIN shipinstruct_dict si ON l.l_shipinstruct = si.l_shipinstruct;

CREATE TABLE orders (
  o_orderkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  o_custkey INTEGER NOT NULL,
  o_orderdate INTEGER NOT NULL,
  o_shippriority INTEGER NOT NULL,
  o_orderpriority_id INTEGER NOT NULL
);

INSERT INTO orders
SELECT o.o_orderkey,
       o.o_custkey,
       CAST(replace(o.o_orderdate, '-', '') AS INTEGER) AS o_orderdate,
       o.o_shippriority,
       op.orderpriority_id
FROM orders_src o
JOIN orderpriority_dict op ON o.o_orderpriority = op.o_orderpriority;

CREATE TABLE customer (
  c_custkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  c_mktsegment_id INTEGER NOT NULL,
  c_nationkey INTEGER NOT NULL
);

INSERT INTO customer
SELECT c.c_custkey,
       m.mktsegment_id,
       CAST(c.c_nationkey AS INTEGER)
FROM customer_src c
JOIN mktsegment_dict m ON c.c_mktsegment = m.c_mktsegment;

CREATE TABLE supplier (
  s_suppkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  s_nationkey INTEGER NOT NULL
);

INSERT INTO supplier
SELECT s_suppkey, CAST(s_nationkey AS INTEGER) FROM supplier_src;

CREATE TABLE part (
  p_partkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  p_brand_id INTEGER NOT NULL,
  p_container_id INTEGER NOT NULL,
  p_size INTEGER NOT NULL,
  p_promo INTEGER NOT NULL
);

INSERT INTO part
SELECT p.p_partkey,
       b.brand_id,
       ct.container_id,
       CAST(p.p_size AS INTEGER),
       CAST(CASE WHEN p.p_type LIKE 'PROMO%' THEN 1 ELSE 0 END AS INTEGER)
FROM part_src p
JOIN brand_dict b ON p.p_brand = b.p_brand
JOIN container_dict ct ON p.p_container = ct.p_container;

CREATE TABLE nation (
  n_nationkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  n_regionkey INTEGER NOT NULL
);

INSERT INTO nation
SELECT CAST(n_nationkey AS INTEGER), CAST(n_regionkey AS INTEGER) FROM nation_src;

CREATE TABLE region (
  r_regionkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
  r_name_id INTEGER NOT NULL
);

INSERT INTO region
SELECT CAST(r.r_regionkey AS INTEGER), rn.regionname_id
FROM region_src r
JOIN regionname_dict rn ON r.r_name = rn.r_name;

CREATE TABLE tpch_integer_load_stats AS
SELECT 'lineitem_rows' AS metric, count(*)::VARCHAR AS value FROM lineitem
UNION ALL
SELECT 'orders_rows', count(*)::VARCHAR FROM orders
UNION ALL
SELECT 'customer_rows', count(*)::VARCHAR FROM customer
UNION ALL
SELECT 'supplier_rows', count(*)::VARCHAR FROM supplier
UNION ALL
SELECT 'part_rows', count(*)::VARCHAR FROM part
UNION ALL
SELECT 'nation_rows', count(*)::VARCHAR FROM nation
UNION ALL
SELECT 'region_rows', count(*)::VARCHAR FROM region
UNION ALL
SELECT 'returnflag_cardinality', count(*)::VARCHAR FROM returnflag_dict
UNION ALL
SELECT 'linestatus_cardinality', count(*)::VARCHAR FROM linestatus_dict
UNION ALL
SELECT 'mktsegment_cardinality', count(*)::VARCHAR FROM mktsegment_dict
UNION ALL
SELECT 'building_mktsegment_id', CAST(mktsegment_id AS VARCHAR)
FROM mktsegment_dict
WHERE c_mktsegment = 'BUILDING';
EOF

echo "Loading reduced integer TPC-H DB"
echo "  DB:     $DB_PATH"
echo "  input:  $TBL_DIR"
echo "  duckdb: $DUCKDB_BIN"

"$DUCKDB_BIN" "$DB_PATH" < "$TMP_SQL"

echo "Created tables:"
echo "  lineitem(l_orderkey INTEGER, l_shipdate INTEGER, l_extendedprice INTEGER,"
echo "           l_discount INTEGER, l_quantity INTEGER, l_tax INTEGER,"
echo "           l_returnflag_id INTEGER, l_linestatus_id INTEGER)"
echo "  orders(o_orderkey INTEGER, o_custkey INTEGER, o_orderdate INTEGER, o_shippriority INTEGER)"
echo "  customer(c_custkey INTEGER, c_mktsegment_id INTEGER)"
echo "  returnflag_dict, linestatus_dict, mktsegment_dict"
echo "Stats:"
"$DUCKDB_BIN" "$DB_PATH" -csv -c "SELECT * FROM tpch_integer_load_stats ORDER BY metric;"
