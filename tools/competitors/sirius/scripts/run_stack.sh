#!/usr/bin/env bash
# Sirius (legacy gpu_processing) on the paper's Stack Overflow Q11 and Q14.
# Narrow types use INT32 plus SMALLINT where
# the Stack range audit proves the values fit; strings are dictionary ids
# resolved at run time (dense rank by name = lexicographic, like Proteus
# dstring dicts). Sirius's GPU path does not use INT8 in this setup, so the
# narrowest type used here is SMALLINT.
# Queries are the SAME rewritten forms verified against DuckDB for Proteus
# (tools/workloads/stack/baseline.py), with dict ids substituted.
#
# Usage:
#   run_stack.sh load   # build the database from raw/*.tbl (CPU only)
#   run_stack.sh run [q11 q14]   # GPU benchmark
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMING_CHECKER="$SCRIPT_DIR/../../check_timings.py"
WORK_ROOT="${WORK_ROOT:-${TMPDIR:-/tmp}/proteus-competitors/sirius}"
SIRIUS_BUILD_DIR="${SIRIUS_BUILD_DIR:-${TMPDIR:-/tmp}/sirius-legacy-release}"
DUCKDB_BIN="${DUCKDB_BIN:-$SIRIUS_BUILD_DIR/duckdb}"
DB_PATH="${DB_PATH:-$WORK_ROOT/stack.duckdb}"
TBL_DIR="${TBL_DIR:-}"
OUT_ROOT="${OUT_ROOT:-$WORK_ROOT/stack-runs}"
ITERATIONS="${ITERATIONS:-11}"
GPU_CACHE_SIZE="${GPU_CACHE_SIZE:-12 GB}"
GPU_PROCESSING_SIZE="${GPU_PROCESSING_SIZE:-20 GB}"
PINNED_MEMORY_SIZE="${PINNED_MEMORY_SIZE:-32 GB}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

MODE="${1:-}"
shift || true

do_load() {
  [ -x "$DUCKDB_BIN" ] || { echo "missing Sirius DuckDB binary: $DUCKDB_BIN"; exit 1; }
  [ -n "$TBL_DIR" ] || { echo "set TBL_DIR to the raw Stack .tbl directory"; exit 1; }
  for t in site tag question tag_question answer so_user badge; do
    [ -f "$TBL_DIR/$t.tbl" ] || { echo "missing $TBL_DIR/$t.tbl"; exit 1; }
  done
  TMP_SQL=$(mktemp "${TMPDIR:-/tmp}/stack_std_load_XXXXXX.sql")
  trap 'rm -f "$TMP_SQL"' EXIT
  cat > "$TMP_SQL" <<EOF
PRAGMA threads=16;
DROP TABLE IF EXISTS site; DROP TABLE IF EXISTS tag;
DROP TABLE IF EXISTS question; DROP TABLE IF EXISTS tag_question;
DROP TABLE IF EXISTS answer; DROP TABLE IF EXISTS so_user;
DROP TABLE IF EXISTS badge;
DROP TABLE IF EXISTS sitename_dict; DROP TABLE IF EXISTS tagname_dict;
DROP TABLE IF EXISTS badgename_dict;

CREATE TEMP VIEW site_src AS SELECT * FROM read_csv('$TBL_DIR/site.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'site_id':'INTEGER','site_name':'VARCHAR'});
CREATE TEMP VIEW tag_src AS SELECT * FROM read_csv('$TBL_DIR/tag.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'id':'INTEGER','site_id':'INTEGER','name':'VARCHAR'});
CREATE TEMP VIEW badge_src AS SELECT * FROM read_csv('$TBL_DIR/badge.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'site_id':'INTEGER','user_id':'INTEGER','name':'VARCHAR'});

CREATE TABLE sitename_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY site_name) AS INTEGER) AS site_name_id,
         site_name
  FROM (SELECT DISTINCT site_name FROM site_src);
CREATE TABLE tagname_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY name) AS INTEGER) AS name_id, name
  FROM (SELECT DISTINCT name FROM tag_src);
CREATE TABLE badgename_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY name) AS INTEGER) AS name_id, name
  FROM (SELECT DISTINCT name FROM badge_src);

CREATE TABLE site AS
  SELECT CAST(s.site_id AS SMALLINT) AS site_id,
         CAST(d.site_name_id AS SMALLINT) AS site_name_id
  FROM site_src s JOIN sitename_dict d USING (site_name);
CREATE TABLE tag AS
  SELECT t.id, CAST(t.site_id AS SMALLINT) AS site_id, d.name_id
  FROM tag_src t JOIN tagname_dict d USING (name);
CREATE TABLE badge AS
  SELECT CAST(b.site_id AS SMALLINT) AS site_id, b.user_id,
         CAST(d.name_id AS SMALLINT) AS name_id
  FROM badge_src b JOIN badgename_dict d USING (name);

CREATE TABLE question AS
SELECT id, CAST(site_id AS SMALLINT) AS site_id, owner_user_id, creation_date,
       CAST(favorite_count AS SMALLINT) AS favorite_count, view_count,
       CAST(score AS SMALLINT) AS score
FROM read_csv('$TBL_DIR/question.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'id':'INTEGER','site_id':'INTEGER','owner_user_id':'INTEGER',
           'creation_date':'INTEGER','favorite_count':'INTEGER',
           'view_count':'INTEGER','score':'INTEGER'});
CREATE TABLE tag_question AS
SELECT question_id, tag_id, CAST(site_id AS SMALLINT) AS site_id
FROM read_csv('$TBL_DIR/tag_question.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'question_id':'INTEGER','tag_id':'INTEGER','site_id':'INTEGER'});
CREATE TABLE answer AS
SELECT id, CAST(site_id AS SMALLINT) AS site_id, question_id, owner_user_id,
       creation_date
FROM read_csv('$TBL_DIR/answer.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'id':'INTEGER','site_id':'INTEGER','question_id':'INTEGER',
           'owner_user_id':'INTEGER','creation_date':'INTEGER'});
CREATE TABLE so_user AS
SELECT id, CAST(site_id AS SMALLINT) AS site_id, reputation, downvotes,
       account_id
FROM read_csv('$TBL_DIR/so_user.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'id':'INTEGER','site_id':'INTEGER','reputation':'INTEGER',
           'downvotes':'INTEGER','account_id':'INTEGER'});
SELECT 'site', count(*) FROM site UNION ALL
SELECT 'tag', count(*) FROM tag UNION ALL
SELECT 'question', count(*) FROM question UNION ALL
SELECT 'tag_question', count(*) FROM tag_question UNION ALL
SELECT 'answer', count(*) FROM answer UNION ALL
SELECT 'so_user', count(*) FROM so_user UNION ALL
SELECT 'badge', count(*) FROM badge;
EOF
  "$DUCKDB_BIN" "$DB_PATH" < "$TMP_SQL"
  echo "LOAD_DONE $DB_PATH"
}

duckdb_scalar() {
  "$DUCKDB_BIN" "$DB_PATH" -noheader -list -c "$1" | tr -d '"'
}

# id lookups (resolved once per run)
resolve_ids() {
  SITE_ID() { duckdb_scalar "SELECT site_name_id FROM sitename_dict WHERE site_name='$1';"; }
  S_SO=$(SITE_ID stackoverflow); S_SU=$(SITE_ID superuser)
  T_Q11=$(duckdb_scalar "SELECT string_agg(name_id,',') FROM tagname_dict WHERE name IN ('api','audio','events','file-io','forms','functional-programming','java-ee','serialization','sqlite','ubuntu','unicode');")
  T_Q14=$(duckdb_scalar "SELECT string_agg(name_id,',') FROM tagname_dict WHERE name IN ('android-asynctask','asp.net-mvc-4','css3','entity-framework-4','inner-join','jquery','sublimetext3');")
  B_Q14=$(duckdb_scalar "SELECT string_agg(name_id,',') FROM badgename_dict WHERE name IN ('Analytical','Beta','Deputy','Fanatic','Great Question','Not a Robot','Unsung Hero');")
}

query_sql() {
  case "$1" in
    q11) cat <<SQL
SELECT count(*) FROM tag t, site s, question q, tag_question tq
WHERE t.site_id = s.site_id AND q.site_id = s.site_id
  AND tq.site_id = s.site_id
  AND tq.question_id = q.id AND tq.tag_id = t.id
  AND s.site_name_id = ${S_SO}
  AND t.name_id IN (${T_Q11})
  AND q.favorite_count >= 5 AND q.favorite_count <= 5000
SQL
;;
    q14) cat <<SQL
SELECT count(*)
FROM site s, so_user u1, question q1, answer a1, tag t1, tag_question tq1, badge b
WHERE s.site_id = q1.site_id AND s.site_id = u1.site_id
  AND s.site_id = a1.site_id AND s.site_id = t1.site_id
  AND s.site_id = tq1.site_id AND s.site_id = b.site_id
  AND q1.id = tq1.question_id AND q1.id = a1.question_id
  AND a1.owner_user_id = u1.id AND t1.id = tq1.tag_id
  AND b.user_id = u1.id AND u1.account_id <> -1
  AND s.site_name_id IN (${S_SO},${S_SU})
  AND t1.name_id IN (${T_Q14})
  AND q1.view_count >= 10 AND q1.view_count <= 1000
  AND u1.downvotes >= 0 AND u1.downvotes <= 10
  AND b.name_id IN (${B_Q14})
SQL
;;
  esac
}

do_run() {
  [ -x "$DUCKDB_BIN" ] || { echo "missing Sirius DuckDB binary: $DUCKDB_BIN"; exit 1; }
  [ -f "$DB_PATH" ] || { echo "missing Stack database: $DB_PATH"; exit 1; }
  [ -f "$TIMING_CHECKER" ] || { echo "missing timing checker: $TIMING_CHECKER"; exit 1; }
  [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "ITERATIONS must be a positive integer" >&2; exit 2; }
  QUERIES=("$@"); [ ${#QUERIES[@]} -gt 0 ] || QUERIES=(q11 q14)
  for query in "${QUERIES[@]}"; do
    case "$query" in q11|q14) ;; *) echo "only paper queries q11 and q14 are supported"; exit 2 ;; esac
  done
  OUT_DIR="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT_DIR"
  resolve_ids
  SUMMARY="$OUT_DIR/summary.csv"
  echo "query,iteration,status,runtime_s,execute_ms" > "$SUMMARY"
  failures=0
  for q in "${QUERIES[@]}"; do
    qdir="$OUT_DIR/$q"; mkdir -p "$qdir"
    sql_file="$qdir/query.sql"; log_file="$qdir/time.log"
    {
      echo "CALL gpu_buffer_init('${GPU_CACHE_SIZE}', '${GPU_PROCESSING_SIZE}', pinned_memory_size='${PINNED_MEMORY_SIZE}');"
      echo ".timer on"
      for i in $(seq 1 "$ITERATIONS"); do
        echo ".print __SIRIUS_${q}_ITER_${i}_BEGIN__"
        printf 'CALL gpu_processing("%s");\n' "$(query_sql "$q" | tr '\n' ' ' | sed 's/"/""/g')"
        echo ".print __SIRIUS_${q}_ITER_${i}_END__"
      done
    } > "$sql_file"
    previous_lines=$(wc -l < "$SUMMARY")
    query_rc=0
    "$DUCKDB_BIN" "$DB_PATH" < "$sql_file" > "$log_file" 2>&1 || query_rc=$?
    awk -v query="$q" '
      $0 ~ "__SIRIUS_" query "_ITER_" && $0 ~ "_BEGIN__" {
        marker=$0; sub(/^.*_ITER_/,"",marker); sub(/_BEGIN__.*$/,"",marker)
        current=marker+0; begun[current]=1; next }
      $0 ~ "__SIRIUS_" query "_ITER_" && $0 ~ "_END__" {
        marker=$0; sub(/^.*_ITER_/,"",marker); sub(/_END__.*$/,"",marker)
        ended[marker+0]=1; current=0; next }
      current>0 && /fallback to DuckDB/ { fb[current]=1 }
      current>0 && /Catalog Error|Parser Error|Binder Error|Invalid Error|Segmentation fault|ERROR:|Error:/ { err[current]=1 }
      current>0 && /Execute query time:/ { ex[current]=$(NF-1); if(current>max)max=current }
      current>0 && /Run Time \(s\): real/ { rt[current]=$5; if(current>max)max=current }
      END { for(i=1;i<=max;i++){ s="ok"; if(!begun[i] || !ended[i])s="error"; if(err[i])s="error"; if(fb[i])s="fallback";
        printf "%s,%d,%s,%s,%s\n", query, i, s,
          (rt[i]==""?"NA":rt[i]), (ex[i]==""?"NA":ex[i]) } }
    ' "$log_file" >> "$SUMMARY"
    validation_rc=0
    python3 "$TIMING_CHECKER" sirius \
      --file "$SUMMARY" \
      --query "$q" \
      --iterations "$ITERATIONS" \
      --from-line "$previous_lines" || validation_rc=$?
    if (( query_rc != 0 || validation_rc != 0 )); then
      echo "ERROR: $q failed (duckdb status $query_rc, timing validation status $validation_rc); see $log_file" >&2
      failures=$((failures + 1))
    fi
    echo "done $q -> $log_file"
  done
  echo "SUMMARY: $SUMMARY"
  cat "$SUMMARY"
  if (( failures != 0 )); then
    echo "ERROR: $failures Sirius Stack query run(s) failed; partial results remain in $OUT_DIR" >&2
    return 1
  fi
}

case "$MODE" in
  load) do_load ;;
  run) do_run "$@" ;;
  *) echo "usage: $0 load|run [q11 q14]"; exit 1 ;;
esac
