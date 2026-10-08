#!/usr/bin/env bash
set -euo pipefail

# HeavyDB on the paper's Stack Overflow Q11 and Q14.
#  - DuckDB preprocess: raw/*.tbl -> per-table TSVs, strings dict-encoded to
#    INTEGER ids (dense rank by name, lexicographic like Proteus dstring);
#    also writes resolved query-constant ids to ids.env.
#  - Narrow types: each integer column uses the smallest signed HeavyDB
#    type validated by the Stack range audit; strings remain dictionary ids.
#  - configurable HeavyDB fragment size: default 32M rows; use a value above
#    every table's cardinality for the one-fragment variant.
#  - Timing: 1 cold + WARM_RUNS warm; client "Execution time" per run plus the
#    HEAVYDB_BENCH_* envelope markers sliced from the server INFO log
#    (preferred metric = EXECUTE_PLAN_ENVELOPE_MS warm mean).
# Queries are the same rewritten forms verified for Proteus
# (tools/workloads/stack/baseline.py), with dict ids substituted.

ROOT_DIR=${HEAVYDB_ROOT:-${ROOT_DIR:-}}
WORK_ROOT=${WORK_ROOT:-${TMPDIR:-/tmp}/proteus-competitors/heavydb}
BUILD_DIR=${BUILD_DIR:-${ROOT_DIR:+$ROOT_DIR/build-release}}
DATA_DIR=${DATA_DIR:-$WORK_ROOT/stack-data}
TBL_DIR=${TBL_DIR:-}
PREP_DIR=${PREP_DIR:-$WORK_ROOT/stack-prepared}
DUCKDB_BIN=${DUCKDB_BIN:-$(command -v duckdb || true)}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/stack-runs}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6274}
FRAG_SIZE=${FRAG_SIZE:-32000000}
PASSWORD=${PASSWORD:-HyperInteractive}
EXTRA_PREFIX=${EXTRA_PREFIX:-}
PIXI_ENV=${PIXI_ENV:-${ROOT_DIR:+$ROOT_DIR/.pixi/envs/default}}
MAVEN_TMP=${MAVEN_TMP:-$WORK_ROOT/maven-tmp}
LOAD_MODE=${LOAD_MODE:-if_missing}
PREPROCESS_MODE=${PREPROCESS_MODE:-if_missing}
WARM_RUNS=${WARM_RUNS:-10}
QUERIES=${QUERIES:-"q11 q14"}

[[ -n "$ROOT_DIR" ]] || { echo "Set HEAVYDB_ROOT to the HeavyDB checkout." >&2; exit 1; }
if [[ "$PREPROCESS_MODE" != "never" ]]; then
  [[ -n "$TBL_DIR" ]] || { echo "Set TBL_DIR to the raw Stack .tbl directory." >&2; exit 1; }
fi
for query in $QUERIES; do
  case "$query" in q11|q14) ;; *) echo "Only paper queries q11 and q14 are supported." >&2; exit 2 ;; esac
done

HEAVYDB_BIN="$BUILD_DIR/bin/heavydb"
HEAVYSQL_BIN="$BUILD_DIR/bin/heavysql"
INITHEAVY_BIN="$BUILD_DIR/bin/initheavy"
LD_PATH="$EXTRA_PREFIX/lib:$PIXI_ENV/lib:${LD_LIBRARY_PATH:-}"
RUN_DIR="$RUN_ROOT/$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$RUN_DIR/logs/heavydb.out"

mkdir -p "$RUN_DIR"/{sql,query_outputs,logs,stats} "$PREP_DIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$RUN_DIR/run.log"; }

run_heavysql() {  # out_file; sql on stdin
  /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
    -p "$PASSWORD" --port "$PORT" -t > "$1" 2>&1
}

server_is_up() {
  printf 'SELECT 1;\n' | /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" \
    "$HEAVYSQL_BIN" -p "$PASSWORD" --port "$PORT" \
    > "$RUN_DIR/logs/connection_check.txt" 2>&1
}

start_server() {
  if server_is_up; then
    log "HeavyDB already running on port $PORT."; SERVER_STARTED=0; return
  fi
  mkdir -p "$DATA_DIR" "$MAVEN_TMP"
  if [[ ! -d "$DATA_DIR/catalogs" ]]; then
    log "Initializing HeavyDB data dir: $DATA_DIR"
    /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$INITHEAVY_BIN" \
      --data "$DATA_DIR" --skip-geo > "$RUN_DIR/logs/initheavy.out" 2>&1
  fi
  log "Starting HeavyDB on GPU $GPU_ID."
  /usr/bin/env TMPDIR="${TMPDIR:-/tmp}" CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HEAVYDB_BENCH_PROFILE_QUERY=1 \
    MAVEN_OPTS="-Djava.io.tmpdir=$MAVEN_TMP" LD_LIBRARY_PATH="$LD_PATH" \
    "$HEAVYDB_BIN" "$DATA_DIR" "--port=$PORT" \
    "--allowed-import-paths=[\"$PREP_DIR\"]" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!; SERVER_STARTED=1
  for _ in $(seq 1 300); do server_is_up && { log "HeavyDB up."; return; }; sleep 1; done
  echo "HeavyDB did not become ready: $SERVER_LOG" >&2; exit 1
}

stop_server() {
  if [[ ${SERVER_STARTED:-0} -eq 1 && -n ${SERVER_PID:-} ]]; then
    log "Stopping HeavyDB PID $SERVER_PID."
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

preprocess() {
  local required=(ids.env site.tsv tag.tsv question.tsv tag_question.tsv answer.tsv so_user.tsv badge.tsv)
  local complete=1
  for f in "${required[@]}"; do
    [[ -s "$PREP_DIR/$f" ]] || { complete=0; break; }
  done
  if [[ "$PREPROCESS_MODE" == never ]] || { [[ "$PREPROCESS_MODE" == if_missing ]] && [[ "$complete" -eq 1 ]]; }; then
    log "Skipping preprocess."; return
  fi
  log "Preprocessing raw .tbl -> dict-encoded TSVs with DuckDB."
  cat > "$RUN_DIR/sql/preprocess.sql" <<EOF
PRAGMA threads=16;
CREATE TEMP VIEW site_src AS SELECT * FROM read_csv('$TBL_DIR/site.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'site_id':'INTEGER','site_name':'VARCHAR'});
CREATE TEMP VIEW tag_src AS SELECT * FROM read_csv('$TBL_DIR/tag.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'id':'INTEGER','site_id':'INTEGER','name':'VARCHAR'});
CREATE TEMP VIEW badge_src AS SELECT * FROM read_csv('$TBL_DIR/badge.tbl',
  header=false, delim='|', quote='', escape='',
  columns={'site_id':'INTEGER','user_id':'INTEGER','name':'VARCHAR'});
CREATE TEMP TABLE sitename_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY site_name) AS INTEGER) AS site_name_id, site_name
  FROM (SELECT DISTINCT site_name FROM site_src);
CREATE TEMP TABLE tagname_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY name) AS INTEGER) AS name_id, name
  FROM (SELECT DISTINCT name FROM tag_src);
CREATE TEMP TABLE badgename_dict AS
  SELECT CAST(ROW_NUMBER() OVER (ORDER BY name) AS INTEGER) AS name_id, name
  FROM (SELECT DISTINCT name FROM badge_src);
COPY (SELECT s.site_id, d.site_name_id FROM site_src s JOIN sitename_dict d USING (site_name))
  TO '$PREP_DIR/site.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT t.id, t.site_id, d.name_id FROM tag_src t JOIN tagname_dict d USING (name))
  TO '$PREP_DIR/tag.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT b.site_id, b.user_id, d.name_id FROM badge_src b JOIN badgename_dict d USING (name))
  TO '$PREP_DIR/badge.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT * FROM read_csv('$TBL_DIR/question.tbl', header=false, delim='|',
  quote='', escape='', columns={'id':'INTEGER','site_id':'INTEGER',
  'owner_user_id':'INTEGER','creation_date':'INTEGER','favorite_count':'INTEGER',
  'view_count':'INTEGER','score':'INTEGER'}))
  TO '$PREP_DIR/question.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT * FROM read_csv('$TBL_DIR/tag_question.tbl', header=false, delim='|',
  quote='', escape='', columns={'question_id':'INTEGER','tag_id':'INTEGER','site_id':'INTEGER'}))
  TO '$PREP_DIR/tag_question.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT * FROM read_csv('$TBL_DIR/answer.tbl', header=false, delim='|',
  quote='', escape='', columns={'id':'INTEGER','site_id':'INTEGER',
  'question_id':'INTEGER','owner_user_id':'INTEGER','creation_date':'INTEGER'}))
  TO '$PREP_DIR/answer.tsv' (DELIMITER '\t', HEADER false);
COPY (SELECT * FROM read_csv('$TBL_DIR/so_user.tbl', header=false, delim='|',
  quote='', escape='', columns={'id':'INTEGER','site_id':'INTEGER',
  'reputation':'INTEGER','downvotes':'INTEGER','account_id':'INTEGER'}))
  TO '$PREP_DIR/so_user.tsv' (DELIMITER '\t', HEADER false);
COPY (
  SELECT 'S_SO='||(SELECT site_name_id FROM sitename_dict WHERE site_name='stackoverflow')
  UNION ALL SELECT 'S_SU='||(SELECT site_name_id FROM sitename_dict WHERE site_name='superuser')
  UNION ALL SELECT 'T_Q11='||(SELECT string_agg(name_id,',') FROM tagname_dict WHERE name IN ('api','audio','events','file-io','forms','functional-programming','java-ee','serialization','sqlite','ubuntu','unicode'))
  UNION ALL SELECT 'T_Q14='||(SELECT string_agg(name_id,',') FROM tagname_dict WHERE name IN ('android-asynctask','asp.net-mvc-4','css3','entity-framework-4','inner-join','jquery','sublimetext3'))
  UNION ALL SELECT 'B_Q14='||(SELECT string_agg(name_id,',') FROM badgename_dict WHERE name IN ('Analytical','Beta','Deputy','Fanatic','Great Question','Not a Robot','Unsung Hero'))
) TO '$PREP_DIR/ids.env' (DELIMITER '\t', HEADER false, QUOTE '');
EOF
  "$DUCKDB_BIN" < "$RUN_DIR/sql/preprocess.sql" > "$RUN_DIR/logs/preprocess.out" 2>&1
  [[ -s "$PREP_DIR/ids.env" ]] || { echo "preprocess failed; see $RUN_DIR/logs/preprocess.out" >&2; exit 1; }
}

table_count() {
  local tmp="$RUN_DIR/logs/count_$1.txt"
  printf 'SELECT COUNT(*) AS n FROM %s;\n' "$1" | \
    /usr/bin/env LD_LIBRARY_PATH="$LD_PATH" "$HEAVYSQL_BIN" \
      -p "$PASSWORD" --port "$PORT" > "$tmp" 2>&1 || return 1
  awk '/^[0-9]+$/ {print; found=1} END {if (!found) exit 1}' "$tmp"
}

load_tables() {
  if [[ "$LOAD_MODE" == never ]]; then return; fi
  if [[ "$LOAD_MODE" == if_missing ]] && table_count badge >/dev/null 2>&1; then
    log "Tables already loaded."; return
  fi
  log "Creating + loading Stack tables (fragment_size=$FRAG_SIZE, narrow integer schema)."
  cat > "$RUN_DIR/sql/schema.sql" <<'SQL'
DROP TABLE IF EXISTS site;
DROP TABLE IF EXISTS tag;
DROP TABLE IF EXISTS question;
DROP TABLE IF EXISTS tag_question;
DROP TABLE IF EXISTS answer;
DROP TABLE IF EXISTS so_user;
DROP TABLE IF EXISTS badge;
CREATE TABLE site (site_id SMALLINT, site_name_id SMALLINT) WITH (fragment_size=32000000);
CREATE TABLE tag (id INTEGER, site_id SMALLINT, name_id INTEGER) WITH (fragment_size=32000000);
CREATE TABLE question (id INTEGER, site_id SMALLINT, owner_user_id INTEGER,
  creation_date INTEGER, favorite_count SMALLINT, view_count INTEGER,
  score SMALLINT) WITH (fragment_size=32000000);
CREATE TABLE tag_question (question_id INTEGER, tag_id INTEGER,
  site_id SMALLINT) WITH (fragment_size=32000000);
CREATE TABLE answer (id INTEGER, site_id SMALLINT, question_id INTEGER,
  owner_user_id INTEGER, creation_date INTEGER) WITH (fragment_size=32000000);
CREATE TABLE so_user (id INTEGER, site_id SMALLINT, reputation INTEGER,
  downvotes INTEGER, account_id INTEGER) WITH (fragment_size=32000000);
CREATE TABLE badge (site_id SMALLINT, user_id INTEGER, name_id TINYINT)
  WITH (fragment_size=32000000);
SQL
  sed -i "s/fragment_size=32000000/fragment_size=${FRAG_SIZE}/g" "$RUN_DIR/sql/schema.sql"
  run_heavysql "$RUN_DIR/logs/schema.out" < "$RUN_DIR/sql/schema.sql"
  for t in site tag question tag_question answer so_user badge; do
    printf "COPY %s FROM '%s' WITH (delimiter='\\t', header='false');\n" \
      "$t" "$PREP_DIR/$t.tsv" | run_heavysql "$RUN_DIR/logs/load_$t.out"
    log "loaded $t: $(table_count "$t" || echo FAILED) rows"
  done
}

info_log() {
  ls -t "$DATA_DIR"/log/heavydb.INFO.*.log 2>/dev/null | head -1 || echo "$SERVER_LOG"
}
log_lines() { local f; f=$(info_log); [[ -f "$f" ]] && wc -l < "$f" || echo 0; }

extract_markers() {  # query run start end
  local slice="$RUN_DIR/logs/profile_$1_run$2.log" f
  f=$(info_log)
  if [[ ! -f "$f" || "$4" -le "$3" ]]; then : > "$slice"
  else sed -n "$(($3 + 1)),$4p" "$f" > "$slice"; fi
  awk -v query="$1" -v run="$2" '
    BEGIN { m[1]="HEAVYDB_BENCH_QUERY_KERNEL_MS"
            m[2]="HEAVYDB_BENCH_GPU_EXEC_ENVELOPE_MS"
            m[3]="HEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS"
            m[4]="HEAVYDB_BENCH_THRIFT_RESULT_CONVERT_MS" }
    { for(i=1;i<=4;i++) if(index($0,m[i])) {
        n=split($0,p,"elapsed_ms="); if(n>1){v=p[2]; sub(/[[:space:]].*/,"",v);
        sum[m[i]]+=v+0; cnt[m[i]]++} } }
    END { for(i=1;i<=4;i++) printf "%s\t%s\t%s\t%d\t%.6f\n",
          query, run, m[i], cnt[m[i]]+0, sum[m[i]]+0 }
  ' "$slice" >> "$RUN_DIR/profile_timings.tsv"
}

write_queries() {
  # shellcheck disable=SC1090
  source "$PREP_DIR/ids.env"
  # Explicit JOIN ... ON syntax ensures Calcite produces composite-key
  # equijoins instead of a cross-product followed by one large filter.
  cat > "$RUN_DIR/sql/q11.sql" <<SQL
SELECT count(*)
FROM tag_question tq
JOIN tag t ON t.id = tq.tag_id AND t.site_id = tq.site_id
JOIN question q ON q.id = tq.question_id AND q.site_id = tq.site_id
JOIN site s ON s.site_id = tq.site_id
WHERE s.site_name_id = ${S_SO}
  AND t.name_id IN (${T_Q11})
  AND q.favorite_count >= 5 AND q.favorite_count <= 5000;
SQL
  # Push the badge-name filter through a derived table before the join chain.
  cat > "$RUN_DIR/sql/q14.sql" <<SQL
SELECT count(*)
FROM (SELECT user_id, site_id FROM badge WHERE name_id IN (${B_Q14})) b
JOIN site s ON s.site_id = b.site_id
JOIN so_user u1 ON u1.id = b.user_id AND u1.site_id = b.site_id
JOIN answer a1 ON a1.owner_user_id = u1.id AND a1.site_id = u1.site_id
JOIN question q1 ON q1.id = a1.question_id AND q1.site_id = a1.site_id
JOIN tag_question tq1 ON tq1.question_id = q1.id AND tq1.site_id = q1.site_id
JOIN tag t1 ON t1.id = tq1.tag_id AND t1.site_id = tq1.site_id
WHERE s.site_name_id IN (${S_SO},${S_SU})
  AND t1.name_id IN (${T_Q14})
  AND q1.view_count >= 10 AND q1.view_count <= 1000
  AND u1.downvotes >= 0 AND u1.downvotes <= 10
  AND u1.account_id <> -1;
SQL
}

extract_timing_ms() {
  awk -F'Execution time: | ms' '/Execution time:/ {print $2; exit}' "$1"
}

run_queries() {
  write_queries
  printf 'query\trun\tms\n' > "$RUN_DIR/timings.tsv"
  : > "$RUN_DIR/profile_timings.tsv"
  for q in $QUERIES; do
    total=$((1 + WARM_RUNS))
    for run in $(seq 1 "$total"); do
      before=$(log_lines)
      out="$RUN_DIR/query_outputs/${q}_run${run}.txt"
      run_heavysql "$out" < "$RUN_DIR/sql/$q.sql" || true
      after=$(log_lines)
      ms=$(extract_timing_ms "$out"); ms=${ms:-NA}
      printf '%s\t%s\t%s\n' "$q" "$run" "$ms" >> "$RUN_DIR/timings.tsv"
      extract_markers "$q" "$run" "$before" "$after"
    done
    log "done $q"
  done
}

preprocess
start_server
load_tables
run_queries
log "RUN_DIR: $RUN_DIR"
