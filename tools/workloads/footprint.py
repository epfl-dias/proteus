#!/usr/bin/env python3
# Space occupancy per (workload, query, codec) = GPU-resident bytes of the columns a
# query scans = sum of the on-disk <rel>.csv.<col>_<part>_<n> block-file sizes in that
# codec's dataset. Geometry-independent (tpb/ipt change only the decode launch config,
# not the resident data). Scanned columns are parsed from each query's scan("rel",{...}).
#
#   python3 tools/workloads/footprint.py > footprint.csv
#
# PORTABILITY: the dataset paths below must match run_modes.sh for this machine. Adjust
# WORKLOADS[*]["ds"] when moving to another box (e.g. the A100 server).
import re, os, glob, csv, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get("PROTEUS_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
QDIR = os.environ.get(
    "QDIR", os.path.join(REPO_ROOT, "apps", "benchmarks", "olap", "prepared-queries"))
def env_path(name, default):
    return os.environ.get(name, default)

WORKLOADS = {
  "ssb":  {"dir": f"{QDIR}/ssb",  "q": {"Q1_1":"q1_1_nojoin","Q2_1":"q2_1","Q3_1":"q3_1","Q4_1":"q4_1"},
           "ds": {"NONE":env_path("SSB_NONE_DS", f"{DATA_ROOT}/ssb/uncompressed"),
                  "RFOR":env_path("SSB_RFOR_DS", f"{DATA_ROOT}/ssb/rfor-t1024"),
                  "RFOR512":env_path("SSB_RFOR512_DS", f"{DATA_ROOT}/ssb/rfor-t512"),
                  "FLBP":env_path("SSB_FLBP_DS", f"{DATA_ROOT}/ssb/flbp-t1024")}},
  "tpch": {"dir": f"{QDIR}/tpch", "q": {"Q5":"q05","Q12":"q12"},
           "ds": {"NONE":env_path("TPCH_NONE_DS", f"{DATA_ROOT}/tpch/uncompressed"),
                  "RFOR":env_path("TPCH_RFOR_DS", f"{DATA_ROOT}/tpch/rfor-t1024"),
                  "RFOR512":env_path("TPCH_RFOR512_DS", f"{DATA_ROOT}/tpch/rfor-t512"),
                  "FLBP":env_path("TPCH_FLBP_DS", f"{DATA_ROOT}/tpch/flbp-t1024")}},
  "stack":{"dir": f"{QDIR}/stack","q": {"Q11":"q11","Q14":"q14"},
           "ds": {"NONE":env_path("STACK_NONE_DS", f"{DATA_ROOT}/stack/uncompressed"),
                  "RFOR":env_path("STACK_RFOR_DS", f"{DATA_ROOT}/stack/rfor-t1024"),
                  "RFOR512":env_path("STACK_RFOR512_DS", f"{DATA_ROOT}/stack/rfor-t512"),
                  "FLBP":env_path("STACK_FLBP_DS", f"{DATA_ROOT}/stack/flbp-t1024")}},
}
SCAN = re.compile(r'scan\(\s*"([A-Za-z_0-9]+)"\s*,\s*\{([^}]*)\}', re.S)
COL  = re.compile(r'"([A-Za-z_0-9]+)"')
PREPARED_STATEMENT = re.compile(r'(?m)^[ \t]*PreparedStatement\s+')

def scanned_cols(path):
    txt = open(path, encoding="utf-8").read()
    prepared_statements = list(PREPARED_STATEMENT.finditer(txt))
    if prepared_statements:
        start = prepared_statements[0].start()
        end = prepared_statements[1].start() if len(prepared_statements) > 1 else len(txt)
        txt = txt[start:end]
    return [(rel, c) for rel, cols in SCAN.findall(txt) for c in COL.findall(cols)]

def col_bytes(dsdir, rel, col):
    total = 0
    for f in glob.glob(f"{dsdir}/{rel}.csv.{col}_*_*"):
        b = os.path.basename(f)
        if b.endswith(".metadata.json"): continue
        if not re.search(rf"\.{re.escape(col)}_\d+_\d+$", b): continue
        try: total += os.path.getsize(os.path.realpath(f))
        except OSError: pass
    return total

w = csv.writer(sys.stdout)
w.writerow(["workload","query","codec","n_scanned_cols","space_bytes","space_MiB"])
for wl, cfg in WORKLOADS.items():
    for qid, fbase in cfg["q"].items():
        cols = scanned_cols(f"{cfg['dir']}/{fbase}.cpp")
        for codec, dsdir in cfg["ds"].items():
            fp = sum(col_bytes(dsdir, rel, c) for rel, c in cols)
            w.writerow((wl, qid, codec, len(cols), fp, round(fp/2**20, 2)))
