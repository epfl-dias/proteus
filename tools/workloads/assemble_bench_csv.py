#!/usr/bin/env python3
# Combine a bench_matrix.sh run (per-config timings) with footprint.py (space occupancy)
# into one CSV: rows = (workload, query, codec, mode, tile, tpb, ipt), columns = query time
# + space occupancy. NONE has no decode mode/geometry (mode/tile/tpb/ipt = "-"); RFOR/FLBP
# get one row per (mode, geometry). Space is geometry-independent; RFOR tile-512 uses the
# _512 dataset footprint (RFOR512), else falls back to the tile-1024 RFOR footprint.
#
#   python3 tools/workloads/assemble_bench_csv.py <matrix_out_base> <footprint.csv> > bench.csv
import sys, os, re, csv, glob

base, fp_csv = sys.argv[1], sys.argv[2]
space = {}  # (wl,q,codec) -> (bytes, MiB, ncols)
with open(fp_csv) as f:
    for r in csv.DictReader(f):
        space[(r["workload"], r["query"], r["codec"])] = (r["space_bytes"], r["space_MiB"], r["n_scanned_cols"])

TAG = re.compile(r"^(rfor|flbp)_(op|reg|pd)_(.+)$")
def parse_tag(tag):
    if tag.startswith("none_"): return "NONE", None, tag[5:]
    m = TAG.match(tag)
    return (m.group(1).upper(), m.group(2), m.group(3)) if m else (None, None, None)

def space_for(wl, q, codec, tile):
    key = (wl, q, codec)
    if codec == "RFOR" and tile == "512":
        alt = space.get((wl, q, "RFOR512"))
        if alt and alt[0] not in ("", "0"): return alt
    return space.get(key, ("", "", ""))

rows = []          # (wl,q,codec,mode,tile,tpb,ipt,time,MiB,bytes,ncols,gate)
none_seen = set()
for d in sorted(glob.glob(f"{base}/*_*_t*x*")):
    m = re.search(r"/([a-z]+)_(op|reg|pd)_t(\d+)_(\d+)x(\d+)$", d)
    if not m: continue
    wl, dmode, tile, tpb, ipt = m.groups()
    pl = f"{d}/progress.log"
    if not os.path.exists(pl): continue
    for line in open(pl):
        text = line.strip()
        mm = re.match(r"(\S+) time=([\d.]+|NA) ms result_vs_none=(\S+)", text)
        sm = re.match(r"(\S+) SKIP \((.*)\)", text)
        if mm:
            tag, t, gate = mm.groups()
        elif sm:
            tag, reason = sm.groups()
            t, gate = "NA", "SKIP:" + reason.replace(",", ";")
        else:
            continue
        codec, tmode, q = parse_tag(tag)
        if codec is None: continue
        if codec == "NONE":
            if (wl, q) in none_seen: continue
            none_seen.add((wl, q))
            sp = space.get((wl, q, "NONE"), ("", "", ""))
            rows.append((wl, q, "NONE", "-", "-", "-", "-", t, sp[1], sp[0], sp[2], gate))
        else:
            sp = space_for(wl, q, codec, tile)
            rows.append((wl, q, codec, dmode, tile, tpb, ipt, t, sp[1], sp[0], sp[2], gate))

cor = {"NONE":0, "RFOR":1, "FLBP":2}; mor = {"-":0, "op":1, "reg":2, "pd":3}
rows.sort(key=lambda r: (r[0], r[1], cor.get(r[2],9), mor.get(r[3],9), r[4], r[5], r[6]))
w = csv.writer(sys.stdout)
w.writerow(["workload","query","codec","mode","tile","tpb","ipt","time_ms","space_MiB","space_bytes","n_scanned_cols","result_vs_none"])
w.writerows(rows)
