#!/usr/bin/env bash
# Full Proteus paper matrix: each published query x codec x decode geometry,
# via run_modes.sh. Two tile passes:
#   - tile 1024: NONE + RFOR + FLBP, geometries {256x4, 128x8}
#   - tile  512: RFOR only (FLBP vector is fixed 1024), geometries {128x4, 256x2}
#     (uses the *_512 dataset; run_modes.sh skips a query if that dataset is absent)
# Times per (query, codec, tile, geometry); pair with footprint.py for space occupancy.
# Portable: dataset paths live in run_modes.sh (env-overridable) -- adjust for the A100 box.
#
#   bash tools/workloads/bench_matrix.sh
#   WORKLOADS="ssb" OUT_BASE=/path bash tools/workloads/bench_matrix.sh
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
RM="$HERE/run_modes.sh"
WORKLOADS="${WORKLOADS:-ssb tpch stack}"
MODES="${MODES:-op reg pd}"               # decode modes: op=--decom_op, reg=--decom_reg, pd=--pre_decom
GEOMS_1024="${GEOMS_1024:-256x4 128x8}"   # tpb x ipt == 1024 (avoid 512x2 / 64x16)
GEOMS_512="${GEOMS_512:-128x4 256x2}"     # tpb x ipt == 512  (RFOR only; avoid 64x8)
OUT_BASE="${OUT_BASE:-${LOG_DIR:-$WORKLOAD_OUTPUT_ROOT}/bench_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_BASE"
echo "=== bench matrix modes=[$MODES] -> $OUT_BASE ==="
run_pass() { # mode tile codecs geoms
  local mode="$1" tile="$2" codecs="$3" geoms="$4"
  for wl in $WORKLOADS; do
    for g in $geoms; do
      echo "### $wl mode=$mode tile=$tile geom=$g codecs=[$codecs] ###"
      WORKLOAD="$wl" MODE="$mode" TILE="$tile" CODECS="$codecs" \
        TPB="${g%x*}" IPT="${g#*x}" OUT="$OUT_BASE/${wl}_${mode}_t${tile}_${g}" bash "$RM"
    done
  done
}
for m in $MODES; do
  run_pass "$m" 1024 "RFOR FLBP" "$GEOMS_1024"
  run_pass "$m" 512  "RFOR"      "$GEOMS_512"
done
echo "MATRIX_DONE  (per-config progress.log under $OUT_BASE/<wl>_t<tile>_<geom>/)"
