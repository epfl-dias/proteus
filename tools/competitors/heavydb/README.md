# HeavyDB benchmark

The artifact targets HeavyDB v9.0.0 at paper commit `da23f70` (full revision
`da23f70ea060f308bb9536ec8515a4ef1d07a4bd`). The patch contains the build
compatibility changes used on the benchmark host and opt-in server timing
markers. Marker collection is enabled only when the runner sets
`HEAVYDB_BENCH_PROFILE_QUERY=1`.

## Build

The paper setup used CUDA 11.8, GCC 12.4, and LLVM/Clang 14.0. The locked Pixi
environment supplies GCC 12.4 and the remaining CMake/Ninja dependencies.
Clone the pinned source and run the complete runtime build:

```bash
git clone https://github.com/heavyai/heavydb.git
cd heavydb
git checkout da23f70ea060f308bb9536ec8515a4ef1d07a4bd
cd /path/to/proteus

export HEAVYDB_ROOT=/path/to/heavydb
export CUDA_ROOT=/usr/local/cuda-11.8
bash tools/competitors/heavydb/scripts/build.sh
```

The helper applies the included patch, installs the locked Pixi environment,
configures a release build, and builds `heavydb`, `heavysql`, and `initheavy`.
Set `LLVM_DIR`, `CLANG_LIB`, `CUDA_HOST_COMPILER`, or `EXTRA_CMAKE_ARGS` when
those dependencies are installed in nonstandard locations. Set `EXTRA_PREFIX`
for an additional dependency prefix. The included minimal Clang package shim
matches the benchmark build and links through the explicitly selected
`libclang-cpp` shared library.

## Run

Set a work directory on storage with enough space for the SF100 integer TSVs,
HeavyDB storage, and logs. A DuckDB CLI is used only for deterministic data
preprocessing; the Sirius build from the sibling directory is suitable.

```bash
export HEAVYDB_ROOT=/path/to/heavydb
export BUILD_DIR=$HEAVYDB_ROOT/build-release
export WORK_ROOT=/data/cb2t-heavydb
export DUCKDB_BIN=/path/to/duckdb

bash tools/competitors/heavydb/scripts/run_ssb.sh \
  --ssb-dir /data/ssb/sf100

bash tools/competitors/heavydb/scripts/run_tpch.sh \
  --raw-dir /data/tpch/sf100

TBL_DIR=/data/stack/raw \
  bash tools/competitors/heavydb/scripts/run_stack.sh
```

The SSB and TPC-H runners preprocess, load, start HeavyDB, execute the paper
query set listed in the parent README, summarize timing markers, and stop only
the server they started.
Use `--skip-preprocess` or `--skip-load` to reuse materialized state. The Stack
runner has the equivalent `PREPROCESS_MODE` and `LOAD_MODE` environment
controls. `WARM_RUNS` defaults to 10. `FRAG_SIZE` defaults to the paper's
32-million-row HeavyDB fragments; set it only when intentionally reproducing a
different fragment-count sensitivity point.
The runners exit nonzero if any requested run fails or lacks a valid numeric
timing; partial timing files and raw logs remain in the run directory.

The preferred HeavyDB metric is the warm mean of
`HEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS` in each run directory's
`stats/profile_timing_summary.tsv` (or `profile_timings.tsv` for Stack).
