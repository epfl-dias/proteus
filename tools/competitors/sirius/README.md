# Sirius benchmark

The artifact targets paper commit `8d8353a` (full revision
`8d8353a31ec2b1a46a90981fd064d592e3e5afe2`) with its DuckDB v1.5.2 submodule.
The included patch selects the
legacy `gpu_processing` execution path used for the comparison and exposes the
CUDA profiler controls used by the optional profiling modes.

## Build

The paper setup used DuckDB v1.5.2, CUDA 12.9, and GCC 14.3. Clone the pinned
source with its submodules, then build the legacy runtime with the upstream
locked `cuda12` Pixi environment:

```bash
git clone --recurse-submodules https://github.com/sirius-db/sirius.git
cd sirius
git checkout 8d8353a31ec2b1a46a90981fd064d592e3e5afe2
cd /path/to/proteus

export SIRIUS_ROOT=/path/to/sirius
export SIRIUS_BUILD_DIR=/data/build/sirius-legacy-release
bash tools/competitors/sirius/scripts/build.sh
```

The build helper applies the patch, verifies the DuckDB v1.5.2 submodule,
installs the frozen environment that resolves CUDA 12.9 and GCC 14.3, and
builds DuckDB plus the local Sirius extension. Its output binary is
`$SIRIUS_BUILD_DIR/duckdb`.

## Prepare data

The loaders convert raw SF100 `.tbl` files to the integer-only DuckDB schemas
used by Sirius. The SSB and TPC-H `std` schemas use 32-bit `INTEGER`; the latest
Stack paper run uses `SMALLINT` for range-audited narrow columns.

```bash
SIRIUS_BUILD_DIR=/data/build/sirius-legacy-release \
  bash tools/competitors/sirius/scripts/load_ssb.sh \
  /data/cb2t-sirius/ssb.duckdb /data/ssb/sf100

SIRIUS_BUILD_DIR=/data/build/sirius-legacy-release \
  bash tools/competitors/sirius/scripts/load_tpch.sh \
  /data/cb2t-sirius/tpch.duckdb /data/tpch/sf100

SIRIUS_BUILD_DIR=/data/build/sirius-legacy-release \
DB_PATH=/data/cb2t-sirius/stack.duckdb \
TBL_DIR=/data/stack/raw \
  bash tools/competitors/sirius/scripts/run_stack.sh load
```

## Run

```bash
export SIRIUS_BUILD_DIR=/data/build/sirius-legacy-release
export WORK_ROOT=/data/cb2t-sirius

bash tools/competitors/sirius/scripts/run_ssb.sh \
  --db $WORK_ROOT/ssb.duckdb

bash tools/competitors/sirius/scripts/run_tpch.sh \
  --db $WORK_ROOT/tpch.duckdb

DB_PATH=$WORK_ROOT/stack.duckdb \
  bash tools/competitors/sirius/scripts/run_stack.sh run
```

SSB and TPC-H default to 11 iterations: one warmup followed by ten measured
runs. Their per-query GPU cache and processing-pool defaults are encoded in the
runners exactly as used by the latest paper runs. Their `summary.csv` files
report each selected pool size, per-iteration status, and execution time.
The optional `--mode nsys`, `--mode ncu`, and `--mode profile` modes require the
corresponding NVIDIA command-line tools and write their output below the
selected run directory.
