# CB2T experiment artifact

This directory contains the Proteus-only workflow used to generate the input
datasets and reproduce the query measurements from the CB2T paper. The reported
experiments were built and run directly on a bare-metal NVIDIA A100 host.

The artifact includes no measured results, plots, or private experiment notes.
HeavyDB and Sirius use the separate
[`tools/competitors`](../competitors/README.md) workflow.

## Bare-metal requirements

- A Linux host with an NVIDIA A100 and CUDA 11.3.
- LLVM/Clang 14.0 built with libc++ and its CMake package files.
- CMake 3.23 or newer and the standard Proteus system dependencies.
- nvCOMP 4.2 for CUDA 11, including its CMake package files.
- NVIDIA GPUDirect Storage (`cufile.h` and `libcufile`), normally supplied by
  the CUDA toolkit. Set `GDS_ROOT` if it is installed separately.
- `numactl`, Python 3, Bash, and sufficient local storage for the SF100 data.
- NVIDIA Nsight Compute only for the optional memory-traffic profile.

Point the build helper at the LLVM and nvCOMP installations. Proteus uses the
`Debug` CMake configuration for experiments; the project still enables its
optimized compiler flags in this configuration.

```bash
export LLVM_ROOT=/opt/llvm-14
export CUDA_ROOT=/usr/local/cuda-11.3
export NVCOMP_ROOT=/opt/nvcomp/nvcomp-linux-x86_64-4.2.0.11_cuda11-archive
bash tools/workloads/build_bare_metal.sh
```

The helper performs an unfiltered build of the complete CB2T artifact
configuration: the three query benchmarks, their query libraries, the OLAP
runtime, and the dataset partitioner. The artifact configuration excludes
legacy planner, OLTP, test, and unrelated application targets that are not used
by these experiments. It does not install the build. Set `BUILD` to use a
directory other than `cmake-build-release`. Keep `LLVM_ROOT` exported when
generating datasets or running benchmarks so libc++ and LLVM are visible to the
executables.

## Dataset layout

Set `PROTEUS_DATA_ROOT` to a directory with the following uncompressed Proteus
datasets. Each directory must contain `catalog.json` and the block data and
metadata produced by the standard Proteus data-preparation workflow.

```text
$PROTEUS_DATA_ROOT/
  ssb/uncompressed/
  tpch/uncompressed/
  stack/uncompressed/
```

The evaluated queries are:

- SSB: Q1.1, Q2.1, Q3.1, Q4.1
- TPC-H: Q5, Q12
- Stack: Q11, Q14

Generate the RFOR tile-1024/tile-512 and FLBP tile-1024 variants:

```bash
export PROTEUS_DATA_ROOT=/data/proteus-cb2t
export BUILD="$PWD/cmake-build-release"
bash tools/workloads/generate_cb2t_datasets.sh
```

`generate_cb2t_datasets.sh` materializes logical column bytes before
repartitioning, generates only the columns used by the TPC-H and Stack query
sets, and validates row counts and compression metadata. Existing complete
outputs are skipped. Set `FORCE=1` to replace an incomplete or stale output.
The same dataset-driven generator handles SSB, TPC-H, and Stack; it reads each
workload's schema from `catalog.json` or the supplied query-column list.

Every dataset path can also be overridden independently, for example with
`TPCH_NONE_DS`, `TPCH_RFOR_DS`, or `STACK_FLBP_DS`. The complete variable list
and default layout are in `common.sh`.

## Timing matrix

The benchmark matrix uses the execution-placement names from the paper:

- `Shared`: fused CB2T with decoded tiles in shared memory
- `Register`: fused CB2T with decoded values retained in registers
- `Global-stage`: fused CB2T with a bounded global-memory staging area
- `Pre-decom`: whole-input pre-decompression into global memory
- `Uncom`: uncompressed reference

Run the main Shared/Register/Pre-decom matrix:

```bash
export PROTEUS_DATA_ROOT=/data/proteus-cb2t
export BUILD="$PWD/cmake-build-release"
export WORKLOAD_OUTPUT_ROOT=/data/proteus-cb2t-runs
bash tools/workloads/bench_matrix.sh
```

The default geometry matrix is RFOR/FLBP tile 1024 at 256x4 and 128x8,
plus RFOR tile 512 at 128x4 and 256x2. Override `WORKLOADS`, `MODES`,
`GEOMS_1024`, or `GEOMS_512` to run a subset.

Run one workload or query directly:

```bash
WORKLOAD=tpch MODE=shared QUERIES="Q5 Q12" \
  bash tools/workloads/run_modes.sh
```

`run_modes.sh` records timing only. It intentionally disables result dumping so
that output materialization does not perturb the measurement. A missing dataset,
failed process, or missing millisecond timing makes the runner fail. Use
`gate_results.sh` for the separate correctness matrix described below.

## Global-stage

Run the bounded Global-stage sweep used for the paper:

```bash
bash tools/workloads/run_global_stage_sweep.sh
```

The A100 default is an 18-row-group window. `CONFIGS` accepts explicit
`codec:tile:threads-per-block:items-per-thread` records, and `WINDOWS` controls
the staging-window sweep.

For the focused SSB Q1.1 Nsight Compute traffic experiment:

```bash
NCU=/path/to/ncu bash tools/workloads/run_global_stage_ncu.sh
```

All commands write logs and generated CSV summaries under
`WORKLOAD_OUTPUT_ROOT` (default: `$TMPDIR/proteus-cb2t`). No generated output is
stored in the repository.

## Separate correctness matrix

Run from the same build and dataset configuration as the timing matrix:

```bash
bash tools/workloads/gate_results.sh
python3 -m unittest discover -s tools/workloads/tests
```

The gate covers SSB Q1.1/Q2.1/Q3.1/Q4.1, TPC-H Q5/Q12, and Stack
Q11/Q14. Each query runs an uncompressed reference, then RFOR and FLBP
at tile 1024 with 256x4 and 128x8, and RFOR at tile 512 with 128x4 and
256x2. Every compressed configuration runs `Shared`, `Register`, `Pre-decom`,
and `Global-stage`. FLBP tile 512 is
unsupported and is never scheduled. Global-stage runs windows 16 (the
`run_modes.sh` default) and 18 (the published Global-stage sweep default).
This is 248 executions: 8 references and 240 compressed checks.
Set `WORKLOADS="ssb"` or `WINDOWS="18"` to select a subset; a subset pass
is not evidence for the full matrix. `SSB_QUERIES`, `TPCH_QUERIES`, and
`STACK_QUERIES` select published queries for focused diagnosis and reject any
query outside the paper scope. `LAYOUTS` and `MODES` can narrow a diagnostic
run to the named supported layouts and modes. Any subset pass is diagnostic
only. For a Global-stage sweep with additional windows, use the same `WINDOWS`
list here. Custom experimental geometries outside the published defaults
require extending the gate's layout table.

The gate enables result dumping only in its own benchmark child processes.
These runs still execute benchmark warmups/iterations, but their timings must
not be used as performance measurements. `run_modes.sh` explicitly unsets an
inherited `PROTEUS_RESULT_DUMP_DIR` and continues reporting `NOT_CHECKED`.
Timing, correctness, sensitivity, and profiling runners share one fixture lock
because they change the same input symlinks. A concurrent runner exits instead
of switching another run's dataset. The gate restores the prior symlinks on exit.

Every invocation creates a fresh `correctness-*` directory under
`WORKLOAD_OUTPUT_ROOT` (or an existing `LOG_DIR` parent). It retains the base
commit, per-run shell-quoted command and dataset path, process logs, result
files, and `results.tsv`. All selected datasets must contain `catalog.json`.
Missing binaries or datasets, missing/empty/multiple outputs, schema failures,
comparison failures, and all nonzero process exits fail the gate. In
particular, teardown aborts (rc 134) are infrastructure failures even if a
result was dumped. When a usable result exists, the gate still compares it and
records `match` or `mismatch` separately from the process failure. `results.tsv`
classifies exact completed-result differences as `query_mismatch` and setup,
process, or capture failures as `infrastructure`; logs remain for diagnosis.
The gate continues to record remaining configurations and exits nonzero.

Every row is first checked against the published projection's field count.
CSV fields are then compared exactly, without floating-point tolerances. Scalar
aggregates (SSB Q1.1 and Stack Q11/Q14) and totally ordered results (SSB
Q2.1/Q4.1 and TPC-H Q12) are order-sensitive. TPC-H Q5 orders by revenue;
SSB Q3.1 orders by year and revenue. For these two queries only, rows within
equal ORDER BY key groups are compared as multisets, retaining duplicate
multiplicities and the sequence of key groups. This permits SQL-legal tie
permutations without hiding an ordering regression. Q5 projects
`(n_name, revenue)`, so its tie key is field 1; Q3.1 projects
`(c_nation, s_nation, d_year, revenue)`, so its tie key is fields 2 and 3.
These indices follow `prepared-queries/tpch/q05.cpp` and
`prepared-queries/ssb/q3_1.cpp`.

The legacy `gate_rfor512_results.sh` remains a narrow historical check and is
superseded by this gate for artifact-wide coverage.

### Validation status and A100 requirements

CPU-only tests exercise matrix enumeration, order/tie/duplicate handling,
and failure on missing datasets, absent/multiple outputs, mismatches, and
teardown aborts using a simulated launcher. They do not validate GPU results.
All 248 real executions remain unverified until run with the built artifact
and matching experiment datasets on the A100 setup. This includes every SSB
query, both tile-1024 codecs, both geometries at each supported tile size,
and all four execution modes. The memory-pool defaults reproduce the A100
policy from `a100_gpu_buffer_policy.sh`; `GPU_BUFFERS`, `UNCOM_GPU_BUFFERS`,
`PRE_DECOM_GPU_BUFFERS`, and `CPU_BUFFERS` can be overridden. Other compatible CUDA
hardware may run the gate with suitable resources, but cannot establish the
published A100 configuration's correctness. A passing comparison establishes
agreement with this artifact's uncompressed implementation, not independent
SQL-oracle validation.
