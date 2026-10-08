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

Every dataset path can also be overridden independently, for example with
`TPCH_NONE_DS`, `TPCH_RFOR_DS`, or `STACK_FLBP_DS`. The complete variable list
and default layout are in `common.sh`.

## Timing matrix

The benchmark matrix covers the paper's execution modalities:

- `op`: fused shared-memory CB2T
- `reg`: register-materializing CB2T
- `stream`: bounded global-staging CB2T
- `pd`: pre-decompression
- `NONE`: uncompressed reference

Run the main fused/register/pre-decompression matrix:

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
WORKLOAD=tpch MODE=op QUERIES="Q5 Q12" \
  bash tools/workloads/run_modes.sh
```

`run_modes.sh` records timing only. It intentionally disables result dumping so
that output materialization does not perturb the measurement. Use
`gate_rfor512_results.sh` for the separate byte-identity correctness gate.

## Global staging

Run the bounded global-staging sweep used for the paper:

```bash
bash tools/workloads/run_streaming_sweep.sh
```

The A100 default is an 18-row-group window. `CONFIGS` accepts explicit
`codec:tile:threads-per-block:items-per-thread` records, and `WINDOWS` controls
the staging-window sweep.

For the focused SSB Q1.1 Nsight Compute traffic experiment:

```bash
NCU=/path/to/ncu bash tools/workloads/run_streaming_ncu.sh
```

All commands write logs and generated CSV summaries under
`WORKLOAD_OUTPUT_ROOT` (default: `$TMPDIR/proteus-cb2t`). No generated output is
stored in the repository.
