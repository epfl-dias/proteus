# Competitor benchmark artifact

This directory reproduces the HeavyDB and Sirius comparisons from the CB2T
paper on a bare-metal NVIDIA A100. It contains source patches, runtime
build commands, integer dataset loaders, the evaluated SQL queries, and timing
runners. Generated databases, logs, timing CSVs, and profiler output are not
part of the repository.

The public competitor runners expose only the queries reported in the paper:

- SSB: Q1.1, Q2.1, Q3.1, Q4.1
- TPC-H: Q5, Q12
- Stack: Q11, Q14

Both engines consume the deterministic reduced-integer schemas used by the
paper. SSB and TPC-H use the final `*_std` Sirius loaders; the latest Stack
comparison uses `SMALLINT` where its range audit proves the values fit. HeavyDB
uses its smallest validated signed type. String predicates use
lexicographically ordered dictionary IDs.

Use the engine-specific instructions in [heavydb/README.md](heavydb/README.md)
and [sirius/README.md](sirius/README.md). The runners perform one cold iteration
and ten measured iterations by default. Keep their generated work directories
outside the Proteus checkout. The query selectors reject workloads outside the
paper scope above.

The paper toolchains are system-specific: Proteus uses CUDA 11.3 and
LLVM/Clang 14.0; HeavyDB `da23f70` uses CUDA 11.8, GCC 12.4, and LLVM/Clang
14.0; Sirius `8d8353a` uses DuckDB v1.5.2, CUDA 12.9, and GCC 14.3.
