# Star Schema Benchmark
This directory implements the SSB using Proteus' PreparedQueries/QueryShaper infrastructure. 

Assumes that input data is located at inputs/ssbm{100|1000} relative to the working directory of the executable. 

The benchmark uses GoogleBenchmark. See the [GoogleBenchmark user docs](https://github.com/google/benchmark/blob/main/docs/user_guide.md) for command line arguments, such as filtering which benchmarks to run.
e.g. to only run SSB on the CPU at scale factor 100, you may use:
```shell
./proteus-benchmark-ssb --benchmark_filter=.*CPU.*\/100\/.*
```
