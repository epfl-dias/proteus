/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
        Data Intensive Applications and Systems Laboratory (DIAS)
                Ecole Polytechnique Federale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include <cli-flags.hpp>
#include <olap-perf-util/tpch-aliases.hpp>
#include <platform/topology/topology.hpp>
#include <tpch/query.hpp>

#include "tpch-fixture.hpp"

#define TPCH_BENCH_DEFINE_CPU(name, prepFunction) \
  BENCHMARK_DEFINE_F(TPCHCPUOnly, CPU_##name)     \
  (benchmark::State & st) {                       \
    auto statement = prepFunction(*shaper.get()); \
    warmUp(statement);                            \
    runBenchmark(statement, st);                  \
  }

#define TPCH_BENCH_REGISTER_CPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(TPCHCPUOnly, CPU_##name)     \
      ->Args({100, 1})                              \
      ->UseRealTime()                               \
      ->Unit(benchmark::kMillisecond)               \
      ->Iterations(5);

FOR_TPCH_QUERY(TPCH_BENCH_DEFINE_CPU)
FOR_TPCH_QUERY(TPCH_BENCH_REGISTER_CPU)

#define TPCH_BENCH_DEFINE_GPU(name, prepFunction)          \
  BENCHMARK_DEFINE_F(TPCHGPUOnly, GPU_##name)              \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define TPCH_BENCH_REGISTER_GPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(TPCHGPUOnly, GPU_##name)     \
      ->Args({100, 1})                              \
      ->UseRealTime()                               \
      ->Unit(benchmark::kMillisecond)               \
      ->Iterations(5);

FOR_TPCH_QUERY(TPCH_BENCH_DEFINE_GPU)
FOR_TPCH_QUERY(TPCH_BENCH_REGISTER_GPU)

#define TPCH_BENCH_DEFINE_GPU_LAZY(name, prepFunction)     \
  BENCHMARK_DEFINE_F(TPCHGPUOnlyLazy, GPULazy_##name)      \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define TPCH_BENCH_REGISTER_GPU_LAZY(name, prepFunction) \
  BENCHMARK_REGISTER_F(TPCHGPUOnlyLazy, GPULazy_##name)  \
      ->Args({100, 1})                                   \
      ->UseRealTime()                                    \
      ->Unit(benchmark::kMillisecond)                    \
      ->Iterations(3);

FOR_TPCH_QUERY(TPCH_BENCH_DEFINE_GPU_LAZY)
FOR_TPCH_QUERY(TPCH_BENCH_REGISTER_GPU_LAZY)

#define TPCH_BENCH_DEFINE_HYBRID(name, prepFunction)       \
  BENCHMARK_DEFINE_F(TPCHHybrid, HYBRID_##name)            \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define TPCH_BENCH_REGISTER_HYBRID(name, prepFunction) \
  BENCHMARK_REGISTER_F(TPCHHybrid, HYBRID_##name)      \
      ->Args({100, 1})                                 \
      ->UseRealTime()                                  \
      ->Unit(benchmark::kMillisecond)                  \
      ->Iterations(5);

FOR_TPCH_QUERY(TPCH_BENCH_DEFINE_HYBRID)
FOR_TPCH_QUERY(TPCH_BENCH_REGISTER_HYBRID)

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  auto olap = proteus::from_cli::olap("Benchmark TPC-H", &argc, &argv);
  LOG(INFO) << "Finished initialization";
  LOG(INFO) << "Running in: " << std::filesystem::current_path() << " \n";

  ::benchmark::RunSpecifiedBenchmarks();
}
