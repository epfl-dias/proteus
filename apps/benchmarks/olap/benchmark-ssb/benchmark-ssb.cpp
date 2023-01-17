/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

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
#include <olap-perf-util/ssb-aliases.hpp>
#include <platform/topology/topology.hpp>
#include <ssb/query.hpp>

#include "ssb-fixture.hpp"

// CPU only
#define SSB_BENCH_DEFINE_CPU(name, prepFunction)  \
  BENCHMARK_DEFINE_F(SSBCPUOnly, CPU_##name)      \
  (benchmark::State & st) {                       \
    auto statement = prepFunction(*shaper.get()); \
    warmUp(statement);                            \
    runBenchmark(statement, st);                  \
  }

#define SSB_BENCH_REGISTER_CPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(SSBCPUOnly, CPU_##name)     \
      ->Args({100, 1})                             \
      ->UseRealTime()                              \
      ->Unit(benchmark::kMillisecond)              \
      ->Iterations(5);                             \
  BENCHMARK_REGISTER_F(SSBCPUOnly, CPU_##name)     \
      ->Args({1000, 1})                            \
      ->UseRealTime()                              \
      ->Unit(benchmark::kMillisecond)              \
      ->Iterations(5);

// We separate define and register, so we can apply arguments
// (X Macro pattern because, to my knowledge, the RegisterBenchmark function
// does not work with fixtures)
FOR_SSB_QUERY(SSB_BENCH_DEFINE_CPU)
FOR_SSB_QUERY(SSB_BENCH_REGISTER_CPU)

// GPU only
#define SSB_BENCH_DEFINE_GPU(name, prepFunction)           \
  BENCHMARK_DEFINE_F(SSBGPUOnly, GPU_##name)               \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define SSB_BENCH_REGISTER_GPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(SSBGPUOnly, GPU_##name)     \
      ->Args({100, 1})                             \
      ->UseRealTime()                              \
      ->Unit(benchmark::kMillisecond)              \
      ->Iterations(5);                             \
  BENCHMARK_REGISTER_F(SSBGPUOnly, GPU_##name)     \
      ->Args({1000, 1})                            \
      ->UseRealTime()                              \
      ->Unit(benchmark::kMillisecond)              \
      ->Iterations(5);

FOR_SSB_QUERY(SSB_BENCH_DEFINE_GPU)
FOR_SSB_QUERY(SSB_BENCH_REGISTER_GPU)

// Hybrid CPU-GPU
#define SSB_BENCH_DEFINE_HYBRID(name, prepFunction)        \
  BENCHMARK_DEFINE_F(SSBHybrid, HYBRID_##name)             \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define SSB_BENCH_REGISTER_HYBRID(name, prepFunction) \
  BENCHMARK_REGISTER_F(SSBHybrid, HYBRID_##name)      \
      ->Args({100, 1})                                \
      ->UseRealTime()                                 \
      ->Unit(benchmark::kMillisecond)                 \
      ->Iterations(5);                                \
  BENCHMARK_REGISTER_F(SSBHybrid, HYBRID_##name)      \
      ->Args({1000, 1})                               \
      ->UseRealTime()                                 \
      ->Unit(benchmark::kMillisecond)                 \
      ->Iterations(5);

FOR_SSB_QUERY(SSB_BENCH_DEFINE_HYBRID)
FOR_SSB_QUERY(SSB_BENCH_REGISTER_HYBRID)

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  auto olap = proteus::from_cli::olap("Benchmark SSB", &argc, &argv);
  LOG(INFO) << "Finished initialization";
  LOG(INFO) << "Running in: " << std::filesystem::current_path() << " \n";

  ::benchmark::RunSpecifiedBenchmarks();
}
