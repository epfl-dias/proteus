/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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

#include <gflags/gflags.h>

#include <cli-flags.hpp>
#include <platform/topology/topology.hpp>
#include <stack/query.hpp>

#include "stack-fixture.hpp"

DECLARE_string(compression_type);

DEFINE_string(stack_dataset_path, "",
              "Optional Stack dataset directory override. When empty, the "
              "benchmark selects the dataset from --compression_type.");

namespace {
std::filesystem::path stackDatasetPathFromFlags() {
  if (!FLAGS_stack_dataset_path.empty()) {
    return FLAGS_stack_dataset_path;
  }
  LOG(FATAL) << "--stack_dataset_path is required for Stack benchmarks";
}

void pointStackInputAtSelectedDataset() {
  const auto dataset_path = stackDatasetPathFromFlags();
  CHECK(std::filesystem::exists(dataset_path))
      << "Stack dataset path does not exist: " << dataset_path;

  const auto inputs_dir = std::filesystem::path{"inputs"};
  const auto stack_link = inputs_dir / "stack";
  std::error_code ec;
  std::filesystem::create_directories(inputs_dir, ec);
  CHECK(!ec) << "Failed to create " << inputs_dir << ": " << ec.message();

  if (std::filesystem::exists(stack_link, ec) ||
      std::filesystem::is_symlink(stack_link, ec)) {
    CHECK(std::filesystem::is_symlink(stack_link, ec))
        << stack_link << " exists and is not a symlink";
    std::filesystem::remove(stack_link, ec);
    CHECK(!ec) << "Failed to remove " << stack_link << ": " << ec.message();
  }

  std::filesystem::create_directory_symlink(dataset_path, stack_link, ec);
  CHECK(!ec) << "Failed to point " << stack_link << " at " << dataset_path
             << ": " << ec.message();
  LOG(INFO) << "Stack input " << stack_link << " -> " << dataset_path;
}
}  // namespace

// Stack queries reported in the paper (see prepared-queries/stack).
#define FOR_STACK_QUERY(DO)              \
  DO(Q11, stack::Query::prepareQ11)      \
  DO(Q14, stack::Query::prepareQ14)

#define STACK_BENCH_DEFINE_GPU(name, prepFunction)         \
  BENCHMARK_DEFINE_F(StackGPUOnly, GPU_##name)             \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define STACK_BENCH_REGISTER_GPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(StackGPUOnly, GPU_##name)     \
      ->Args({1, 1})                                 \
      ->UseRealTime()                                \
      ->Unit(benchmark::kMillisecond)                \
      ->Iterations(10);

FOR_STACK_QUERY(STACK_BENCH_DEFINE_GPU)
FOR_STACK_QUERY(STACK_BENCH_REGISTER_GPU)

// IN-GPU RESIDENT: data preloaded GPU-resident by the scan; run one query per
// process (see StackInGPU note in stack-fixture.hpp).
#define STACK_BENCH_DEFINE_INGPU(name, prepFunction)       \
  BENCHMARK_DEFINE_F(StackInGPU, INGPU_##name)             \
  (benchmark::State & st) {                                \
    auto num_gpus = topology::getInstance().getGpuCount(); \
    if (num_gpus == 0) {                                   \
      st.SkipWithError("System has no GPUs");              \
    }                                                      \
    auto statement = prepFunction(*shaper.get());          \
    warmUp(statement);                                     \
    runBenchmark(statement, st);                           \
  }

#define STACK_BENCH_REGISTER_INGPU(name, prepFunction) \
  BENCHMARK_REGISTER_F(StackInGPU, INGPU_##name)       \
      ->Args({1, 1})                                   \
      ->UseRealTime()                                  \
      ->Unit(benchmark::kMillisecond)                  \
      ->Iterations(10);

FOR_STACK_QUERY(STACK_BENCH_DEFINE_INGPU)
FOR_STACK_QUERY(STACK_BENCH_REGISTER_INGPU)

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  auto olap = proteus::from_cli::olap("Benchmark Stack", &argc, &argv);
  pointStackInputAtSelectedDataset();
  LOG(INFO) << "Finished initialization";
  LOG(INFO) << "Running in: " << std::filesystem::current_path() << " \n";

  ::benchmark::RunSpecifiedBenchmarks();
}
