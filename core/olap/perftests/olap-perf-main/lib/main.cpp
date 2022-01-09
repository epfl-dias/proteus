/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <benchmark/benchmark.h>

#include <cli-flags.hpp>
#include <platform/util/glog.hpp>
#include <storage/storage-manager.hpp>

extern bool FLAGS_benchmark_counters_tabular;
extern ::fLS::clstring FLAGS_benchmark_out;
extern ::fLS::clstring FLAGS_benchmark_out_format;

int main(int argc, char** argv) {
  gflags::AllowCommandLineReparsing();
  auto ctx = proteus::from_cli::olap("Some proteus perftests", &argc, &argv);

  FLAGS_benchmark_counters_tabular = true;
  FLAGS_benchmark_out = "perf.json";
  FLAGS_benchmark_out_format = "json";
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  benchmark::RunSpecifiedBenchmarks();

  auto& sm = StorageManager::getInstance();
  sm.unloadAll();

  return 0;
}
