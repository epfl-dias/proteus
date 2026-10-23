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

#ifndef PROTEUS_TPCH_FIXTURE_HPP
#define PROTEUS_TPCH_FIXTURE_HPP

#include <benchmark/benchmark.h>

#include <cli-flags.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <platform/common/common.hpp>
#include <query-shaping/experimental-shapers.hpp>
#include <query-shaping/query-shaper.hpp>
#include <storage/storage-manager.hpp>
#include <tpch/query.hpp>

DECLARE_string(compression_type);

inline bool tpchUseCompressedShaper() { return FLAGS_compression_type != "NONE"; }

inline std::vector<std::string> tpchInputDirs(size_t SF) {
  return {"inputs/tpch" + std::to_string(SF)};
}

inline std::string tpchCatalogPath(size_t SF) {
  return "inputs/tpch" + std::to_string(SF);
}

class TPCHBaseFixture : public benchmark::Fixture {
 public:
  size_t SF;
  size_t warmup;
  std::shared_ptr<proteus::QueryShaper> shaper;
  std::map<std::string, std::function<double(proteus::InputPrefixQueryShaper&)>>
      stats;

 public:
  virtual void setLoaders(benchmark::State& state) = 0;
  virtual void setQueryShaper(benchmark::State& state) = 0;

  void SetUp(benchmark::State& state) override {
    SF = state.range(0);
    warmup = state.range(1);
    state.counters["scale_factor"] = SF;
    stats = tpch::Query::getStats(SF);
    setLoaders(state);
    setQueryShaper(state);
  }

  void warmUp(PreparedStatement& statement) {
    if (warmup) {
      for (size_t i = 0; i < warmup; i++) {
        time_block t("Warmup  " + std::to_string(i) + " :");
        statement.execute();
      }
    }
  }

  void runBenchmark(PreparedStatement& statement, benchmark::State& state) {
    LOG(INFO) << "state.iterations(): " << state.max_iterations;
    std::vector<std::vector<std::chrono::milliseconds>> pipelineTimes(
        state.max_iterations);
    size_t num_iterations = 0;
    for (auto _ : state) {
      statement.execute(pipelineTimes[num_iterations]);
      num_iterations++;
    }

    size_t num_pipelines = pipelineTimes[0].size();

    std::vector<int64_t> sum_of_pipeline_times(num_pipelines, 0);
    for (auto singleItPipelineTimes : pipelineTimes) {
      for (size_t j = 0; j < singleItPipelineTimes.size(); j++) {
        sum_of_pipeline_times.at(j) += singleItPipelineTimes.at(j).count();
      }
    }
    for (size_t j = 0; j < num_pipelines; j++) {
      state.counters["average_pipeline_" + std::to_string(j) + "_time_ms"] =
          sum_of_pipeline_times.at(j) / num_iterations;
    }
  }

  void TearDown(benchmark::State& state) override {
    auto& sm = StorageManager::getInstance();
    sm.unloadAll();
    sm.dropAllCustomLoaders();
    shaper.reset();
  }
};

class TPCHCPUOnly : public TPCHBaseFixture {
 public:
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    shaper = make_shaper<proteus::CPUOnlySingleServer>(SF, std::move(stats));
  }
};

class TPCHGPUOnly : public TPCHBaseFixture {
 public:
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    if (tpchUseCompressedShaper()) {
      shaper = std::make_shared<proteus::GPUOnlySingleServerCompressed>(
          tpchInputDirs(SF), tpchCatalogPath(SF), std::move(stats),
          /*allowMoves=*/false, /*scan_memmove_slack=*/4,
          /*scan_router_slack=*/4, /*slack=*/16,
          /*emit_compressed_blocks=*/true);
      return;
    }
    shaper = make_shaper<proteus::GPUOnlySingleServer>(SF, std::move(stats));
  }
};

class TPCHGPUOnlyLazy : public TPCHBaseFixture {
 public:
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    if (tpchUseCompressedShaper()) {
      shaper = std::make_shared<proteus::GPUOnlySingleServerCompressed>(
          tpchInputDirs(SF), tpchCatalogPath(SF), std::move(stats),
          /*allowMoves=*/false, /*scan_memmove_slack=*/4,
          /*scan_router_slack=*/4, /*slack=*/16,
          /*emit_compressed_blocks=*/true);
      return;
    }
    shaper =
        make_shaper<proteus::LazyGPUOnlySingleServer>(SF, std::move(stats));
  }
};

// IN-GPU RESIDENT (--in_gpu=true semantics): the scan preloads all blocks into
// GPU memory at plugin construction and emits GPU-resident pointers, so
// executions pay no NVMe->GPU streaming. Uses the preloading shaper
// unconditionally (NONE: uncompressed blocks resident; RFOR/FLBP: compressed
// blocks resident). Run one query per process (--benchmark_filter='INGPU_Qxx/')
// -- each benchmark preloads its own plugins and buffers are freed only at
// process exit, so chaining many INGPU benchmarks accumulates GPU memory.
class TPCHInGPU : public TPCHBaseFixture {
 public:
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    shaper = std::make_shared<proteus::GPUOnlySingleServerCompressed>(
        tpchInputDirs(SF), tpchCatalogPath(SF), std::move(stats),
        /*allowMoves=*/false, /*scan_memmove_slack=*/4,
        /*scan_router_slack=*/4, /*slack=*/16,
        /*emit_compressed_blocks=*/tpchUseCompressedShaper());
  }
};

class TPCHHybrid : public TPCHBaseFixture {
 public:
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    shaper = make_shaper<proteus::HybridSingleServer>(SF, std::move(stats));
  }
};

#endif  // PROTEUS_TPCH_FIXTURE_HPP
