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

#ifndef PROTEUS_SSB_FIXTURE_HPP
#define PROTEUS_SSB_FIXTURE_HPP

#include <benchmark/benchmark.h>

#include <cli-flags.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <platform/common/common.hpp>
#include <query-shaping/experimental-shapers.hpp>
#include <query-shaping/query-shaper.hpp>
#include <storage/storage-manager.hpp>

class SSBBaseFixture : public benchmark::Fixture {
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
    auto& sm = StorageManager::getInstance();
    stats = ssb::Query::getStats(SF);
    setLoaders(state);
    setQueryShaper(state);
  }

  /**
   * Not a part of the standard lifecycle, must be called explicitly by
   * benchmark functions
   */
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

class SSBCPUOnly : public SSBBaseFixture {
 public:
  //  default loader is to CPUs, so all data will start memory resident
  void setLoaders(benchmark::State& state) override {}

  void setQueryShaper(benchmark::State& state) override {
    auto* shaperPtr = new proteus::CPUOnlySingleServer{
        "inputs/ssbm" + std::to_string(SF) + "/", stats};
    shaper.reset(shaperPtr);
  }
};

class SSBGPUOnly : public SSBBaseFixture {
 public:
  //  default loader is to CPUs, so all data will start memory resident
  void setLoaders(benchmark::State& state) override {}
  void setQueryShaper(benchmark::State& state) override {
    auto* shaperPtr = new proteus::GPUOnlySingleServer{
        "inputs/ssbm" + std::to_string(SF) + "/", stats, true};
    shaper.reset(shaperPtr);
  }
};

class SSBHybrid : public SSBBaseFixture {
 public:
  //  default loader is to CPUs, so all data will start memory resident
  void setLoaders(benchmark::State& state) override {}
  void setQueryShaper(benchmark::State& state) override {
    auto* shaperPtr = new proteus::HybridSingleServer{
        "inputs/ssbm" + std::to_string(SF) + "/", stats, true};
    shaper.reset(shaperPtr);
  }
};

#endif  // PROTEUS_SSB_FIXTURE_HPP
