/*
Proteus -- High-performance query processing on heterogeneous hardware.

Copyright (c) 2024
Data Intensive Applications and Systems Laboratory (DIAS)
École Polytechnique Fédérale de Lausanne

All Rights Reserved.

Permission to use, copy, modify and distribute this software and
          its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
        the software, derivative works or modified versions, and any
                                           portions thereof, and that both
notices appear in supporting documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
        DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
            RESULTING FROM THE USE OF THIS SOFTWARE.
                */
#include <chrono>
#include <cli-flags.hpp>
#include <codegen/expressions/expressionTypes.hpp>
#include <fstream>
#include <magic_enum.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/profiling.hpp>
#include <query-shaping/experimental-shapers.hpp>
#include <query-shaping/nvme-shapers.hpp>
#include <query-shaping/query-shaper.hpp>
#include <ssb/query.hpp>
#include <thread>
#include <vector>

#include "util.hpp"

struct QueryBenchResult {
  std::vector<std::chrono::milliseconds> pipeline_times;
  std::chrono::milliseconds average_query_time;
  std::string label;
};

QueryBenchResult benchmark_query(const std::string& label,
                                 PreparedStatement& statement,
                                 size_t num_iterations) {
  std::vector<std::vector<std::chrono::milliseconds>> pipeline_times(
      num_iterations);
  // warmup
  // statement.execute();
  profiling::ProfileRegionType pr_type = profiling::ProfileRegionType(label);
  // std::cout << "profiling::ProfileRegionType label " << label << std::endl;
  //  Print exact time
  //  std::cout << "Start execution of " << label << " at "
  //            << get_current_date_str_precise() << std::endl;

  for (int i = 0; i < num_iterations; i++) {
    // profiling::resume();
    profiling::ProfileRegion pr(pr_type);
    // this profiling is done in the execute function?
    auto res = statement.execute(pipeline_times[i]);
  }
  //  std::cout << "End execution of " << label << " at "
  //            << get_current_date_str_precise() << std::endl;

  // profiling::pause();

  const size_t num_pipelines = pipeline_times[0].size();
  std::vector<std::chrono::milliseconds> sum_of_pipeline_times(num_pipelines);
  for (auto per_it_pipeline_times : pipeline_times) {
    for (size_t j = 0; j < per_it_pipeline_times.size(); j++) {
      sum_of_pipeline_times.at(j) += per_it_pipeline_times.at(j);
    }
  }

  std::vector<std::chrono::milliseconds> mean_pipeline_times(num_pipelines);
  for (size_t j = 0; j < num_pipelines; j++) {
    mean_pipeline_times[j] =
        std::chrono::milliseconds(sum_of_pipeline_times.at(j) / num_iterations);
  }

  QueryBenchResult result;
  result.label = label;
  result.pipeline_times = mean_pipeline_times;
  result.average_query_time = std::chrono::milliseconds(
      std::accumulate(mean_pipeline_times.begin(), mean_pipeline_times.end(),
                      std::chrono::milliseconds(0)));
  return result;
}

std::string bench_ssb_nvme_vary_bw_new(
    int sf, int server_number, Shaper shaper_type = Shaper::NVMEGPU,
    int num_iterations = 5, int scan_router_slack = 2,
    int scan_memmove_slack = 4, bool compressed = false, int query_number = 11,
    int num_drives = 1) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";
  LOG(INFO) << " is compressed " << compressed;
  std::stringstream result_string;
  result_string << "query,"
                << "is_compressed,"
                << "num_drives,"
                << "diascld,"
                << "time_ms,"
                << "date,"
                << "scan_router_slack,"
                << "scan_memmove_slack,"
                << "shaper" << std::endl;

  const auto all_md_dirs = get_input_dir(sf, server_number, num_drives);

  std::vector<std::pair<decltype(&ssb::Query::prepare11), std::string>> queries;

  if (query_number == 11) {
    queries.push_back({ssb::Query::prepare11, "ssb_Q1.1"});
  }

  if (query_number == 12) {
    queries.push_back({ssb::Query::prepare12, "ssb_Q1.2"});
  }

  if (query_number == 13) {
    queries.push_back({ssb::Query::prepare13, "ssb_Q1.3"});
  }

  if (query_number == 21) {
    queries.push_back({ssb::Query::prepare21, "ssb_Q2.1"});
  }

  if (query_number == 22) {
    queries.push_back({ssb::Query::prepare22, "ssb_Q2.2"});
  }

  if (query_number == 23) {
    queries.push_back({ssb::Query::prepare23, "ssb_Q2.3"});
  }

  if (query_number == 31) {
    queries.push_back({ssb::Query::prepare31, "ssb_Q3.1"});
  }

  if (query_number == 32) {
    queries.push_back({ssb::Query::prepare32, "ssb_Q3.2"});
  }

  if (query_number == 33) {
    queries.push_back({ssb::Query::prepare33, "ssb_Q3.3"});
  }

  if (query_number == 34) {
    queries.push_back({ssb::Query::prepare34, "ssb_Q3.4"});
  }

  if (query_number == 41) {
    queries.push_back({ssb::Query::prepare41, "ssb_Q4.1"});
  }

  if (query_number == 42) {
    queries.push_back({ssb::Query::prepare42, "ssb_Q4.2"});
  }

  if (query_number == 43) {
    queries.push_back({ssb::Query::prepare43, "ssb_Q4.3"});
  }

  for (const auto& md_dirs : all_md_dirs) {
    for (auto [query_prep_func, query_name] : queries) {
      /// important, because the relations are all the same from the point of
      /// view of the catalog we need to drop the catalog to ensure we use the
      /// right plugin instance for each configurations of md files

      CatalogParser::getInstance().clear();
      proteus::InputPrefixQueryShaper* shaper;

      // it is in memory NOW!
      switch (shaper_type) {
        case Shaper::MEMORYCPU:
          shaper = new proteus::CPUOnlySingleServer(
              "/inputs/ssbm" + std::to_string(sf) + "/",
              ssb::Query::getStats(sf), true, 16);
          break;
        case Shaper::MEMORYGPU:
          shaper = new proteus::GPUOnlySingleServer(
              "inputs/ssbm" + std::to_string(sf) + "/",
              ssb::Query::getStats(sf), true, 16);
          break;
        case Shaper::MEMORYGPU_LAZY:
          shaper = new proteus::LazyGPUOnlySingleServer(
              "inputs/ssbm" + std::to_string(sf) + "/",
              ssb::Query::getStats(sf), true, 16);
          break;
        default:
          LOG(FATAL) << "not set up for this server: " << server_number;
      }

      auto& sm = StorageManager::getInstance();
      sm.setDefaultLoader(
          [](StorageManager& stManager, const std::string& fileName,
             size_t typeSize) {
            return FileRecord::loadToGpus(fileName, typeSize);
          },
          true);

      auto prep_query = query_prep_func(*shaper);

      auto bench_res = benchmark_query(query_name, prep_query, num_iterations);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      result_string << bench_res.label << ","
                    << (compressed ? "true," : "false,") << md_dirs.size()
                    << "," << server_number << ","
                    << bench_res.average_query_time.count() << ","
                    << get_current_date_str() << "," << scan_router_slack << ","
                    << scan_memmove_slack << ","
                    << magic_enum::enum_name(shaper_type) << std::endl;
    }
  }
  return result_string.str();
}

DECLARE_bool(bench_varybw_gpu_ssb_inmemory);
DEFINE_bool(bench_varybw_gpu_ssb_inmemory, true,
            "In memory execution GPU only vary number of NVMes used with "
            "uncompressed data using "
            "--scale_factor and varying the number of drives used");

DECLARE_bool(lazy);
DEFINE_bool(lazy, false, "Use lazy loading (GPU execution only)");

DECLARE_int32(scale_factor);
DEFINE_int32(scale_factor, 100, "SSB scale factor");

DECLARE_int32(server_number);
DEFINE_int32(server_number, 49, "server number (DIAS internal)");

DECLARE_int32(num_iterations);
DEFINE_int32(num_iterations, 5, "Number of types to run each query");

DECLARE_int32(query);
DEFINE_int32(query, 11, "SSB query to execute (11-43)");

DECLARE_int32(num_drives);
DEFINE_int32(num_drives, 1,
             "num drives used as bandwidth query to execute (11-43)");

DECLARE_string(result_file);
DEFINE_string(result_file, "",
              "[optional] output file for results [default: stdout]");

int main(int argc, char* argv[]) {
  auto ctx = proteus::from_cli::olap("adm-play", &argc, &argv);

  std::stringstream ss;
  std::optional<std::ofstream> out = std::nullopt;

  // std::cout << "FLAGS_query: " << FLAGS_query << std::endl;
  // std::cout << "FLAGS_num_drives: " << FLAGS_num_drives << std::endl;

  if (!FLAGS_result_file.empty()) {
    out = std::ofstream(FLAGS_result_file);
    CHECK(out->is_open()) << "Could not open result file " << FLAGS_result_file;
  }

  if (FLAGS_bench_varybw_gpu_ssb_inmemory) {
    LOG(INFO) << "running FLAGS_bench_varybw_gpu_ssb_inmemory";

    if (!FLAGS_lazy) {
      auto res = bench_ssb_nvme_vary_bw_new(
          FLAGS_scale_factor, FLAGS_server_number, Shaper::MEMORYGPU,
          FLAGS_num_iterations, 4, 4, false, FLAGS_query, FLAGS_num_drives);
      ss << res;
      ss << std::endl;
      if (out.has_value()) {
        *out << res << std::endl;
      }
    } else {
      auto res = bench_ssb_nvme_vary_bw_new(
          FLAGS_scale_factor, FLAGS_server_number, Shaper::MEMORYGPU_LAZY,
          FLAGS_num_iterations, 4, 4, false, FLAGS_query, FLAGS_num_drives);
      ss << res;
      ss << std::endl;
      if (out.has_value()) {
        *out << res << std::endl;
      }
    }
  }

  std::cout << ss.str();
  auto& sm = StorageManager::getInstance();
  sm.unloadAll();
  return 0;
}
