/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2025
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

#ifndef PROTEUS_SSB_SYNTHETIC_BENCHMARK_HPP
#define PROTEUS_SSB_SYNTHETIC_BENCHMARK_HPP

#include <query-shaping/nvme-shapers.hpp>

#include "ssb-synthetic-queries.hpp"
#include "util.hpp"

#include <gflags/gflags.h>
DECLARE_bool(decom_op);

struct SynthJoinBenchmarkArgs {
  int server_number;
  int num_drives = 4;
  Shaper shaper_type = Shaper::NVMECPU;
  int num_iterations = 5;
  CompressionType compression_type = CompressionType::NONE;
  int scale_factor;
  std::pair<std::function<decltype(synthetic_join_zero)>, std::string>
      prep_query_function = {synthetic_join_zero, "synthetic_join_zero"};
  SynthJoinQueryArgs query_args;
  //std::vector<int> lo_quantity_le_predicates = {1, 50}; should be a parameters but ok like this
  std::vector<int> lo_quantity_le_predicates = {25};
  std::string header() {
    return "query_name,num_drives,compression_type,scale_factor,server_number,"
           "shaper";
  }
};

std::ostream& operator<<(std::ostream& os, const SynthJoinBenchmarkArgs& args) {
  os << args.prep_query_function.second << "," << args.num_drives << ","
     << magic_enum::enum_name(args.compression_type) << "," << args.scale_factor
     << "," << args.server_number << ","
     << magic_enum::enum_name(args.shaper_type);
  return os;
}

std::string bench_synthetic_join_query(SynthJoinBenchmarkArgs args) {
  std::stringstream result_string;
  result_string << "time_ms,query_name,quantity_le,date,scan_memmove_dop_per_"
                   "gpu,scan_memmove_slack,"
                << args.header() << std::endl;

  const auto all_md_dirs =
      [compression_type = args.compression_type, sf = args.scale_factor,
       server_number = args.server_number, num_drives = args.num_drives]() {
        return get_input_dirs(sf, server_number, num_drives, compression_type);
      }();

  for (const auto& md_dirs : all_md_dirs) {
    /// important, because the relations are all the same from the point of
    /// view of the catalog we need to drop the catalog to ensure we use the
    /// right plugin instance for each configurations of md files
    CatalogParser::getInstance().clear();

    // Shaper is only used for the plan and to provide the SSB stats
    // no slacks or affinitizers are currently used from the shaper
    std::shared_ptr<proteus::GPUOnlyNVMeMorsel> shaper =
        std::make_shared<proteus::GPUOnlyNVMeMorsel>(
            md_dirs, "inputs/ssbm" + std::to_string(args.scale_factor), ssb::Query::getStats(args.scale_factor),
            true, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);

    for (const auto& pred : args.lo_quantity_le_predicates) {
      LOG(INFO) << "Running query with lo_quantity_le_predicate: " << pred;
      SynthJoinQueryArgs query_args = args.query_args;
      query_args.lo_quantity_le_predicate = pred;
      query_args.morph = shaper;
      auto prep_query = args.prep_query_function.first(query_args);
      auto query_name = args.prep_query_function.second;

      auto bench_res =
          benchmark_query(query_name, prep_query, args.num_iterations);
      for (auto& query_time : bench_res.per_query_times) {
        result_string << format_ms_1dp(query_time) << "," << bench_res.label
                      << ","
                      << pred << "," << get_current_date_str() << ","
                      << query_args.scan_memmove_dop_per_gpu << ","
                      << query_args.scan_memmove_slack << "," << args
                      << std::endl;
      }
      LOG(INFO) << bench_res.label
                << " average time: " << format_ms_1dp(bench_res.average_query_time);
    }
  }

  return result_string.str();
}

#endif  // PROTEUS_SSB_SYNTHETIC_BENCHMARK_HPP
