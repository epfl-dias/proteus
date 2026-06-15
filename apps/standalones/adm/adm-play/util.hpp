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
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef PROTEUS_ADM_BENCH_UTIL_HPP
#define PROTEUS_ADM_BENCH_UTIL_HPP
#include <chrono>
#include <filesystem>
#include <olap/plan/prepared-statement.hpp>
#include <platform/util/glog.hpp>
#include <query-shaping/query-shaper.hpp>
#include <string>
#include <vector>
#include <magic_enum.hpp>

struct QueryBenchResult {
  using MillisecondsFp = std::chrono::duration<double, std::milli>;
  std::vector<MillisecondsFp> pipeline_times;
  std::vector<MillisecondsFp> per_query_times;
  MillisecondsFp average_query_time{};
  std::string label;
};


QueryBenchResult benchmark_query(const std::string& label,
                                 PreparedStatement& statement,
                                 size_t num_iterations);

std::string format_ms_1dp(QueryBenchResult::MillisecondsFp ms);

void check_vector_paths(const std::vector<std::string> &paths);

enum class Shaper { NVMECPU, NVMEGPU, INGPU };

enum class CompressionType { NONE, CASCADED, LZ4, GDEFLATE, RFOR, FLBP };


std::vector<std::vector<std::string>> get_input_dir_lz4(int sf,
                                                        int server_number,
                                                        int num_drives);

std::vector<std::vector<std::string>> get_input_dir_cascaded(int sf,
                                                             int server_number,
                                                             int num_drives);

std::vector<std::vector<std::string>> get_input_dir_lz4_64k(int sf,
                                                        int server_number,
                                                        int num_drives);

std::vector<std::vector<std::string>> get_input_dir_cascaded_64k(int sf,
                                                             int server_number,
                                                             int num_drives);

std::vector<std::vector<std::string>> get_input_dir_rfor(int sf,
                                                         int server_number,
                                                         int num_drives);

std::vector<std::vector<std::string>> get_input_dir_flbp(int sf,
                                                         int server_number,
                                                         int num_drives);

std::vector<std::vector<std::string>> get_input_dir_uncompressed(
    int sf, int server_number, int num_drives);

std::vector<std::vector<std::string>> get_input_dirs(
    const int sf, const int server_number, int num_drives,
    CompressionType compression_type);

std::string get_current_date_str();

std::string get_current_date_str_precise();

PreparedStatement small_scan(proteus::QueryShaper &morph,
                             const std::string &lo_column);

PreparedStatement scan_two_columns(proteus::QueryShaper &morph,
                                   const std::string &lo_col1,
                                   const std::string &lo_col2);

PreparedStatement scan_six_columns(proteus::QueryShaper &morph,
                                   const std::string &lo_col1,
                                   const std::string &lo_col2,
                                   const std::string &lo_col3,
                                   const std::string &lo_col4,
                                   const std::string &lo_col5,
                                   const std::string &lo_col6);

#endif  // PROTEUS_ADM_BENCH_UTIL_HPP
