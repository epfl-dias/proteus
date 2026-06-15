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

#include "util.hpp"
#include <iomanip>
#include <numeric>
#include <sstream>
#include <platform/util/profiling.hpp>

#include "base/commandlineflags.h"


DECLARE_bool(decom_op);
DECLARE_bool(decom_reg);
DECLARE_bool(precompute_rl);
DECLARE_bool(in_gpu);
DECLARE_bool(use_metadata_in_block);
DECLARE_bool(warm_up);
DECLARE_bool(print_res);
DECLARE_int32(tile_size);

QueryBenchResult benchmark_query(const std::string& label, PreparedStatement& statement,
                size_t num_iterations) {
  std::vector<std::vector<std::chrono::milliseconds>> pipeline_times(
      num_iterations);
  std::vector<QueryBenchResult::MillisecondsFp> per_query_times(num_iterations);

  if (FLAGS_warm_up)
  {
    LOG(INFO) << "Warming up for " << label;
    statement.execute();
    LOG(INFO) << "end warmup";
  }
    profiling::ProfileRegionType pr_type = profiling::ProfileRegionType(label);
    // std::cout << "profiling::ProfileRegionType label " << label << std::endl;
    //  Print exact time
    //  std::cout << "Start execution of " << label << " at "
    //            << get_current_date_str_precise() << std::endl;
  //std::this_thread::sleep_for(std::chrono::seconds(10));

  for (int i = 0; i < num_iterations; i++) {
    LOG(INFO) << "begin_run_iteration " << i << "/" << num_iterations << " for "
              << label;
    // PreparedStatement::execute() already controls profiler start/stop.
    // Calling resume/pause here as well nests cudaProfilerStart/Stop and can
    // confuse external profilers (notably ncu replay mode).
    profiling::ProfileRegion pr(pr_type);
    profiling::ProfileRegion_ProcessRange pr1(pr_type);
    // this profiling is done in the execute function?
    const auto query_begin = std::chrono::steady_clock::now();
    auto res = statement.execute(pipeline_times[i]);
    const auto query_end = std::chrono::steady_clock::now();
    per_query_times[i] = std::chrono::duration_cast<QueryBenchResult::MillisecondsFp>(
        query_end - query_begin);
    if (FLAGS_print_res)
    {
      std::cout << "<QUERY RESULT BEGIN>\n";
      std::cout << res << std::endl;
      std::cout << "\n<END QUERY RESULT>\n";
    }
  }
  //  std::cout << "End execution of " << label << " at "
  //            << get_current_date_str_precise() << std::endl;

  const size_t num_pipelines = pipeline_times[0].size();
  std::vector<QueryBenchResult::MillisecondsFp> sum_of_pipeline_times(
      num_pipelines, QueryBenchResult::MillisecondsFp{0});
  for (const auto &per_it_pipeline_times : pipeline_times) {
    for (size_t j = 0; j < per_it_pipeline_times.size(); j++) {
      sum_of_pipeline_times.at(j) +=
          std::chrono::duration_cast<QueryBenchResult::MillisecondsFp>(
              per_it_pipeline_times.at(j));
    }
  }

  std::vector<QueryBenchResult::MillisecondsFp> mean_pipeline_times(
      num_pipelines, QueryBenchResult::MillisecondsFp{0});
  for (size_t j = 0; j < num_pipelines; j++) {
    mean_pipeline_times[j] =
        sum_of_pipeline_times.at(j) / static_cast<double>(num_iterations);
  }

  const auto total_query_time =
      std::accumulate(per_query_times.begin(), per_query_times.end(),
                      QueryBenchResult::MillisecondsFp{0});

  QueryBenchResult result;
  result.per_query_times = per_query_times;
  result.label = label;
  result.pipeline_times = mean_pipeline_times;
  result.average_query_time =
      total_query_time / static_cast<double>(num_iterations);
  return result;
}

std::string format_ms_1dp(QueryBenchResult::MillisecondsFp ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << ms.count();
  return oss.str();
}

void check_vector_paths(const std::vector<std::string> &paths) {
  {
    for (const auto &path : paths) {
      CHECK(std::filesystem::exists(path)) << "path not found " << path;
      CHECK(not std::filesystem::is_empty(path)) << "path is empty " << path;
    }
  }
}

std::vector<std::vector<std::string>> get_input_dir_lz4(int sf,
                                                        int server_number,
                                                        int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  if (sf == 1000) {
    if (server_number == 49) {
      if (sf == 100) {
        LOG(FATAL) << "not sf100 on diascld49";
      }

      if (sf == 1000) {
        std::vector<std::string> drives;
        switch (num_drives) {
          case 1: {
            drives = {
                "/nvme20/nicholso/data/compressed_ssbm1000"  // node 5
            };
            break;
          }
          case 2: {
            drives = {
                "/nvme18/nicholso/data/compressed_ssbm1000_0_2",  // node 4
                "/nvme22/nicholso/data/compressed_ssbm1000_1_2"   // node 5
            };
            break;
          }

          case 4: {
            drives = {
                "/nvme0/nicholso/data/compressed_ssbm1000_0_4",   // node 4
                "/nvme13/nicholso/data/compressed_ssbm1000_1_4",  // node 7
                "/nvme28/nicholso/data/compressed_ssbm1000_2_4",  // node 6
                "/nvme22/nicholso/data/compressed_ssbm1000_3_4"   // node 5
            };
            break;
          }
          case 8: {
            drives = {"/nvme0/nicholso/data/compressed_ssbm1000_0_8", // node 6
                "/nvme13/nicholso/data/compressed_ssbm1000_1_8", // node 4
                "/nvme14/nicholso/data/compressed_ssbm1000_2_8", // node 7
                "/nvme16/nicholso/data/compressed_ssbm1000_3_8", // node 4
                "/nvme28/nicholso/data/compressed_ssbm1000_4_8", // node 6
                "/nvme20/nicholso/data/compressed_ssbm1000_5_8", // node 5
                "/nvme21/nicholso/data/compressed_ssbm1000_6_8", // node 5
                "/nvme24/nicholso/data/compressed_ssbm1000_7_8" // node 7
            };
            break;
          }

            //        case 12: {
            //        drives = {
            //              "/nvme0/nicholso/data/compressed_ssbm1000_0_12",
            //              "/nvme7/nicholso/data/compressed_ssbm1000_1_12",
            //              "/nvme13/nicholso/data/compressed_ssbm1000_2_12",
            //              "/nvme14/nicholso/data/compressed_ssbm1000_3_12",
            //              "/nvme6/nicholso/data/compressed_ssbm1000_4_12",
            //              "/nvme9/nicholso/data/compressed_ssbm1000_5_12",
            //              "/nvme28/nicholso/data/compressed_ssbm1000_6_12",
            //              "/nvme29/nicholso/data/compressed_ssbm1000_7_12",
            //              "/nvme20/nicholso/data/compressed_ssbm1000_8_12",
            //              "/nvme21/nicholso/data/compressed_ssbm1000_9_12",
            //              "/nvme24/nicholso/data/compressed_ssbm1000_10_12",
            //              "/nvme25/nicholso/data/compressed_ssbm1000_11_12"};
            //        break;
            //      }
          default:
            LOG(FATAL) << "num_drives not supported: " << num_drives;
        }

        check_vector_paths(drives);
        return {drives};
      }
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dir_cascaded(int sf,
                                                             int server_number,
                                                             int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  if (server_number == 49) {
    if (sf == 100) {
      LOG(FATAL) << "not sf100 on diascld49";
    }

    if (sf == 1000) {
      std::vector<std::string> drives;

      switch (num_drives) {
        case 1: {
          drives = {
              "/nvme22/nicholso/data/cascaded_compressed_ssbm1000_0_1" // node 5

          };
          break;
        }
        case 2: {
          drives = {
              "/nvme21/nicholso/data/cascaded_compressed_ssbm1000_0_2", // node 5
              "/nvme18/nicholso/data/cascaded_compressed_ssbm1000_1_2", // node 4
          };
          break;
        }

        case 4: {
          drives = {
              "/nvme16/nicholso/data/cascaded_compressed_ssbm1000_0_4", // node 4
              "/nvme23/nicholso/data/cascaded_compressed_ssbm1000_1_4", // node 5
              "/nvme28/nicholso/data/cascaded_compressed_ssbm1000_2_4", // node 6
              "/nvme26/nicholso/data/cascaded_compressed_ssbm1000_3_4" // node 7
          };
          break;
        }

          //        case 6: {
          //          drives = {
          //              "/nvme10/chasiali/data/compressed_ssbm1000_0_8",
          //              "/nvme5/chasiali/data/compressed_ssbm1000_1_8",
          //              "/nvme7/chasiali/data/compressed_ssbm1000_2_8",
          //              "/nvme3/chasiali/data/compressed_ssbm1000_3_8",
          //              "/nvme6/chasiali/data/compressed_ssbm1000_4_8",
          //              "/nvme13/chasiali/data/compressed_ssbm1000_5_8",
          //              //          "/nvme4/chasiali/data/compressed_ssbm1000_5_8",
          //              "/nvme9/chasiali/data/compressed_ssbm1000_6_8",
          //              "/nvme8/chasiali/data/compressed_ssbm1000_7_8"};
          //          break;
          //        }

        case 8: {
          drives = {
              "/nvme16/nicholso/data/cascaded_compressed_ssbm1000_0_8", // node 4 nvme 16
              "/nvme22/nicholso/data/cascaded_compressed_ssbm1000_1_8", // node 5 nvme 22
              "/nvme26/nicholso/data/cascaded_compressed_ssbm1000_2_8", // node 7 nvme 26
              "/nvme29/nicholso/data/cascaded_compressed_ssbm1000_3_8", // node 6 nvme 29
              "/nvme18/nicholso/data/cascaded_compressed_ssbm1000_4_8", // node 4 nvme 18
              "/nvme21/nicholso/data/cascaded_compressed_ssbm1000_5_8", // node 5 nvme 21
              "/nvme28/nicholso/data/cascaded_compressed_ssbm1000_6_8", // node 6 nvme 28
              "/nvme27/nicholso/data/cascaded_compressed_ssbm1000_7_8" // node 7 nvme 27
          };
          break;
        }
          //        case 12: {
          //          drives = {
          //              "/nvme10/chasiali/data/compressed_ssbm1000_0_12",
          //              "/nvme5/chasiali/data/compressed_ssbm1000_1_12",
          //              "/nvme7/chasiali/data/compressed_ssbm1000_2_12",
          //              "/nvme3/chasiali/data/compressed_ssbm1000_3_12",
          //              "/nvme6/chasiali/data/compressed_ssbm1000_4_12",
          //              "/nvme2/chasiali/data/compressed_ssbm1000_5_12",
          //              //        "/nvme4/chasiali/data/compressed_ssbm1000_5_12",
          //              "/nvme9/chasiali/data/compressed_ssbm1000_6_12",
          //              "/nvme8/chasiali/data/compressed_ssbm1000_7_12",
          //              "/nvme11/chasiali/data/compressed_ssbm1000_8_12",
          //              "/nvme13/chasiali/data/compressed_ssbm1000_9_12",
          //              "/nvme12/chasiali/data/compressed_ssbm1000_10_12",
          //              "/nvme1/chasiali/data/compressed_ssbm1000_11_12"};
          //          break;
          //        }
        default:
          LOG(FATAL) << "num_drives not supported: " << num_drives;
      }

      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dir_cascaded_64k(int sf,
                                                             int server_number,
                                                             int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  if (server_number == 49) {
    std::vector<std::string> drives;
    if (sf == 100) {
      // LOG(FATAL) << "not sf100 on diascld49";
      switch (num_drives) {
        case 1: {
          drives = {
            "/nvme31/duric/data/compressed_cascaded_64k_ssbm100"
          };
          break;
        }
      }
      check_vector_paths(drives);
      return {drives};
    }

    if (sf == 1000) {
      std::vector<std::string> drives;

      switch (num_drives) {
        case 1: {
          drives = {
              "/nvme21/nicholso/data/compressed_casc_64k_ssbm1000" // node 5
          };
          break;
        }
        case 2: {
          drives = {
              "/nvme18/nicholso/data/compressed_casc_64k_ssbm1000_0_2", // node 5
              "/nvme23/nicholso/data/compressed_casc_64k_ssbm1000_1_2", // node 4
          };
          break;
        }
        case 4: {
          // LOG(INFO) << "USING 4 DRIVES";
          drives = {
              "/nvme18/nicholso/data/compressed_casc_64k_ssbm1000_0_4", // node 4
              "/nvme23/nicholso/data/compressed_casc_64k_ssbm1000_1_4", // node 5
              "/nvme28/nicholso/data/compressed_casc_64k_ssbm1000_2_4", // node 6
              "/nvme25/nicholso/data/compressed_casc_64k_ssbm1000_3_4" // node 7
          };
          break;
        }

        case 8: {
          drives = {
              "/nvme16/nicholso/data/compressed_casc_64k_ssbm1000_0_8", // node 4 nvme 16
              "/nvme22/nicholso/data/compressed_casc_64k_ssbm1000_1_8", // node 5 nvme 22
              "/nvme26/nicholso/data/compressed_casc_64k_ssbm1000_2_8", // node 7 nvme 26
              "/nvme29/nicholso/data/compressed_casc_64k_ssbm1000_3_8", // node 6 nvme 29
              "/nvme18/nicholso/data/compressed_casc_64k_ssbm1000_4_8", // node 4 nvme 18
              "/nvme21/nicholso/data/compressed_casc_64k_ssbm1000_5_8", // node 5 nvme 21
              "/nvme28/nicholso/data/compressed_casc_64k_ssbm1000_6_8", // node 6 nvme 28
              "/nvme27/nicholso/data/compressed_casc_64k_ssbm1000_7_8" // node 7 nvme 27
          };

          break;
        }
        default:
          LOG(FATAL) << "num_drives not supported: " << num_drives;
      }

      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dir_lz4_64k(int sf,
                                                                 int server_number,
                                                                 int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  if (server_number == 49) {
    if (sf == 100) {
      LOG(FATAL) << "not sf100 on diascld49";
    }

    if (sf == 1000) {
      std::vector<std::string> drives;

      switch (num_drives) {
        case 1: {
          drives = {
              "/nvme21/nicholso/data/compressed_lz4_64k_ssbm1000" // node 5
          };
          break;
        }
        case 2: {
          drives = {
              "/nvme18/nicholso/data/compressed_lz4_64k_ssbm1000_0_2", // node 5
              "/nvme23/nicholso/data/compressed_lz4_64k_ssbm1000_1_2", // node 4
          };
          break;
        }
        case 4: {
          drives = {
              "/nvme18/nicholso/data/compressed_lz4_64k_ssbm1000_0_4", // node 4
              "/nvme23/nicholso/data/compressed_lz4_64k_ssbm1000_1_4", // node 5
              "/nvme28/nicholso/data/compressed_lz4_64k_ssbm1000_2_4", // node 6
              "/nvme25/nicholso/data/compressed_lz4_64k_ssbm1000_3_4" // node 7
          };
          break;
        }

        case 8: {
          drives = {
              "/nvme16/nicholso/data/compressed_lz4_64k_ssbm1000_0_8", // node 4 nvme 16
              "/nvme22/nicholso/data/compressed_lz4_64k_ssbm1000_1_8", // node 5 nvme 22
              "/nvme26/nicholso/data/compressed_lz4_64k_ssbm1000_2_8", // node 7 nvme 26
              "/nvme29/nicholso/data/compressed_lz4_64k_ssbm1000_3_8", // node 6 nvme 29
              "/nvme18/nicholso/data/compressed_lz4_64k_ssbm1000_4_8", // node 4 nvme 18
              "/nvme21/nicholso/data/compressed_lz4_64k_ssbm1000_5_8", // node 5 nvme 21
              "/nvme28/nicholso/data/compressed_lz4_64k_ssbm1000_6_8", // node 6 nvme 28
              "/nvme27/nicholso/data/compressed_lz4_64k_ssbm1000_7_8" // node 7 nvme 27
          };

          break;
        }
        default:
          LOG(FATAL) << "num_drives not supported: " << num_drives;
      }

      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}
std::vector<std::vector<std::string>> get_input_dir_gdeflate_64k(int sf,
                                                            int server_number,
                                                            int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  if (server_number == 49) {
    if (sf == 100) {
      LOG(FATAL) << "not sf100 on diascld49";
    }

    if (sf == 1000) {
      std::vector<std::string> drives;

      switch (num_drives) {
        case 1: {
          drives = {
              "/nvme22/nicholso/data/compressed_gdeflate_64k_ssbm1000" // node 5
          };
          break;
        }
        case 2: {
          drives = {
              "/nvme18/nicholso/data/compressed_gdeflate_64k_ssbm1000_0_2", // node 5
              "/nvme23/nicholso/data/compressed_gdeflate_64k_ssbm1000_1_2", // node 4
          };
          break;
        }
        case 4: {
          drives = {
              "/nvme18/nicholso/data/compressed_gdeflate_64k_ssbm1000_0_4", // node 4
              "/nvme23/nicholso/data/compressed_gdeflate_64k_ssbm1000_1_4", // node 5
              "/nvme28/nicholso/data/compressed_gdeflate_64k_ssbm1000_2_4", // node 6
              "/nvme25/nicholso/data/compressed_gdeflate_64k_ssbm1000_3_4" // node 7
          };
          break;
        }

        case 8: {
          drives = {
              "/nvme16/nicholso/data/compressed_gdeflate_64k_ssbm1000_0_8", // node 4 nvme 16
              "/nvme22/nicholso/data/compressed_gdeflate_64k_ssbm1000_1_8", // node 5 nvme 22
              "/nvme26/nicholso/data/compressed_gdeflate_64k_ssbm1000_2_8", // node 7 nvme 26
              "/nvme29/nicholso/data/compressed_gdeflate_64k_ssbm1000_3_8", // node 6 nvme 29
              "/nvme18/nicholso/data/compressed_gdeflate_64k_ssbm1000_4_8", // node 4 nvme 18
              "/nvme21/nicholso/data/compressed_gdeflate_64k_ssbm1000_5_8", // node 5 nvme 21
              "/nvme28/nicholso/data/compressed_gdeflate_64k_ssbm1000_6_8", // node 6 nvme 28
              "/nvme27/nicholso/data/compressed_gdeflate_64k_ssbm1000_7_8" // node 7 nvme 27
          };

          break;
        }
        default:
          LOG(FATAL) << "num_drives not supported: " << num_drives;
      }

      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dir_rfor(int sf,
                                                         int server_number,
                                                         int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";
  CHECK(FLAGS_tile_size == 512 || FLAGS_tile_size == 1024)
      << "--tile_size must be 512 or 1024 (got " << FLAGS_tile_size << ")";
  CHECK(!(FLAGS_in_gpu && !FLAGS_use_metadata_in_block))
      << "--in_gpu + RFOR requires --use_metadata_in_block=true";
  // In in-GPU RFOR runs we always use metadata-in-block datasets.
  // The runtime flag is reserved for out-of-GPU routing.
  const bool need_inblock_metadata_dirs =
      FLAGS_in_gpu || FLAGS_use_metadata_in_block;
  const bool need_precomputed_cumulative_rl =
      FLAGS_decom_reg && FLAGS_precompute_rl;
  if (need_precomputed_cumulative_rl) {
    LOG(INFO) << "RFOR input selection: using precomputed cumulative-RL datasets "
                 "for --decom_reg path.";
  }

  std::vector<std::string> drives;
  if (server_number == 49) {
    if (sf == 100) {

      switch (num_drives) {
        case 1: {
            if (need_inblock_metadata_dirs)
            {
              // Select metadata-in-block dataset by RFOR layout tile size.
              if (FLAGS_tile_size == 512)
              {
                drives = {need_precomputed_cumulative_rl
                              ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg_512"
                              : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op_512"};
              }
              else if (FLAGS_tile_size == 1024)
              {
                drives = {need_precomputed_cumulative_rl
                              ? "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_reg"
                              : "/nvme31/ssbm100/compressed_rfor_ssbm100_decom_op"};
              }
              else
              {
                LOG(FATAL) << "Unsupported RFOR dataset tile_size="
                           << FLAGS_tile_size
                           << " for SF100 metadata-in-block dataset selection";
              }
            }
            else
            {
              if (FLAGS_tile_size == 1024)
              {
                drives = {"/nvme31/ssbm100/compressed_rfor_ssbm100"};
              }
              else
              {
                LOG(FATAL) << "No SF100 non-metadata RFOR dataset configured for tile_size="
                           << FLAGS_tile_size
                           << ". Available path expects tile_size=1024: "
                           << "/nvme31/ssbm100/compressed_rfor_ssbm100";
              }
            }
          break;
        }

      default:
        LOG(FATAL) << "num_drives not supported for RFOR sf100: " << num_drives;
    }
    check_vector_paths(drives);
    return {drives};
    }

    if (sf == 1000) {
      switch (num_drives) {
        case 1: {
            if (need_inblock_metadata_dirs)
            {
              drives = {"/nvme25/boffa/data/compressed_rfor_ssbm1000_decom_op"};
            }
            else
            {
              drives = {
                "/nvme25/duric/data/compressed_rfor_ssbm1000"
              };
            }
          break;
        }

        case 2: {
          drives = {
            "/nvme16/duric/data/compressed_rfor_ssbm1000_0_2", // node 4
            "/nvme28/duric/data/compressed_rfor_ssbm1000_1_2", // node 5
        };
          break;
        }

        case 4: {
          drives = {
            "/nvme16/duric/data/compressed_rfor_ssbm1000_0_4", // node 4
            "/nvme23/duric/data/compressed_rfor_ssbm1000_1_4", // node 5
            "/nvme28/duric/data/compressed_rfor_ssbm1000_2_4", // node 6
            "/nvme26/duric/data/compressed_rfor_ssbm1000_3_4" // node 7
        };
          break;
        }

        case 8: {
          if (need_inblock_metadata_dirs)
          {
            drives = {
              "/nvme16/boffa/data/compressed_rfor_ssbm1000_decom_op_0_8", // node 4 nvme 16
              "/nvme22/boffa/data/compressed_rfor_ssbm1000_decom_op_1_8", // node 5 nvme 22
              "/nvme26/boffa/data/compressed_rfor_ssbm1000_decom_op_2_8", // node 7 nvme 26
              "/nvme29/boffa/data/compressed_rfor_ssbm1000_decom_op_3_8", // node 6 nvme 29
              "/nvme18/boffa/data/compressed_rfor_ssbm1000_decom_op_4_8", // node 4 nvme 18
              "/nvme21/boffa/data/compressed_rfor_ssbm1000_decom_op_5_8", // node 5 nvme 21
              "/nvme28/boffa/data/compressed_rfor_ssbm1000_decom_op_6_8", // node 6 nvme 28
              "/nvme27/boffa/data/compressed_rfor_ssbm1000_decom_op_7_8" // node 7 nvme 27
              };
          }
          else
          {
            drives = {
              "/nvme16/boffa/data/compressed_rfor_ssbm1000_0_8", // node 4 nvme 16
              "/nvme22/boffa/data/compressed_rfor_ssbm1000_1_8", // node 5 nvme 22
              "/nvme26/boffa/data/compressed_rfor_ssbm1000_2_8", // node 7 nvme 26
              "/nvme29/boffa/data/compressed_rfor_ssbm1000_3_8", // node 6 nvme 29
              "/nvme18/boffa/data/compressed_rfor_ssbm1000_4_8", // node 4 nvme 18
              "/nvme21/boffa/data/compressed_rfor_ssbm1000_5_8", // node 5 nvme 21
              "/nvme28/boffa/data/compressed_rfor_ssbm1000_6_8", // node 6 nvme 28
              "/nvme27/boffa/data/compressed_rfor_ssbm1000_7_8" // node 7 nvme 27
          };
          }

          break;
        }

        default:
          LOG(FATAL) << "num_drives not supported for RFOR sf1000: " << num_drives;
      }
      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "RFOR not set up for server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dir_flbp(int sf,
                                                         int server_number,
                                                         int num_drives) {
  (void)sf;
  (void)server_number;
  (void)num_drives;
  std::vector<std::string> drives = {
      "/nvme31/ssbm100/compressed_flbp_ssbm100_decom_op"};
  check_vector_paths(drives);
  return {drives};
}

std::vector<std::vector<std::string>> get_input_dir_uncompressed(
    int sf, int server_number, int num_drives) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";

  std::vector<std::string> drives;

  if (server_number == 49) {
    if (sf == 100) {
      switch (num_drives) {
        case 1: {
          drives = {
            "/nvme31/ssbm100/uncompressed"
        };
          break;
        }
        default:
          LOG(FATAL) << "num_drives not supported for Uncompressed sf100: " << num_drives;
      }
      check_vector_paths(drives);
      return {drives};
    }
    if (sf == 1000) {
      switch (num_drives) {
        case 1: {
          drives = {"/nvme16/nicholso/data/sbm1000"};  // node 4;
          //drives = {"/scratch/data/ssbm1000"}; //  <-- ???
          break;
        }
        case 2: {
          drives = {
            "/nvme18/nicholso/data/sbm1000_0_2", // node 4
            "/nvme23/nicholso/data/sbm1000_1_2" // node 5
        };
          break;
        }

        case 4: {
          drives = {
            "/nvme25/nicholso/data/sbm1000_0_4", // node 7
            "/nvme18/nicholso/data/sbm1000_1_4", // node 4
            "/nvme28/nicholso/data/sbm1000_2_4", // node 6
            "/nvme22/nicholso/data/sbm1000_3_4" // node 5
        };
          break;
        }

        case 8: {
          drives = {
            "/nvme16/nicholso/data/sbm1000_0_8", // node 4
            "/nvme19/nicholso/data/sbm1000_1_8", // node 4
            "/nvme28/nicholso/data/sbm1000_2_8", // node 6
            "/nvme27/nicholso/data/sbm1000_3_8", // node 7
            "/nvme28/nicholso/data/sbm1000_4_8", // node 6
            "/nvme20/nicholso/data/sbm1000_5_8", // node 5
            "/nvme21/nicholso/data/sbm1000_6_8", // node 5
            "/nvme24/nicholso/data/sbm1000_7_8" // node 7
        };
          break;
        }

          //        case 10: {
          //          drives = {"/nvme0/nicholso/data/sbm1000_0_10",
          //                    "/nvme7/nicholso/data/sbm1000_1_10",
          //                    "/nvme13/nicholso/data/sbm1000_2_10",
          //                    "/nvme14/nicholso/data/sbm1000_3_10",
          //                    "/nvme6/nicholso/data/sbm1000_4_10",
          //                    "/nvme28/nicholso/data/sbm1000_5_10",
          //                    "/nvme29/nicholso/data/sbm1000_6_10",
          //                    "/nvme20/nicholso/data/sbm1000_7_10",
          //                    "/nvme21/nicholso/data/sbm1000_8_10",
          //                    "/nvme24/nicholso/data/sbm1000_9_10"};
          //          break;
          //        }

        case 12: {
          drives = {
            "/nvme16/nicholso/data/sbm1000_0_12", // node 4
            "/nvme22/nicholso/data/sbm1000_1_12", // node 5
            "/nvme29/nicholso/data/sbm1000_2_12", // node 6
            "/nvme24/nicholso/data/sbm1000_3_12", // node 7
            "/nvme18/nicholso/data/sbm1000_4_12", // node 4
            "/nvme23/nicholso/data/sbm1000_5_12", // node 5
            "/nvme30/nicholso/data/sbm1000_6_12", // node 6
            "/nvme25/nicholso/data/sbm1000_7_12", // node 7
            "/nvme19/nicholso/data/sbm1000_8_12", // node 4
            "/nvme28/nicholso/data/sbm1000_9_12", // node 6
            "/nvme26/nicholso/data/sbm1000_10_12", // node 7
            "/nvme27/nicholso/data/sbm1000_11_12" // node 7
        };
          break;
        }
        default:
          LOG(FATAL) << "num_drives not supported: " << num_drives;
      }

      check_vector_paths(drives);
      return {drives};
    }
  }
  LOG(FATAL) << "not set up for this server: " << server_number;
}

std::vector<std::vector<std::string>> get_input_dirs(
    const int sf, const int server_number, int num_drives,
    CompressionType compression_type) {
  CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";
  CHECK_EQ(server_number, 49) << "currently only setup for diascld49";


  switch (compression_type) {
    case CompressionType::NONE:
      LOG(INFO) << "------------------------------NONE COMPRESSION TYPE------------------------------";
      return get_input_dir_uncompressed(sf, server_number, num_drives);
    case CompressionType::CASCADED:
      return get_input_dir_cascaded_64k(sf, server_number, num_drives);
    case CompressionType::LZ4:
      return get_input_dir_lz4_64k(sf, server_number, num_drives);
    case CompressionType::GDEFLATE:
      return get_input_dir_gdeflate_64k(sf, server_number, num_drives);
    case CompressionType::RFOR:
      LOG(INFO) << "------------------------------RFOR COMPRESSION TYPE------------------------------";
      return get_input_dir_rfor(sf, server_number, num_drives);
    case CompressionType::FLBP:
      LOG(INFO) << "------------------------------FLBP COMPRESSION TYPE------------------------------";
      return get_input_dir_flbp(sf, server_number, num_drives);
    default:
      LOG(FATAL) << "compression type not implemented"
                 << magic_enum::enum_name(compression_type);
  }
}

std::string get_current_date_str() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm *now_tm = std::localtime(&now_time_t);
  char buffer[50];
  std::strftime(buffer, sizeof(buffer), "%Y %B %e %H:%M:%S", now_tm);
  return {buffer};
}

std::string get_current_date_str_precise() {
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_tm = *std::localtime(&now_time_t);

  // Get nanoseconds by using a duration since the last second
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()) %
            1'000'000'000;

  // Format the time including nanoseconds
  std::ostringstream oss;
  oss << std::put_time(&now_tm, "%Y %B %e %H:%M:%S") << '.' << std::setw(9)
      << std::setfill('0') << ns.count();

  return oss.str();
}

PreparedStatement small_scan(proteus::QueryShaper &morph,
                             const std::string &lo_column) {
  morph.setQueryName("small_scan");

  auto scan = morph.scan("lineorder", {lo_column});
  auto gpu_pip =
      morph.distribute_probe(scan)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(lt(arg[lo_column], 1),
                                     expressions::Selectivity(0.000001));
          })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_column]};
              },
              {SUM});
  return morph.collect(gpu_pip)
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[lo_column]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}

PreparedStatement scan_two_columns(proteus::QueryShaper &morph,
                                   const std::string &lo_col1,
                                   const std::string &lo_col2) {
  morph.setQueryName("two_col_scan");

  auto scan = morph.scan("lineorder", {lo_col1, lo_col2});
  auto gpu_pip =
      morph.distribute_probe(scan)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(lt(arg[lo_col1], 1),
                                     expressions::Selectivity(0.000001));
          })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_col1]};
              },
              {SUM});
  return morph.collect(gpu_pip)
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[lo_col1]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}

PreparedStatement scan_six_columns(proteus::QueryShaper &morph,
                                   const std::string &lo_col1,
                                   const std::string &lo_col2,
                                   const std::string &lo_col3,
                                   const std::string &lo_col4,
                                   const std::string &lo_col5,
                                   const std::string &lo_col6) {
  morph.setQueryName("two_col_scan");

  auto scan = morph.scan(
      "lineorder", {lo_col1, lo_col2, lo_col3, lo_col4, lo_col5, lo_col6});
  auto gpu_pip =
      morph.distribute_probe(scan)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(gt(arg[lo_col1], 1),
                                     expressions::Selectivity(0.000001));
          })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_col1]};
              },
              {SUM});
  return morph.collect(gpu_pip)
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[lo_col1]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
