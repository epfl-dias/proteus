/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <cli-flags.hpp>
#include <codegen/expressions/expressionTypes.hpp>
#include <magic_enum.hpp>
#include <olap/operators/decompression-config.hpp>
#include <olap/operators/decompression-flags.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/util/glog.hpp>
#include <query-shaping/experimental-shapers.hpp>
#include <query-shaping/nvme-shapers.hpp>
#include <ssb/query.hpp>
#include <storage/storage-manager.hpp>

#include "../../../../core/olap/lib/util/caching.hpp"
#include "../../../../core/olap/lib/util/catalog.hpp"
#include "util.hpp"

DECLARE_string(compression_type);
DECLARE_bool(decom_op);
DECLARE_bool(decom_reg);
DECLARE_bool(precompute_rl);
DECLARE_bool(pre_decom);
DECLARE_bool(precomputed_offsets);
DECLARE_bool(use_metadata_in_block);
DECLARE_bool(in_gpu);
DECLARE_bool(one_kernel);
DECLARE_bool(emit_tuple_oid_bindings);
DECLARE_int32(tile_size);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);
DECLARE_bool(warm_up);
DECLARE_bool(print_res);
DECLARE_int32(scale_factor);
DECLARE_int32(server_number);
DECLARE_int32(num_iterations);
DECLARE_int32(num_drives);
DECLARE_int32(scan_router_slack);
DECLARE_int32(scan_memmove_slack);
DECLARE_string(result_file);

DEFINE_bool(warm_up, true,
    "Run one warm-up execution before benchmark iterations for each query.");

DEFINE_bool(print_res, false, "Print query results to stdout.");

DEFINE_int32(scale_factor, 100, "SSB scale factor");

DEFINE_int32(server_number, 49, "server number (DIAS internal)");

DEFINE_int32(num_iterations, 5,
             "Number of iterations to run each projectivity query");

DEFINE_int32(num_drives, 1, "Number of drives used by the benchmark");

DEFINE_int32(scan_router_slack, 4, "Router slack for scan pipelines");

DEFINE_int32(scan_memmove_slack, 4, "MemMove slack for scan pipelines");

DEFINE_string(result_file, "",
              "[optional] output file for results [default: stdout]");

namespace {

constexpr std::array<const char*, 9> kProjectedColumns = {
    "lo_orderdate",  "lo_quantity",    "lo_extendedprice",
    "lo_discount",   "lo_partkey",     "lo_suppkey",
    "lo_revenue",    "lo_custkey",     "lo_supplycost",
};

struct BenchGpuMemoryOccupancyBytes {
  size_t compressed_dataset_bytes{0};
  size_t aux_bytes{0};

  [[nodiscard]] size_t total_bytes() const {
    return compressed_dataset_bytes + aux_bytes;
  }
};

[[nodiscard]] BenchGpuMemoryOccupancyBytes
collect_gpu_memory_occupancy_bytes() {
  BenchGpuMemoryOccupancyBytes footprint;
  Catalog::getInstance().forEachPlugin([&](Plugin* pg) {
    if (auto* nvme_pg = dynamic_cast<NvmePlugin*>(pg)) {
      auto nvme_footprint = nvme_pg->getGpuMemoryFootprint();
      footprint.compressed_dataset_bytes +=
          nvme_footprint.compressed_dataset_bytes;
      footprint.aux_bytes += nvme_footprint.aux_bytes;
    }
  });
  return footprint;
}

[[nodiscard]] std::vector<std::filesystem::path> ssb_catalog_dirs_for_sf(
    int sf) {
  return {
      std::filesystem::path("inputs") / "parser",
      std::filesystem::path("inputs") / "plans",
      std::filesystem::path("inputs") / ("ssbm" + std::to_string(sf)),
  };
}

void clear_catalog_parser_for_sf(int sf) {
  CatalogParser::getInstance().clearAndParse(ssb_catalog_dirs_for_sf(sf));
}

void clear_runtime_state_for_sf(int sf) {
  Catalog::getInstance().clear();
  CachingService::getInstance().clear();
  clear_catalog_parser_for_sf(sf);
}

[[nodiscard]] std::string get_decompression_mode_label(
    CompressionType compression_type) {
  if (compression_type == CompressionType::NONE) {
    return "NO_decom";
  }

  if (FLAGS_pre_decom) {
    return "predecom";
  }
  if (FLAGS_decom_reg) {
    return "decom_in_reg";
  }
  if (FLAGS_decom_op) {
    return "decom_in_smem";
  }
  return "decom_in_gmem";
}


[[nodiscard]] std::pair<int, int> get_runtime_decode_shape(
    CompressionType compression_type) {
  if (compression_type == CompressionType::NONE) {
    constexpr auto ints_per_block =
        BlockManager::block_size / sizeof(int32_t);
    constexpr int thread_per_block = 1024;
    constexpr int items_per_thread = ints_per_block / thread_per_block;
    return {thread_per_block, items_per_thread};
  }

  if (compression_type == CompressionType::RFOR && FLAGS_pre_decom) {
    const auto rfor_shape = proteus::rfor::predecompress::resolveRuntimeShape(
        FLAGS_tile_size, FLAGS_threads_per_block, FLAGS_items_per_thread);
    CHECK_GT(rfor_shape.tile_size, 0)
        << "Invalid RFOR pre-decompress runtime shape";
    return {rfor_shape.block_threads, rfor_shape.items_per_thread};
  }

  if (compression_type == CompressionType::FLBP && FLAGS_pre_decom) {
    const auto flbp_shape =
        proteus::flbp::predecompress::resolveRuntimeShape(
            FLAGS_threads_per_block, FLAGS_items_per_thread);
    CHECK_GT(flbp_shape.tile_size, 0)
        << "Invalid FLBP pre-decompress runtime shape";
    return {flbp_shape.block_threads, flbp_shape.items_per_thread};
  }

  return {FLAGS_threads_per_block, FLAGS_items_per_thread};
}

[[nodiscard]] std::vector<std::string> get_projected_columns(size_t count) {
  CHECK_GE(count, 1);
  CHECK_LE(count, kProjectedColumns.size());

  std::vector<std::string> columns;
  columns.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    columns.emplace_back(kProjectedColumns[i]);
  }
  return columns;
}

[[nodiscard]] std::vector<std::string> get_sum_aliases(size_t count) {
  CHECK_GE(count, 1);
  CHECK_LE(count, kProjectedColumns.size());

  std::vector<std::string> aliases;
  aliases.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    aliases.emplace_back("sum_" + std::to_string(i));
  }
  return aliases;
}

[[nodiscard]] std::string join_columns(const std::vector<std::string>& columns) {
  std::ostringstream oss;
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i > 0) {
      oss << '+';
    }
    oss << columns[i];
  }
  return oss.str();
}

[[nodiscard]] std::vector<Monoid> make_sum_monoids(size_t count) {
  return std::vector<Monoid>(count, SUM);
}

template <typename TInputArg>
[[nodiscard]] std::vector<expression_t> build_sum_inputs(
    const TInputArg& arg, const std::vector<std::string>& source_columns,
    const std::vector<std::string>& output_aliases, const std::string& rel_name) {
  CHECK_EQ(source_columns.size(), output_aliases.size());

  std::vector<expression_t> result;
  result.reserve(source_columns.size());

  for (size_t i = 0; i < source_columns.size(); ++i) {
    result.emplace_back(arg[source_columns[i]].as(rel_name, output_aliases[i]));
  }

  return result;
}

template <typename TInputArg>
[[nodiscard]] std::vector<expression_t> build_value_refs(
    const TInputArg& arg, const std::vector<std::string>& columns) {
  std::vector<expression_t> result;
  result.reserve(columns.size());

  for (const auto& column : columns) {
    result.emplace_back(arg[column]);
  }

  return result;
}

[[nodiscard]] RelBuilder scan_lineorder_prefix(
    proteus::InputPrefixQueryShaper& morph, size_t projected_column_count) {
  switch (projected_column_count) {
    case 1:
      return morph.scan("lineorder", {kProjectedColumns[0]});
    case 2:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1]});
    case 3:
      return morph.scan(
          "lineorder",
          {kProjectedColumns[0], kProjectedColumns[1], kProjectedColumns[2]});
    case 4:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3]});
    case 5:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3],
                         kProjectedColumns[4]});
    case 6:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3],
                         kProjectedColumns[4], kProjectedColumns[5]});
    case 7:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3],
                         kProjectedColumns[4], kProjectedColumns[5],
                         kProjectedColumns[6]});
    case 8:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3],
                         kProjectedColumns[4], kProjectedColumns[5],
                         kProjectedColumns[6], kProjectedColumns[7]});
    case 9:
      return morph.scan("lineorder",
                        {kProjectedColumns[0], kProjectedColumns[1],
                         kProjectedColumns[2], kProjectedColumns[3],
                         kProjectedColumns[4], kProjectedColumns[5],
                         kProjectedColumns[6], kProjectedColumns[7],
                         kProjectedColumns[8]});
    default:
      LOG(FATAL) << "Unsupported projected column count: "
                 << projected_column_count;
  }
}

[[nodiscard]] PreparedStatement build_projectivity_query(
    proteus::InputPrefixQueryShaper& morph, size_t projected_column_count) {
  const auto columns = get_projected_columns(projected_column_count);
  const auto sum_aliases = get_sum_aliases(projected_column_count);
  const auto query_name =
      "projectivity_" + std::to_string(projected_column_count);
  const auto local_rel = query_name + "_local";
  const auto global_rel = query_name + "_global";

  morph.setQueryName(query_name);

  auto scan = scan_lineorder_prefix(morph, projected_column_count);
  auto partial = morph.distribute_probe(scan)
                     .unpack()
                     .filter(
                         [first_column =
                              std::string{kProjectedColumns[0]}](const auto& arg)
                             -> expression_t {
                           return expressions::hint(
                               ge(arg[first_column],
                                  std::numeric_limits<int32_t>::min()),
                               expressions::Selectivity(1.0));
                         })
                     .project([columns,
                               sum_aliases,
                               local_rel](const auto& arg)
                                  -> std::vector<expression_t> {
                       return build_sum_inputs(arg, columns, sum_aliases, local_rel);
                     })
                     .reduce(
                         [sum_aliases](const auto& arg)
                             -> std::vector<expression_t> {
                           return build_value_refs(arg, sum_aliases);
                         },
                         make_sum_monoids(columns.size()));

  return morph.collect(partial)
      .reduce(
          [sum_aliases, global_rel](const auto& arg) -> std::vector<expression_t> {
            return build_sum_inputs(arg, sum_aliases, sum_aliases, global_rel);
          },
          make_sum_monoids(columns.size()))
      .print(pg{"pm-csv"})
      .prepare();
}

[[nodiscard]] std::unique_ptr<proteus::InputPrefixQueryShaper> make_shaper(
    int sf, Shaper shaper_type, CompressionType compression_type,
    const std::vector<std::string>& md_dirs) {
  if (shaper_type == Shaper::NVMEGPU) {
    return std::make_unique<proteus::GPUOnlyNVMeMorsel>(
        md_dirs, "inputs/ssbm" + std::to_string(sf), ssb::Query::getStats(sf),
        true, FLAGS_scan_memmove_slack, FLAGS_scan_router_slack, 16);
  }

  CHECK(shaper_type == Shaper::INGPU)
      << "Unsupported shaper type: " << magic_enum::enum_name(shaper_type);
  if (compression_type == CompressionType::NONE) {
    auto shaper = std::make_unique<proteus::GPUOnlySingleServer>(
        "inputs/ssbm" + std::to_string(sf) + "/", ssb::Query::getStats(sf),
        false, 16);

    auto& sm = StorageManager::getInstance();
    sm.setDefaultLoader(
        [](StorageManager&, const std::string& file_name, size_t type_size) {
          return FileRecord::loadToGpus(file_name, type_size);
        },
        true);

    return shaper;
  }

  constexpr bool emit_compressed_blocks = true;
  return std::make_unique<proteus::GPUOnlySingleServerCompressed>(
      md_dirs, "inputs/ssbm" + std::to_string(sf), ssb::Query::getStats(sf),
      false, FLAGS_scan_memmove_slack, FLAGS_scan_router_slack, 16,
      emit_compressed_blocks);
}

void validate_runtime_flags() {
  LOG_ASSERT(!(FLAGS_one_kernel && !FLAGS_in_gpu))
      << "--one_kernel requires --in_gpu=true";
  LOG_ASSERT(FLAGS_tile_size == 512 || FLAGS_tile_size == 1024)
      << "--tile_size must be 512 or 1024 (got " << FLAGS_tile_size << ")";

  const bool has_threads_override = FLAGS_threads_per_block > 0;
  const bool has_items_override = FLAGS_items_per_thread > 0;
  LOG_ASSERT(has_threads_override == has_items_override)
      << "--threads_per_block and --items_per_thread must be set together "
      << "(or both left at 0)";

  if (FLAGS_compression_type == "RFOR") {
    const bool uses_cb2t_decode =
        !FLAGS_pre_decom && (FLAGS_in_gpu || FLAGS_decom_op);
    LOG_ASSERT(FLAGS_threads_per_block >= 0)
        << "--threads_per_block must be >= 0";
    LOG_ASSERT(FLAGS_items_per_thread >= 0)
        << "--items_per_thread must be >= 0";
    if (uses_cb2t_decode) {
      const int cb2t_block_threads =
          FLAGS_threads_per_block > 0 ? FLAGS_threads_per_block
                                      : proteus::rfor::cb2t::kBlockThreads;
      const int cb2t_items_per_thread =
          FLAGS_items_per_thread > 0 ? FLAGS_items_per_thread
                                     : proteus::rfor::cb2t::kItemsPerThread;
      LOG_ASSERT(cb2t_block_threads * cb2t_items_per_thread == FLAGS_tile_size)
          << "RFOR CB2T requires --threads_per_block * --items_per_thread == "
             "--tile_size ("
          << cb2t_block_threads << " * " << cb2t_items_per_thread << " != "
          << FLAGS_tile_size << ")";
    } else if (FLAGS_pre_decom) {
      const auto rfor_shape =
          proteus::rfor::predecompress::resolveRuntimeShape(
              FLAGS_tile_size, FLAGS_threads_per_block,
              FLAGS_items_per_thread);
      LOG_ASSERT(rfor_shape.tile_size > 0)
          << "RFOR --pre_decom supports only 64x8, 128x4, or 256x2 for "
          << "--tile_size=512, and 512x2, 256x4, or 128x8 for --tile_size=1024. "
          << "Requested " << FLAGS_threads_per_block << "x"
          << FLAGS_items_per_thread << " for tile_size=" << FLAGS_tile_size;
    }
  } else if (FLAGS_compression_type == "FLBP") {
    LOG_ASSERT(FLAGS_threads_per_block >= 0)
        << "--threads_per_block must be >= 0";
    LOG_ASSERT(FLAGS_items_per_thread >= 0)
        << "--items_per_thread must be >= 0";
    if (FLAGS_pre_decom) {
      const auto flbp_shape =
          proteus::flbp::predecompress::resolveRuntimeShape(
              FLAGS_threads_per_block, FLAGS_items_per_thread);
      LOG_ASSERT(flbp_shape.tile_size > 0)
          << "FLBP --pre_decom requires positive --threads_per_block and "
          << "--items_per_thread values whose product divides the fixed "
          << "1024-value FastLanes vector. Requested "
          << FLAGS_threads_per_block << "x" << FLAGS_items_per_thread;
      if (FLAGS_tile_size != flbp_shape.tile_size) {
        LOG(INFO)
            << "FLBP --pre_decom ignores the RFOR-style dataset --tile_size "
            << "flag and uses the runtime logical subtile resolved from "
            << "--threads_per_block * --items_per_thread ("
            << flbp_shape.block_threads << " * "
            << flbp_shape.items_per_thread << " = " << flbp_shape.tile_size
            << "). Overriding --tile_size=" << FLAGS_tile_size << " to "
            << flbp_shape.tile_size << ".";
        FLAGS_tile_size = flbp_shape.tile_size;
      }
    }
  }

  if (FLAGS_in_gpu) {
    gflags::CommandLineFlagInfo cpu_buffers_flag;
    const bool has_cpu_buffers_flag =
        gflags::GetCommandLineFlagInfo("cpu_buffers", &cpu_buffers_flag);
    const bool cpu_buffers_is_default =
        has_cpu_buffers_flag && cpu_buffers_flag.is_default;

    if (cpu_buffers_is_default && FLAGS_cpu_buffers > 0.05) {
      LOG(INFO) << "--in_gpu set: lowering default --cpu_buffers from "
                << FLAGS_cpu_buffers
                << " to 0.05 to reduce startup pinned-memory pressure.";
      FLAGS_cpu_buffers = 0.05;
    }
  }

  LOG_ASSERT(!(FLAGS_decom_op && FLAGS_compression_type == "NONE"))
      << "--decom_op requires a compressed input type. "
         "Use --compression_type=RFOR/LZ4/CASCADED/GDEFLATE or disable "
         "--decom_op.";
  LOG_ASSERT(!(FLAGS_precompute_rl && !FLAGS_decom_reg))
      << "--precompute_rl is only valid with --decom_reg=true.";
  LOG_ASSERT(!(FLAGS_precompute_rl && FLAGS_compression_type != "RFOR"))
      << "--precompute_rl currently supports only --compression_type=RFOR.";
  LOG_ASSERT(!(FLAGS_pre_decom && FLAGS_decom_op))
      << "--pre_decom and --decom_op are mutually exclusive.";
  LOG_ASSERT(!(FLAGS_pre_decom && FLAGS_compression_type != "RFOR" &&
               FLAGS_compression_type != "FLBP"))
      << "--pre_decom currently supports only --compression_type=RFOR/FLBP.";
  LOG_ASSERT(!(FLAGS_pre_decom && !FLAGS_in_gpu))
      << "--pre_decom requires --in_gpu.";
  LOG_ASSERT(!(FLAGS_in_gpu && FLAGS_compression_type == "RFOR" &&
               !FLAGS_use_metadata_in_block))
      << "--in_gpu + --compression_type=RFOR requires "
         "--use_metadata_in_block=true.";
  if (FLAGS_decom_op && FLAGS_compression_type == "RFOR") {
    if (FLAGS_in_gpu) {
      LOG(INFO) << "--in_gpu + RFOR: using metadata-in-block decode path.";
    } else if (FLAGS_use_metadata_in_block) {
      LOG(INFO) << "--decom_op + --use_metadata_in_block: using deferred "
                   "CompressedBlockToTuples RFOR decode.";
    } else {
      LOG(INFO) << "--decom_op with --use_metadata_in_block=false: keeping "
                   "legacy MemMove-side RFOR decompression.";
    }
  }
  if (FLAGS_precompute_rl && FLAGS_decom_reg) {
    LOG(INFO)
        << "--precompute_rl + --decom_reg: expecting RFOR datasets with "
           "precomputed cumulative RL and using no-prefix-sum register decode.";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("projectivity_sensitivity");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  if (FLAGS_in_gpu && FLAGS_one_kernel) {
    LOG(INFO) << "projectivity_sensitivity forcing --one_kernel=false to "
                 "avoid the unstable CpuToGpu batch-launch path during "
                 "repeated projectivity sweeps.";
    FLAGS_one_kernel = false;
  }
  if (FLAGS_emit_tuple_oid_bindings) {
    LOG(INFO) << "projectivity_sensitivity forcing "
                 "--emit_tuple_oid_bindings=false because this benchmark "
                 "has no downstream join/pack operators that require tuple "
                 "OID bindings.";
    FLAGS_emit_tuple_oid_bindings = false;
  }
  validate_runtime_flags();

  const auto shaper_type = FLAGS_in_gpu ? Shaper::INGPU : Shaper::NVMEGPU;
  if (shaper_type == Shaper::INGPU) {
    LOG_ASSERT(FLAGS_scale_factor == 100)
        << "INGPU shaper currently only supports SSB SF100, since bigger sf "
           "dont fit GPU memory";
  }

  auto compression_type_opt =
      magic_enum::enum_cast<CompressionType>(FLAGS_compression_type);
  CHECK(compression_type_opt.has_value())
      << "Unsupported compression_type: " << FLAGS_compression_type;
  const auto compression_type = compression_type_opt.value();

  auto ctx = proteus::from_cli::olap();

  std::optional<std::ofstream> out = std::nullopt;
  if (!FLAGS_result_file.empty()) {
    out.emplace(FLAGS_result_file);
    CHECK(out->is_open()) << "Could not open result file " << FLAGS_result_file;
  }

  std::stringstream result;
  result << "query,"
         << "projected_column_count,"
         << "projected_columns,"
         << "iteration,"
         << "compression_type,"
         << "num_drives,"
         << "diascld,"
         << "time_ms,"
         << "date,"
         << "scan_router_slack,"
         << "scan_memmove_slack,"
         << "shaper,"
         << "decompression_type,"
         << "threads_per_block,"
         << "items_per_thread,"
         << "gpu_dataset_compressed_bytes,"
         << "gpu_aux_bytes,"
         << "gpu_total_bytes" << std::endl;

  const auto all_md_dirs = get_input_dirs(
      FLAGS_scale_factor, FLAGS_server_number, FLAGS_num_drives,
      compression_type);
  const auto decomp_mode = get_decompression_mode_label(compression_type);
  const auto [threads_per_block, items_per_thread] =
      get_runtime_decode_shape(compression_type);

  for (const auto& md_dirs : all_md_dirs) {
    check_vector_paths(md_dirs);

    for (size_t projected_column_count = 2;
         projected_column_count <= kProjectedColumns.size();
         ++projected_column_count) {
      clear_catalog_parser_for_sf(FLAGS_scale_factor);

      auto shaper = make_shaper(FLAGS_scale_factor, shaper_type,
                                compression_type, md_dirs);
      auto query =
          build_projectivity_query(*shaper, projected_column_count);
      auto bench_res = benchmark_query(
          "projectivity_" + std::to_string(projected_column_count), query,
          FLAGS_num_iterations);
      const auto gpu_occupancy = collect_gpu_memory_occupancy_bytes();
      const auto columns = get_projected_columns(projected_column_count);
      const auto columns_label = join_columns(columns);

      for (size_t iteration = 0; iteration < bench_res.per_query_times.size();
           ++iteration) {
        result << bench_res.label << ","
               << projected_column_count << ","
               << columns_label << ","
               << iteration << ","
               << magic_enum::enum_name(compression_type) << ","
               << md_dirs.size() << ","
               << FLAGS_server_number << ","
               << format_ms_1dp(bench_res.per_query_times[iteration]) << ","
               << get_current_date_str() << ","
               << FLAGS_scan_router_slack << ","
               << FLAGS_scan_memmove_slack << ","
               << magic_enum::enum_name(shaper_type) << ","
               << decomp_mode << ","
               << threads_per_block << ","
               << items_per_thread << ","
               << gpu_occupancy.compressed_dataset_bytes << ","
               << gpu_occupancy.aux_bytes << ","
               << gpu_occupancy.total_bytes() << std::endl;
      }

      clear_runtime_state_for_sf(FLAGS_scale_factor);
    }
  }

  std::cout << result.str();
  if (out.has_value()) {
    *out << result.str();
  }

  auto& sm = StorageManager::getInstance();
  sm.unloadAll();
  return 0;
}
