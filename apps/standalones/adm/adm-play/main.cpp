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
#include <chrono>
#include <cli-flags.hpp>
#include <codegen/expressions/expressionTypes.hpp>
#include <fstream>
#include <magic_enum.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/operators/decompression-config.hpp>
#include <olap/operators/decompression-flags.hpp>
#include <platform/util/profiling.hpp>
#include <ssb/query.hpp>
#include "ssb-benchmarks.hpp"
#include "ssb-synthetic-benchmark.hpp"
#include "ssb-synthetic-queries.hpp"
#include "util.hpp"


DECLARE_bool(bench_varybw_cpu_ssb);
DEFINE_bool(bench_varybw_cpu_ssb, false,
            "CPU only vary number of NVMes used with uncompressed data using "
            "--scale_factor and varying the number of drives used");

DECLARE_bool(bench_varybw_gpu_ssb);
DEFINE_bool(bench_varybw_gpu_ssb, true,
            "GPU only vary number of NVMes used with uncompressed data using "
            "--scale_factor and varying the number of drives used");

DECLARE_bool(bench_gpu_synth_join_zero);
DEFINE_bool(bench_gpu_synth_join_zero, false, "");

DECLARE_bool(bench_gpu_synth_join_one);
DEFINE_bool(bench_gpu_synth_join_one, false, "");

DECLARE_bool(bench_gpu_synth_join_two);
DEFINE_bool(bench_gpu_synth_join_two, false, "");

DECLARE_bool(bench_gpu_synth_join_three);
DEFINE_bool(bench_gpu_synth_join_three, false, "");

DECLARE_bool(bench_gpu_synth_join_four);
DEFINE_bool(bench_gpu_synth_join_four, false, "");

DECLARE_bool(warm_up);
DEFINE_bool(
    warm_up, true,
    "Run one warm-up execution before benchmark iterations for each query.");

DECLARE_bool(print_res);
DEFINE_bool(print_res, true, "Print query results to stdout.");

DECLARE_int32(scale_factor);
DEFINE_int32(scale_factor, 100, "SSB scale factor");

DECLARE_int32(server_number);
DEFINE_int32(server_number, 49, "server number (DIAS internal)");

DECLARE_int32(num_iterations);
DEFINE_int32(num_iterations, 1, "Number of types to run each query");

DECLARE_int32(query);
DEFINE_int32(query, 11,
             "SSB query to execute (11-43). To run all use query=999. To run *.1 queries use query=99.");

DECLARE_int32(num_drives);
DEFINE_int32(num_drives, 1,
             "num drives used as bandwidth query to execute (11-43)");

DECLARE_string(result_file);
DEFINE_string(result_file, "",
              "[optional] output file for results [default: stdout]");

DECLARE_string(compression_type);
DECLARE_bool(decom_op);
DECLARE_bool(decom_reg);
DECLARE_bool(precompute_rl);
DECLARE_bool(pre_decom);
DECLARE_bool(use_metadata_in_block);
DECLARE_bool(one_kernel);

DECLARE_int32(memmove_dop);
DEFINE_int32(memmove_dop, 12, "");

DECLARE_int32(memmove_slack);
DEFINE_int32(memmove_slack, 4, "");

DECLARE_bool(in_gpu);
DECLARE_int32(tile_size);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);


// TLDR:
// Branch `user/paras/tracing-adm` focuses on GPU-side decompression for RFOR,
// with two paths:
//   1) NVMe -> GPU direct read + decompression inside MemMoveDevice.
//   2) (Optional) decompression in CompressedBlockToTuples when --decom_op=true.
// Target scenario: RFOR-compressed data on NVMe, cuFile read into GPU buffers,
// decompress on GPU, then run query execution.
// Lineage:
//   - user/duric/tracing-adm: introduces RFOR format/support.
//   - user/nicholson/tracing-adm: makes it work for the DaMoN '25 submission.
//
// Query shaper & operator sequence (NVMe -> GPU):
// - Shaper (e.g., GPUOnlyNVMeMorsel) injects routers/mem-moves and switches to GPU.
// - Typical flow:
//   Scan -> HintRowCount -> Router -> MemMoveDevice -> Router -> CpuToGpu
//   -> BlockToTuples -> Select/Project -> GpuHashJoinChained -> GpuReduce
//   -> GpuToCpu -> Router -> Reduce -> Flush
// - The build side also uses MemMoveDevice/Router and can broadcast/route
//   depending on the shaper (for joins it may use MemBroadcastDevice).
//
// Decompression logic in MemMoveDevice:
// - If the input pointer is an NvmePlugin PageId and the page is compressed,
//   MemMoveDevice detects it in make_mem_move_device(), calls nvme_plugin->getPageIoInfo(),
//   cuFileRead() reads the compressed bytes into a GPU buffer, then
//   batch_decompress_block_gpu() runs the RFOR/Cascaded/GZIP/LZ4 decompressor on the GPU and
//   produces a normal uncompressed 2MB block for downstream operators.
// - With --decom_op=false (default), decompression happens in MemMoveDevice.
// - With --decom_op=true:
//   - in_gpu + compressed-scan-emits-compressed-blocks:
//     defer decompression to CompressedBlockToTuples.
//   - otherwise:
//     MemMoveDevice keeps decompression enabled.
//
// Note: I just added Shaper::INGPU in this branch, but the compressed
// in‑GPU‑memory execution is being developed separately in
// `user/boffa/decomp-inGPU`.

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("adm-play");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG_ASSERT(!(FLAGS_one_kernel && !FLAGS_in_gpu))
      << "--one_kernel requires --in_gpu=true";
  LOG_ASSERT(FLAGS_tile_size == 512 || FLAGS_tile_size == 1024)
      << "--tile_size must be 512 or 1024 (got " << FLAGS_tile_size << ")";
  const bool has_threads_override = FLAGS_threads_per_block > 0;
  const bool has_items_override = FLAGS_items_per_thread > 0;
  LOG_ASSERT(has_threads_override == has_items_override)
      << "--threads_per_block and --items_per_thread must be set "
      << "together (or both left at 0)";
  if (FLAGS_compression_type == "RFOR") {
    // Dataset layout (FLAGS_tile_size) must match the active decode strategy.
    // Otherwise RFOR decode can silently return incorrect results.
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
          << "RFOR CB2T requires --threads_per_block * "
             "--items_per_thread == --tile_size ("
          << cb2t_block_threads << " * " << cb2t_items_per_thread
          << " != " << FLAGS_tile_size << ")";
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
            << flbp_shape.items_per_thread << " = "
            << flbp_shape.tile_size << "). Overriding --tile_size="
            << FLAGS_tile_size << " to " << flbp_shape.tile_size << ".";
        FLAGS_tile_size = flbp_shape.tile_size;
      }
    }
  }

    // MemoryManager initializes CPU pinned pools at startup regardless of exact query path.
    // With --in_gpu, default CPU pool size was too high on your machine and caused early mmap ENOMEM.
  if (FLAGS_in_gpu) {
    gflags::CommandLineFlagInfo cpu_buffers_flag;
    const bool has_cpu_buffers_flag =
        gflags::GetCommandLineFlagInfo("cpu_buffers", &cpu_buffers_flag);
    const bool cpu_buffers_is_default =
        has_cpu_buffers_flag && cpu_buffers_flag.is_default;

    if (cpu_buffers_is_default && FLAGS_cpu_buffers > 0.05) {
      LOG(INFO)
          << "--in_gpu set: lowering default --cpu_buffers from "
          << FLAGS_cpu_buffers
          << " to 0.05 to reduce startup pinned-memory pressure.";
      FLAGS_cpu_buffers = 0.05;
    }
  }

  auto ctx = proteus::from_cli::olap();

  std::stringstream ss;
  std::optional<std::ofstream> out = std::nullopt;

  // std::cout << "FLAGS_query: " << FLAGS_query << std::endl;
  // std::cout << "FLAGS_num_drives: " << FLAGS_num_drives << std::endl;

  if (!FLAGS_result_file.empty()) {
    out = std::ofstream(FLAGS_result_file);
    CHECK(out->is_open()) << "Could not open result file " << FLAGS_result_file;
  }

    const auto gpu_shaper_type = FLAGS_in_gpu ? Shaper::INGPU : Shaper::NVMEGPU;

    if (gpu_shaper_type == Shaper::INGPU)
    {
        LOG_ASSERT(FLAGS_scale_factor == 100) << "INGPU shaper currently only supports SSB SF100, since bigger sf dont fit GPU memory";
    }

  // if (FLAGS_bench_varybw_cpu) {
  //   LOG(INFO) << "running bench_varybw_cpu";
  //   auto res = bench_nvme_vary_bw(FLAGS_scale_factor, FLAGS_server_number,
  //                                 Shaper::NVMECPU, FLAGS_num_iterations);
  //   ss << res;
  //   ss << std::endl;
  //   if (out.has_value()) {
  //     *out << res << std::endl;
  //   }
  // }
  //


    LOG_ASSERT(!(FLAGS_decom_op && FLAGS_compression_type == "NONE"))
        << "--decom_op requires a compressed input type. "
           "Use --compression_type=RFOR/LZ4/CASCADED/GDEFLATE or disable --decom_op.";
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
      LOG(INFO) << "--precompute_rl + --decom_reg: expecting RFOR datasets with "
                   "precomputed cumulative RL and using no-prefix-sum register decode.";
    }

  // use for both compressed and uncompressed
  if (FLAGS_bench_varybw_gpu_ssb) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_varybw_gpu_ssb_compressed";

    auto res = bench_ssb_nvme_vary_bw_new(
        FLAGS_scale_factor, FLAGS_server_number, gpu_shaper_type,
        FLAGS_num_iterations, 4, 4, compression_type, FLAGS_query,
        FLAGS_num_drives);
      // Results printing
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  if (FLAGS_bench_gpu_synth_join_zero) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_gpu_synth_join_zero";

    auto res = bench_synthetic_join_query(
        {.num_drives = FLAGS_num_drives,
         .scale_factor = FLAGS_scale_factor,
         .compression_type = compression_type,
         .num_iterations = FLAGS_num_iterations,
         .server_number = FLAGS_server_number,
         .shaper_type = gpu_shaper_type,
         .prep_query_function = {synthetic_join_zero, "synthetic_join_zero"},
         .query_args = {
             .scan_memmove_dop_per_gpu = FLAGS_memmove_dop,
             .scan_memmove_slack = FLAGS_memmove_slack
         }});
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  if (FLAGS_bench_gpu_synth_join_one) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_gpu_synth_join_one";

    auto res = bench_synthetic_join_query(
        {.num_drives = FLAGS_num_drives,
         .scale_factor = FLAGS_scale_factor,
         .compression_type = compression_type,
         .num_iterations = FLAGS_num_iterations,
         .server_number = FLAGS_server_number,
         .shaper_type = gpu_shaper_type,
         .prep_query_function = {synthetic_join_one, "synthetic_join_one"},
         .query_args = {
             .scan_memmove_dop_per_gpu = FLAGS_memmove_dop,
             .scan_memmove_slack = FLAGS_memmove_slack
         }});
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  if (FLAGS_bench_gpu_synth_join_two) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_gpu_synth_join_two";

    auto res = bench_synthetic_join_query(
        {.num_drives = FLAGS_num_drives,
         .scale_factor = FLAGS_scale_factor,
         .compression_type = compression_type,
         .num_iterations = FLAGS_num_iterations,
         .server_number = FLAGS_server_number,
         .shaper_type = gpu_shaper_type,
         .prep_query_function = {synthetic_join_two, "synthetic_join_two"},
         .query_args = {
             .scan_memmove_dop_per_gpu = FLAGS_memmove_dop,
             .scan_memmove_slack = FLAGS_memmove_slack
         }});
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  if (FLAGS_bench_gpu_synth_join_three) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_gpu_synth_join_three";

    auto res = bench_synthetic_join_query(
        {.num_drives = FLAGS_num_drives,
         .scale_factor = FLAGS_scale_factor,
         .compression_type = compression_type,
         .num_iterations = FLAGS_num_iterations,
         .server_number = FLAGS_server_number,
         .shaper_type = gpu_shaper_type,
         .prep_query_function = {synthetic_join_three, "synthetic_join_three"},
         .query_args = {
             .scan_memmove_dop_per_gpu = FLAGS_memmove_dop,
             .scan_memmove_slack = FLAGS_memmove_slack
         }});
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  if (FLAGS_bench_gpu_synth_join_four) {
    auto compression_type =
        magic_enum::enum_cast<CompressionType>(FLAGS_compression_type).value();
    LOG(INFO) << "running bench_gpu_synth_join_three";

    auto res = bench_synthetic_join_query(
        {.num_drives = FLAGS_num_drives,
         .scale_factor = FLAGS_scale_factor,
         .compression_type = compression_type,
         .num_iterations = FLAGS_num_iterations,
         .server_number = FLAGS_server_number,
         .shaper_type = gpu_shaper_type,
         .prep_query_function = {synthetic_join_four, "synthetic_join_four"},
         .query_args = {
             .scan_memmove_dop_per_gpu = FLAGS_memmove_dop,
             .scan_memmove_slack = FLAGS_memmove_slack
         }});
    ss << res;
    ss << std::endl;
    if (out.has_value()) {
      *out << res << std::endl;
    }
  }

  std::cout << ss.str();
  auto& sm = StorageManager::getInstance();
  sm.unloadAll();
  return 0;
}
