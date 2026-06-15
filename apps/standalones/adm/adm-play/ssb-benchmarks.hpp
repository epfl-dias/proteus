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

#ifndef PROTEUS_SSB_BENCHMARKS_HPP
#define PROTEUS_SSB_BENCHMARKS_HPP
#include <chrono>
#include <codegen/expressions/expressionTypes.hpp>
#include <magic_enum.hpp>
#include <olap/operators/decompression-config.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <platform/util/profiling.hpp>
#include <query-shaping/nvme-shapers.hpp>
#include <ssb/query.hpp>
#include <vector>

#include "util.hpp"
#include "query-shaping/experimental-shapers.hpp"
#include "../../../../core/olap/lib/util/caching.hpp"
#include "../../../../core/olap/lib/util/catalog.hpp"

DECLARE_bool(decom_op);
DECLARE_bool(decom_reg);
DECLARE_bool(pre_decom);
DECLARE_bool(use_metadata_in_block);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);

struct BenchGpuMemoryOccupancyBytes {
    size_t compressed_dataset_bytes{0};
    size_t aux_bytes{0};

    [[nodiscard]] size_t total_bytes() const {
        return compressed_dataset_bytes + aux_bytes;
    }
};

inline BenchGpuMemoryOccupancyBytes collect_gpu_memory_occupancy_bytes()
{
    BenchGpuMemoryOccupancyBytes footprint;
    Catalog::getInstance().forEachPlugin([&](Plugin* pg)
    {
        if (auto* nvme_pg = dynamic_cast<NvmePlugin*>(pg))
        {
            auto nvme_footprint = nvme_pg->getGpuMemoryFootprint();
            footprint.compressed_dataset_bytes +=
                nvme_footprint.compressed_dataset_bytes;
            footprint.aux_bytes += nvme_footprint.aux_bytes;
        }
    });
    return footprint;
}

inline std::vector<std::filesystem::path> ssb_catalog_dirs_for_sf(int sf)
{
    return {
        std::filesystem::path("inputs") / "parser",
        std::filesystem::path("inputs") / "plans",
        std::filesystem::path("inputs") / ("ssbm" + std::to_string(sf)),
    };
}

inline void clear_catalog_parser_for_sf(int sf)
{
    CatalogParser::getInstance().clearAndParse(ssb_catalog_dirs_for_sf(sf));
}

std::string bench_nvme_vary_bw(int sf, int server_number,
                               Shaper shaper_type = Shaper::NVMECPU,
                               int num_iterations = 5,
                               int scan_router_slack = 2,
                               int scan_memmove_slack = 4, int num_drives = 1)
{
    CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";
    std::stringstream result_string;
    result_string << "query,"
        << "is_compressed,"
        << "num_drives,"
        << "diascld,"
        << "time_ms,"
        << "date,"
        << "scan_router_slack,"
        << "scan_memmove_slack,"
        << "shaper,"
        << "decompression_type,"
        << "threads_per_block,"
        << "items_per_thread"
    << std::endl;

    const auto all_md_dirs = get_input_dirs(sf, server_number, num_drives, CompressionType::LZ4);

    for (const auto& md_dirs : all_md_dirs)
    {
        /// important, because the relations are all the same from the point of view
        /// of the catalog we need to drop the catalog to ensure we use the right
        /// plugin instance for each configurations of md files
        clear_catalog_parser_for_sf(sf);
        LOG(INFO) << "running with " << md_dirs.size() << " md directories";
        std::unique_ptr<proteus::InputPrefixQueryShaper> shaper;
        switch (shaper_type)
        {
        case Shaper::NVMECPU:
            shaper = std::make_unique<proteus::CPUOnlyNVMeMorsel>(
                md_dirs, "inputs/ssbm" + std::to_string(sf), ssb::Query::getStats(sf), true,
                scan_memmove_slack, scan_router_slack, 16);
            break;
        case Shaper::NVMEGPU:
            shaper = std::make_unique<proteus::GPUOnlyNVMeMorsel>(
                md_dirs, "inputs/ssbm" + std::to_string(sf), ssb::Query::getStats(sf), true,
                scan_memmove_slack, scan_router_slack, 16);
            break;
        case Shaper::INGPU:
            {
                shaper = std::make_unique<proteus::GPUOnlySingleServer>("inputs/ssbm" + std::to_string(sf) + "/",
                                                                        ssb::Query::getStats(sf), true,
                                                                        16);
            }
            break;
        }
        const auto prt = profiling::ProfileRegionType(
            std::string("bench_nvme_vary_bw::") +
            std::string(magic_enum::enum_name(shaper_type)));
        auto prof_reg = profiling::ProfileRegion(prt);
        auto scan_query = small_scan(*shaper, "lo_commitdate");
        auto bench_res =
            benchmark_query("scan_commitdate", scan_query, num_iterations);
        result_string << bench_res.label << ","
            << "false," << md_dirs.size() << "," << server_number << ","
            << format_ms_1dp(bench_res.average_query_time) << ","
            << get_current_date_str() << "," << scan_router_slack << ","
            << scan_memmove_slack << ","
            << magic_enum::enum_name(shaper_type) << std::endl;
    }
    return result_string.str();
}

std::string bench_ssb_nvme_vary_bw_new(
    int sf, int server_number, Shaper shaper_type = Shaper::NVMEGPU,
    int num_iterations = 5, int scan_router_slack = 2,
    int scan_memmove_slack = 4,
    CompressionType compression_type = CompressionType::NONE,
    int query_number = 11, int num_drives = 1)
{
    CHECK(sf == 100 || sf == 1000) << "sf is not 100 or 1000";
    LOG(INFO) << " compression type:  "
        << magic_enum::enum_name(compression_type);
    std::stringstream result_string;
    result_string << "query,"
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

    //  const auto all_md_dirs = compressed
    //                               ? get_input_dirs_compressed(sf,
    //                               server_number) : get_input_dirs(sf,
    //                               server_number);

    const auto all_md_dirs = [&compression_type, &sf, &server_number, &num_drives]()
    {
        return get_input_dirs(sf, server_number, num_drives, compression_type);
    }();

    std::vector<std::pair<decltype(&ssb::Query::prepare11), std::string>> queries;

    if (query_number == 11 or query_number == 999 or query_number == 99 or query_number == 111)
    {
        queries.push_back({ssb::Query::prepare11, "MY_ssb_Q1.1"});
    }

    if (query_number == 12 || query_number == 999 or query_number == 111)
    {
        queries.push_back({ssb::Query::prepare12, "MY_ssb_Q1.2"});
    }

    if (query_number == 13 || query_number == 999 or query_number == 111)
    {
        queries.push_back({ssb::Query::prepare13, "MY_ssb_Q1.3"});
    }

    if (query_number == 21 || query_number == 999 or query_number == 99 or query_number == 222)
    {
        queries.push_back({ssb::Query::prepare21, "MY_ssb_Q2.1"});
    }

    if (query_number == 22 || query_number == 999 or query_number == 222)
    {
        queries.push_back({ssb::Query::prepare22, "MY_ssb_Q2.2"});
    }

    if (query_number == 23 || query_number == 999 or query_number == 222)
    {
        queries.push_back({ssb::Query::prepare23, "MY_ssb_Q2.3"});
    }

    if (query_number == 31 || query_number == 999 or query_number == 99 or query_number == 333)
    {
        queries.push_back({ssb::Query::prepare31, "MY_ssb_Q3.1"});
    }

    if (query_number == 32 || query_number == 999 or query_number == 333)
    {
        queries.push_back({ssb::Query::prepare32, "MY_ssb_Q3.2"});
    }

    if (query_number == 33 || query_number == 999 or query_number == 333)
    {
        queries.push_back({ssb::Query::prepare33, "MY_ssb_Q3.3"});
    }

    if (query_number == 34 || query_number == 999 or query_number == 333)
    {
        queries.push_back({ssb::Query::prepare34, "MY_ssb_Q3.4"});
    }


    if (query_number == 41 || query_number == 999 or query_number == 99 or query_number == 444)
    {
        queries.push_back({ssb::Query::prepare41, "MY_ssb_Q4.1"});
    }

    if (query_number == 42 || query_number == 999 or query_number == 444)
    {
        queries.push_back({ssb::Query::prepare42, "MY_ssb_Q4.2"});
    }

    if (query_number == 43 || query_number == 999 or query_number == 444)
    {
        queries.push_back({ssb::Query::prepare43, "MY_ssb_Q4.3"});
    }

    // Keep deferred clear only for single-query runs. For bundle runs (e.g.
    // query=99/999/111/222/333/444) clear state per query to avoid carrying
    // GPU/plugin state across heterogeneous plans.
    const bool is_multi_query_run = queries.size() > 1;
    const bool use_late_decompression = FLAGS_decom_op || FLAGS_decom_reg;
    const bool defer_inter_query_clear =
    (shaper_type == Shaper::INGPU &&
        compression_type != CompressionType::NONE &&
        use_late_decompression &&
        !is_multi_query_run);

    const bool decom_op = FLAGS_decom_op;
    const bool decom_reg = FLAGS_decom_reg;
    const bool pre_decom = FLAGS_pre_decom;
    std::string decomp_mode_str = "";
    if (compression_type == CompressionType::NONE)
    {
        decomp_mode_str = "NO_decom";
    }
    if (compression_type == CompressionType::RFOR or compression_type == CompressionType::FLBP)
    {
        if (pre_decom)
        {
            decomp_mode_str = "predecom";
        }
        else if (decom_reg)
        {
            decomp_mode_str = "decom_in_reg";
        }
        else if (decom_op)
        {
            decomp_mode_str = "decom_in_smem";
        }
        else
        {
            decomp_mode_str = "decom_in_gmem";
        }
    }

    for (const auto& md_dirs : all_md_dirs)
    {
        check_vector_paths(md_dirs);
        for (auto [query_prep_func, query_name] : queries)
        {

            QueryBenchResult bench_res;
            {
                // Query variants scan different attribute subsets; clear parser metadata
                // so each query can register the exact relation schema it needs.
                clear_catalog_parser_for_sf(sf);
                std::unique_ptr<proteus::InputPrefixQueryShaper> shaper;
                // only GPU for now
                //      switch (shaper_type) {
                //        case Shaper::NVMECPU:
                //          shaper = std::make_unique<proteus::CPUOnlyNVMeMorsel>(
                //              md_dirs, "inputs/ssbm1000", ssb::Query::getStats(sf),
                //              true, scan_memmove_slack, scan_router_slack, 16);
                //          break;
                //        case Shaper::NVMEGPU:
                //          shaper = std::make_unique<proteus::GPUOnlyNVMeMorsel>(
                //              md_dirs, "inputs/ssbm1000", ssb::Query::getStats(sf),
                //              true, scan_memmove_slack, scan_router_slack, 16);
                //          break;
                //      }

                if (shaper_type == Shaper::NVMEGPU)
                {
                    shaper = std::make_unique<proteus::GPUOnlyNVMeMorsel>(
                        md_dirs, "inputs/ssbm" + std::to_string(sf), ssb::Query::getStats(sf), true,
                        scan_memmove_slack, scan_router_slack, 16);
                }
                else if (shaper_type == Shaper::INGPU)
                {
                    if (compression_type == CompressionType::NONE)
                    {
                        shaper = std::make_unique<proteus::GPUOnlySingleServer>(
                            "inputs/ssbm" + std::to_string(sf) + "/",
                            ssb::Query::getStats(sf), false, 16);

                        auto& sm = StorageManager::getInstance();
                        sm.setDefaultLoader(
                            [](StorageManager& stManager, const std::string& fileName,
                               size_t typeSize)
                            {
                                return FileRecord::loadToGpus(fileName, typeSize);
                            },
                            true);
                    }
                    else
                    {
                        // In in-GPU compressed mode, keep scan output in compressed
                        // pointer form. Operator selection decides decompression policy:
                        // - --decom_op=false: CompressedBlockToTuplesMaterialize
                        // - --decom_op=true : CompressedBlockToTuplesDeferred
                        // - --decom_reg=true: CompressedBlockToTuplesReg
                        // - --pre_decom    : PreDecompress + BlockToTuples
                        //
                        // This keeps NvmePlugin agnostic to higher-level
                        // decompression strategy.
                        constexpr bool emit_compressed_blocks = true;
                        // Compressed in-GPU path: drive scan from metadata directories
                        // so --in_gpu + --compression_type=<...> does not silently fall
                        // back to uncompressed inputs/ssbm*. This uses NvmePlugin scan
                        // metadata (md_dirs) while keeping single-server shaping.
                        //
                        // Important: allowMoves=false keeps MemMoveDevice/cuFile out of
                        // execute. NvmePlugin preloads to GPU and emits either compressed
                        // or decompressed block pointers directly.
                        shaper = std::make_unique<proteus::GPUOnlySingleServerCompressed>(
                            md_dirs, "inputs/ssbm" + std::to_string(sf),
                            ssb::Query::getStats(sf), /*allowMoves=*/false,
                            scan_memmove_slack, scan_router_slack, 16,
                            emit_compressed_blocks);
                    }
                }
                else
                {
                    LOG(FATAL) << "shaper not supported: " << magic_enum::enum_name(shaper_type);
                }

                auto prep_query = query_prep_func(*shaper);
                bench_res = benchmark_query(query_name, prep_query, num_iterations);
            }
            const auto gpu_occupancy = collect_gpu_memory_occupancy_bytes();


            auto thread_per_block = FLAGS_threads_per_block;
            auto items_per_thread = FLAGS_items_per_thread;

            if (compression_type == CompressionType::NONE)
            {
                thread_per_block = 1024;
                // ints in 2MB storage block
                auto ints_per_block = BlockManager::block_size / sizeof(int32_t);
                items_per_thread = ints_per_block / thread_per_block;
            }
            else if (compression_type == CompressionType::RFOR && pre_decom)
            {
                const auto rfor_shape =
                    proteus::rfor::predecompress::resolveRuntimeShape(
                        FLAGS_tile_size, FLAGS_threads_per_block,
                        FLAGS_items_per_thread);
                CHECK_GT(rfor_shape.tile_size, 0)
                    << "Invalid RFOR pre-decompress runtime shape";
                thread_per_block = rfor_shape.block_threads;
                items_per_thread = rfor_shape.items_per_thread;
            }
            else if (compression_type == CompressionType::FLBP && pre_decom)
            {
                const auto flbp_shape =
                    proteus::flbp::predecompress::resolveRuntimeShape(
                        FLAGS_threads_per_block, FLAGS_items_per_thread);
                CHECK_GT(flbp_shape.tile_size, 0)
                    << "Invalid FLBP pre-decompress runtime shape";
                thread_per_block = flbp_shape.block_threads;
                items_per_thread = flbp_shape.items_per_thread;
            }
            else if (compression_type == CompressionType::RFOR ||
                     compression_type == CompressionType::FLBP)
            {
                // CB2T decom_op / decom_reg: when the shape is not given on the
                // command line, the operator falls back to the codec's default
                // (see applyCb2tRuntimeOverrides in compressed-block-to-tuples.cpp).
                // Mirror that here so the reported shape is the one actually used.
                const auto codec = compression_type == CompressionType::RFOR
                                       ? proteus::decompression::Cb2tCodec::kRfor
                                       : proteus::decompression::Cb2tCodec::kFlbp;
                const auto defaults =
                    proteus::decompression::getCb2tRuntimeDefaults(codec);
                if (thread_per_block <= 0)
                {
                    thread_per_block = defaults.block_threads;
                }
                if (items_per_thread <= 0)
                {
                    items_per_thread = defaults.items_per_thread;
                }
            }

            for (auto& query_time : bench_res.per_query_times)
            {
                result_string << bench_res.label << ","
                    << magic_enum::enum_name(compression_type) << ","
                    << md_dirs.size() << "," << server_number << ","
                    << format_ms_1dp(query_time) << ","
                    << get_current_date_str() << "," << scan_router_slack
                    << "," << scan_memmove_slack << ","
                    << magic_enum::enum_name(shaper_type) << ","
                    << decomp_mode_str << ","
                    << std::to_string(thread_per_block) << ","
                    << std::to_string(items_per_thread) << ","
                    << gpu_occupancy.compressed_dataset_bytes << ","
                    << gpu_occupancy.aux_bytes << ","
                    << gpu_occupancy.total_bytes() << std::endl;
            }
            if (!defer_inter_query_clear)
            {
                Catalog::getInstance().clear();
                CachingService::getInstance().clear();
                clear_catalog_parser_for_sf(sf);
            }
        }
        if (defer_inter_query_clear)
        {
            // Deferred clear path for in_gpu+compressed with late decompression.
            Catalog::getInstance().clear();
            CachingService::getInstance().clear();
            clear_catalog_parser_for_sf(sf);
        }
    }
    return result_string.str();
}

#endif  // PROTEUS_SSB_BENCHMARKS_HPP
