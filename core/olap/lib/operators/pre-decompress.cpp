/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#include "pre-decompress.hpp"

#include <gflags/gflags.h>
#include <codegen/jit/pipeline.hpp>
#include <algorithm>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <olap/plugins/plugins.hpp>
#include <platform/util/tracing.hpp>
#include <unordered_set>

#include "lib/operators/mem-move/compression.hpp"
#include "lib/util/catalog.hpp"

DECLARE_bool(use_metadata_in_block);

namespace {
std::unordered_set<NvmePlugin *> collectNvmePlugins(
    const std::vector<expression_t> &wantedFields) {
  std::unordered_set<NvmePlugin *> nvme_plugins;
  for (const auto &wantedField : wantedFields) {
    if (!dynamic_cast<const BlockType *>(wantedField.getExpressionType())) {
      continue;
    }
    auto *pg =
        Catalog::getInstance().getPlugin(wantedField.getRegisteredRelName());
    if (auto *nvme_pg = dynamic_cast<NvmePlugin *>(pg)) {
      nvme_plugins.insert(nvme_pg);
    }
  }
  return nvme_plugins;
}
}  // namespace

void runNvmePreDecompressKernel(NvmePlugin *pg) {
  CHECK(pg != nullptr) << "NvmePlugin pointer is null";
  CHECK(pg->preload_to_gpu_) << "Decompression requires preload_to_gpu";
  LOG(INFO) << "PreDecompress begin for relation " << pg->fnamePrefix;
  // Benchmark mode: pay pre-decompression on every execute.
  pg->preloaded_decompressed_ready_.store(false, std::memory_order_release);

  const size_t attr_count = pg->m_attribute_metadata.size();
  const int gpu_index = topology::getInstance().getActiveGpu().index_in_topo;
  set_device_on_scope device_scope(topology::getInstance().getGpus().at(gpu_index));
  if (pg->preloaded_decompressed_part_files_.empty()) {
    pg->preloaded_decompressed_part_files_.resize(attr_count);
  }
  if (pg->preloaded_decompressed_block_ptrs_.empty()) {
    pg->preloaded_decompressed_block_ptrs_.resize(attr_count);
  }

  std::vector<std::span<const uint32_t>> rfor_batch_chunk_sizes;
  std::vector<std::span<char>> rfor_batch_compressed_buffers;
  std::vector<std::span<char>> rfor_batch_decompressed_buffers;
  size_t total_rfor_tiles = 0;
  size_t max_rfor_decompressed_chunk_size = 1;

  std::vector<std::span<const uint32_t>> flbp_batch_chunk_sizes;
  std::vector<std::span<char>> flbp_batch_compressed_buffers;
  std::vector<std::span<char>> flbp_batch_decompressed_buffers;
  size_t total_flbp_vectors = 0;
  size_t max_flbp_decompressed_chunk_size = 1;
  CHECK(FLAGS_use_metadata_in_block)
      << "PreDecompress requires PBMD layout (val_section_sizes/tile_metadata "
         "are no longer carried in JSON); enable --use_metadata_in_block";

  for (size_t attr_no = 0; attr_no < attr_count; ++attr_no) {
    CHECK_LT(attr_no, pg->m_attribute_metadata.size());
    const auto &attr_md = pg->m_attribute_metadata[attr_no];
    const size_t part_count = attr_md.size();

    pg->preloaded_decompressed_part_files_[attr_no].resize(part_count);
    pg->preloaded_decompressed_block_ptrs_[attr_no].resize(part_count);

    for (size_t part_no = 0; part_no < part_count; ++part_no) {
      const auto &part_meta = attr_md[part_no];
      const size_t block_count = part_meta.num_blocks;
      if (block_count == 0) {
        pg->preloaded_decompressed_block_ptrs_[attr_no][part_no].clear();
        continue;
      }

      if (part_meta.data_format ==
          NvmePlugin::AttributePartMetaData::DataFormat_t::UNCOMPRESSED) {
        continue;
      }

      const bool is_rfor = part_meta.compression_algorithm.empty() ||
                           part_meta.compression_algorithm == "RFOR" ||
                           part_meta.compression_algorithm == "rfor";
      const bool is_flbp = part_meta.compression_algorithm == "FLBP" ||
                           part_meta.compression_algorithm == "flbp";
      CHECK(is_rfor || is_flbp)
          << "Unsupported compressed algorithm for in_gpu preload: "
          << part_meta.compression_algorithm;

      if (is_rfor) {
        max_rfor_decompressed_chunk_size = std::max<size_t>(
            max_rfor_decompressed_chunk_size,
            static_cast<size_t>(part_meta.decompressed_chunk_size));
      } else {
        max_flbp_decompressed_chunk_size = std::max<size_t>(
            max_flbp_decompressed_chunk_size,
            static_cast<size_t>(part_meta.decompressed_chunk_size));
      }

      pg->preloaded_decompressed_block_ptrs_[attr_no][part_no].resize(
          block_count, nullptr);

      CHECK_LT(attr_no, pg->preloaded_compressed_part_files_.size());
      CHECK_LT(part_no, pg->preloaded_compressed_part_files_[attr_no].size());
      CHECK(pg->preloaded_compressed_part_files_[attr_no][part_no])
          << "Compressed part not preloaded for attr=" << attr_no
          << " part=" << part_no;

      auto *compressed_gpu_base = static_cast<char *>(const_cast<void *>(
          pg->preloaded_compressed_part_files_[attr_no][part_no]->getData()));

      const size_t total_decompressed_bytes =
          block_count * static_cast<size_t>(BlockManager::block_size);
      if (total_decompressed_bytes == 0) {
        continue;
      }

      auto &decompressed_part_file =
          pg->preloaded_decompressed_part_files_[attr_no][part_no];
      auto &decompressed_block_ptrs =
          pg->preloaded_decompressed_block_ptrs_[attr_no][part_no];
      const size_t old_decompressed_bytes =
          decompressed_part_file ? decompressed_part_file->getFileSize() : 0;
      const bool need_realloc =
          !decompressed_part_file ||
          decompressed_part_file->getFileSize() != total_decompressed_bytes ||
          decompressed_block_ptrs.size() != block_count;
      if (need_realloc) {
        void *new_decompressed_gpu =
            MemoryManager::mallocGpu(total_decompressed_bytes);
        decompressed_part_file =
            std::make_unique<mmap_file>(new_decompressed_gpu,
                                        total_decompressed_bytes);
        decompressed_block_ptrs.assign(block_count, nullptr);
        if (total_decompressed_bytes >= old_decompressed_bytes) {
          pg->gpu_preloaded_decompressed_bytes_.fetch_add(
              total_decompressed_bytes - old_decompressed_bytes,
              std::memory_order_acq_rel);
        } else {
          pg->gpu_preloaded_decompressed_bytes_.fetch_sub(
              old_decompressed_bytes - total_decompressed_bytes,
              std::memory_order_acq_rel);
        }
      } else if (decompressed_block_ptrs.size() != block_count) {
        decompressed_block_ptrs.resize(block_count, nullptr);
      }
      void *decompressed_gpu =
          const_cast<void *>(decompressed_part_file->getData());
      CHECK_NE(decompressed_gpu, nullptr)
          << "Failed to initialize decompressed GPU part buffer";

      for (size_t block_no = 0; block_no < block_count; ++block_no) {
        auto *dst = static_cast<char *>(decompressed_gpu) +
                    block_no * static_cast<size_t>(BlockManager::block_size);
        decompressed_block_ptrs[block_no] = dst;

        if (part_meta.chunk_sizes[block_no].empty()) {
          gpu_run(cudaMemset(dst, 0, BlockManager::block_size));
          continue;
        }

        CHECK_LT(block_no, part_meta.chunk_sizes.size())
            << "Missing chunk_sizes for compressed block " << block_no
            << " in " << part_meta.data_file_path;

        const auto &chunk_sizes = part_meta.chunk_sizes[block_no];

        auto chunk_sizes_span =
            std::span<const uint32_t>(chunk_sizes.data(), chunk_sizes.size());
        auto compressed_span = std::span<char>(
            compressed_gpu_base + part_meta.block_offsets[block_no],
            part_meta.block_sizes[block_no]);
        auto decompressed_span = std::span<char>(
            dst, static_cast<size_t>(BlockManager::block_size));

        if (is_rfor) {
          total_rfor_tiles += chunk_sizes.size();
          rfor_batch_chunk_sizes.emplace_back(chunk_sizes_span);
          rfor_batch_compressed_buffers.emplace_back(compressed_span);
          rfor_batch_decompressed_buffers.emplace_back(decompressed_span);
        } else {
          total_flbp_vectors += chunk_sizes.size();
          flbp_batch_chunk_sizes.emplace_back(chunk_sizes_span);
          flbp_batch_compressed_buffers.emplace_back(compressed_span);
          flbp_batch_decompressed_buffers.emplace_back(decompressed_span);
        }
      }
    }
  }

  if (!rfor_batch_chunk_sizes.empty()) {
    auto gpu_decompressor = GpuDecompressor(
        max_rfor_decompressed_chunk_size,
        /*max_batch_block_count=*/1,
        std::max<size_t>(1, total_rfor_tiles),
        CompressionAlgorithm::RFOR,
        gpu_index);

    int decomp_res = gpu_decompressor.batch_decompress_block_gpu(
        rfor_batch_chunk_sizes, rfor_batch_compressed_buffers,
        rfor_batch_decompressed_buffers,
        /*stream=*/nullptr,
        std::vector<std::span<const uint32_t>>{});
    CHECK_GE(decomp_res, 0) << "Failed to GPU-predecompress RFOR blocks";
  }

  if (!flbp_batch_chunk_sizes.empty()) {
    auto gpu_decompressor = GpuDecompressor(
        max_flbp_decompressed_chunk_size,
        /*max_batch_block_count=*/1,
        std::max<size_t>(1, total_flbp_vectors),
        CompressionAlgorithm::FLBP,
        gpu_index);

    int decomp_res = gpu_decompressor.batch_decompress_block_gpu(
        flbp_batch_chunk_sizes, flbp_batch_compressed_buffers,
        flbp_batch_decompressed_buffers,
        /*stream=*/nullptr,
        std::vector<std::span<const uint32_t>>{});
    CHECK_GE(decomp_res, 0) << "Failed to GPU-predecompress FLBP blocks";
  }

  pg->preloaded_decompressed_ready_.store(true, std::memory_order_release);
  LOG(INFO) << "PreDecompress done for relation " << pg->fnamePrefix;
}

void PreDecompress::produce_(OlapParallelContext *context) {
  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });
  getChild()->produce(context);
}

void PreDecompress::consume(OlapParallelContext *context,
                            const OperatorState &childState) {
  getParent()->consume(context, childState);
}

void PreDecompress::open(Pipeline *pip) {
  event_range<range_log_op::UNPACK_OPEN> er{id, pip->getGeneratorUUID(),
                                            pip->getGroup()};

  if (!gpu) {
    return;
  }

  LOG(INFO) << "PreDecompress open pipeline " << pip->getGeneratorUUID();
  auto nvme_plugins = collectNvmePlugins(wantedFields);

  for (auto *nvme_pg : nvme_plugins) {
    nvme_pg->ensurePreloadedDecompressedBlocks();
    if (nvme_pg->emitsCompressedBlocksForScan()) {
      nvme_pg->setScanEmitCompressedBlocks(false);
    }
  }
}

void PreDecompress::close(Pipeline *pip) {
  event_range<range_log_op::UNPACK_CLOSE> er{id, pip->getGeneratorUUID(),
                                             pip->getGroup()};
}
