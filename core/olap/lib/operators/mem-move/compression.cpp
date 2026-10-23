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

#include "compression.hpp"

#include <gflags/gflags.h>
#include <lz4.h>

#include <magic_enum.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnewline-eof"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#pragma clang diagnostic ignored "-Wdocumentation"

#include "../decompression/flbp-decompression.hpp"
#include "../decompression/rfor-decompression.hpp"

#include <nvcomp/lz4.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/gdeflate.h>
#include <fstream>

#pragma clang diagnostic pop

DECLARE_bool(use_metadata_in_block);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);

GpuDecompressor::GpuDecompressor(size_t max_decomp_chunk_size,
                                 size_t max_batch_block_count,
                                 size_t max_chunks_per_block,
                                 CompressionAlgorithm comp_algo,
                                 int gpu_index_in_topo)
    : m_max_decomp_chunk_size(max_decomp_chunk_size),
      m_comp_algo(comp_algo), last_batch_num_chunks(0) {
  // ensure that we are allocating memory on the correct GPU
  int gpu_index = gpu_index_in_topo;
  if (gpu_index_in_topo == -1) {
    gpu_index = topology::getInstance().getActiveGpu().index_in_topo;
  }
  m_gpu_index_in_topo = gpu_index;

  set_device_on_scope d(topology::getInstance().getGpus()[m_gpu_index_in_topo]);

  const int max_batch_chunks = max_batch_block_count * max_chunks_per_block;
  m_host_compressed_ptrs = static_cast<void **>(
      MemoryManager::mallocPinned(sizeof(void *) * max_batch_chunks));
  m_device_compressed_ptrs = static_cast<void **>(
      MemoryManager::mallocGpu(sizeof(void *) * max_batch_chunks));
  m_host_uncompressed_ptrs = static_cast<void **>(
      MemoryManager::mallocPinned(sizeof(void *) * max_batch_chunks));
  m_device_uncompressed_ptrs = static_cast<void **>(
      MemoryManager::mallocGpu(sizeof(void *) * max_batch_chunks));
  m_host_compressed_chunk_sizes = static_cast<size_t *>(
      MemoryManager::mallocPinned(sizeof(size_t) * max_batch_chunks));

  m_device_compressed_bytes = static_cast<size_t *>(
      MemoryManager::mallocGpu(sizeof(size_t) * max_batch_chunks));

  nvcompStatus_t status;
  m_device_decomp_workspace_size = 0;
  switch (m_comp_algo) {
    case CompressionAlgorithm::LZ4:
      status = nvcompBatchedLZ4DecompressGetTempSize(max_batch_chunks,
                                                     max_decomp_chunk_size,
                                                     &m_device_decomp_workspace_size);
      break;
    case CompressionAlgorithm::CASCADED:
      status = nvcompBatchedCascadedDecompressGetTempSize(max_batch_chunks,
                                                          max_decomp_chunk_size,
                                                          &m_device_decomp_workspace_size);
      break;
    case CompressionAlgorithm::GDEFLATE:
      status = nvcompBatchedGdeflateDecompressGetTempSize(max_batch_chunks,
                                                          max_decomp_chunk_size,
                                                          &m_device_decomp_workspace_size);
      break;
    case CompressionAlgorithm::RFOR: {
      // RFOR uses workspace for pointer arrays instead of nvcomp operations
      const int max_total_tiles = max_batch_block_count * max_chunks_per_block;

      size_t ptr_arrays_size = sizeof(void*) * max_total_tiles * 4;



      // size_t safety_margin = 4096; // 4KB safety margin
      m_device_decomp_workspace_size = ptr_arrays_size;

      // LOG(INFO) << "RFOR workspace: max_total_tiles=" << max_total_tiles
      //           << ", ptr_arrays_size=" << ptr_arrays_size
      //           << ", total_workspace=" << m_device_decomp_workspace_size;

      status = nvcompSuccess;
      break;
    }
    case CompressionAlgorithm::FLBP: {
      // FLBP uses workspace for pointer arrays instead of nvcomp operations
      const int max_total_vectors =
          max_batch_block_count * max_chunks_per_block;
      m_device_decomp_workspace_size = sizeof(void*) * max_total_vectors * 3;
      status = nvcompSuccess;
      break;
    }
    default:
      throw std::runtime_error("Unsupported compression algorithm");
  }
  if (status != nvcompSuccess && m_comp_algo != CompressionAlgorithm::RFOR &&
      m_comp_algo != CompressionAlgorithm::FLBP) {
    throw std::runtime_error(
        std::string("nvcompBatchedLZ4DecompressGetTempSize() failed with: ") +
        std::string(magic_enum::enum_name(status)));
  }
  m_device_decomp_workspace =
      MemoryManager::mallocGpu(m_device_decomp_workspace_size);
  m_device_actual_uncompressed_bytes = static_cast<size_t *>(
      MemoryManager::mallocGpu(sizeof(size_t) * max_batch_chunks));
  m_device_uncompressed_bytes = static_cast<size_t *>(
      MemoryManager::mallocGpu(sizeof(size_t) * max_batch_chunks));

  m_device_status_ptrs = static_cast<nvcompStatus_t *>(
      MemoryManager::mallocGpu(sizeof(nvcompStatus_t) * max_batch_chunks));

  m_host_uncompressed_chunk_sizes = static_cast<size_t *>(
      MemoryManager::mallocPinned(sizeof(size_t) * max_batch_chunks));
  m_device_uncompressed_chunk_sizes = static_cast<size_t *>(
      MemoryManager::mallocGpu(sizeof(size_t) * max_batch_chunks));
}

GpuDecompressor::~GpuDecompressor() {
  MemoryManager::freePinned(m_host_compressed_ptrs);
  MemoryManager::freeGpu(m_device_compressed_ptrs);
  MemoryManager::freePinned(m_host_uncompressed_ptrs);
  MemoryManager::freeGpu(m_device_uncompressed_ptrs);
  MemoryManager::freePinned(m_host_compressed_chunk_sizes);
  MemoryManager::freeGpu(m_device_compressed_bytes);

  MemoryManager::freeGpu(m_device_decomp_workspace);
  MemoryManager::freeGpu(m_device_actual_uncompressed_bytes);
  MemoryManager::freeGpu(m_device_uncompressed_bytes);
  MemoryManager::freeGpu(m_device_status_ptrs);
  MemoryManager::freePinned(m_host_uncompressed_chunk_sizes);
  MemoryManager::freeGpu(m_device_uncompressed_chunk_sizes);
}

int GpuDecompressor::get_last_batch_bytes_decompressed() {
  DCHECK_GT(last_batch_num_chunks, 0) << "No outstanding asynchronous call.";
  // Assume device_status_ptrs is an array of nvcompStatus_t on the device
  if (m_comp_algo != CompressionAlgorithm::RFOR &&
      m_comp_algo != CompressionAlgorithm::FLBP) {
    std::vector<nvcompStatus_t> host_status_ptrs(last_batch_num_chunks);

    // Copy the status from device to host
    gpu_run(cudaMemcpy(host_status_ptrs.data(), m_device_status_ptrs,
                       last_batch_num_chunks * sizeof(nvcompStatus_t),
                       cudaMemcpyDeviceToHost));

    // Check the status of each decompression task
    for (int i = 0; i < last_batch_num_chunks; i++) {
      if (host_status_ptrs[i] != nvcompSuccess) {
        LOG(ERROR) << "GPU decompression failed for chunk " << i
                   << " with status "
                   << magic_enum::enum_name(host_status_ptrs[i]);
        last_batch_num_chunks = 0;
        return -(i + 1);
      }
    }
  }

  std::vector<size_t> host_actual_uncompressed_bytes(last_batch_num_chunks);
  gpu_run(cudaMemcpy(
      host_actual_uncompressed_bytes.data(), m_device_actual_uncompressed_bytes,
      last_batch_num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
  size_t decompressed_size = 0;
  for (size_t i = 0; i < last_batch_num_chunks; i++) {
    decompressed_size += host_actual_uncompressed_bytes[i];
  }
  last_batch_num_chunks = 0;
  return decompressed_size;
}

void **GpuDecompressor::get_device_uncompressed_ptrs(
    const std::vector<std::span<char>> &block_output_buffers,
    const std::vector<std::span<const uint32_t>> &block_chunk_sizes,
    size_t batch_size, cudaStream_t stream) {
  size_t ix_chunk = 0;
  for (size_t i = 0; i < block_output_buffers.size(); ++i) {
    for (size_t j = 0; j < block_chunk_sizes[i].size(); ++j) {
      m_host_uncompressed_ptrs[ix_chunk] =
          block_output_buffers[i].data() + j * m_max_decomp_chunk_size;
      ++ix_chunk;
    }
  }
  DCHECK_EQ(ix_chunk, batch_size);

  gpu_run(cudaMemcpyAsync(m_device_uncompressed_ptrs, m_host_uncompressed_ptrs,
                          sizeof(void *) * batch_size, cudaMemcpyHostToDevice,
                          stream));

  return m_device_uncompressed_ptrs;
}

size_t *GpuDecompressor::get_device_uncompressed_chunk_sizes(
    size_t batch_size, cudaStream_t stream) {
  // We know that every chunk will be < max_decomp_chunk_size
  // so we can skip a call to nvcompBatchedLZ4GetDecompressSizeAsync

  for (size_t i = 0; i < batch_size; ++i) {
    m_host_uncompressed_chunk_sizes[i] = m_max_decomp_chunk_size;
  }

  gpu_run(cudaMemcpyAsync(
      m_device_uncompressed_chunk_sizes, m_host_uncompressed_chunk_sizes,
      sizeof(void *) * batch_size, cudaMemcpyHostToDevice, stream));

  return m_device_uncompressed_chunk_sizes;
}

/* Batch version of the function above. */
size_t *GpuDecompressor::get_device_compressed_bytes(
    const std::vector<std::span<const uint32_t>> &blocks_chunk_sizes,
    size_t batch_size, cudaStream_t stream) {
  size_t ix_chunk = 0;
  for (const auto &block_chunk_size : blocks_chunk_sizes) {
    for (unsigned int chunk_size : block_chunk_size) {
      m_host_compressed_chunk_sizes[ix_chunk] = chunk_size;
      ++ix_chunk;
    }
  }
  DCHECK_EQ(ix_chunk, batch_size);

  gpu_run(cudaMemcpyAsync(
      m_device_compressed_bytes, m_host_compressed_chunk_sizes,
      sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream));

  return m_device_compressed_bytes;
}

/* Batch version of the function above. */
void **GpuDecompressor::get_device_compressed_ptrs(
    const std::vector<std::span<char>> &compressed_buffers,
    const std::vector<std::span<const uint32_t>> &chunk_sizes, size_t batch_size,
    cudaStream_t stream) {
  size_t ix_chunk = 0;
  for (size_t i = 0; i < compressed_buffers.size(); ++i) {
    size_t current_offset = 0;
    for (size_t j = 0; j < chunk_sizes[i].size(); ++j) {
      m_host_compressed_ptrs[ix_chunk] =
          compressed_buffers[i].data() + current_offset;
      current_offset += chunk_sizes[i][j];
      ++ix_chunk;
    }
  }
  DCHECK_EQ(ix_chunk, batch_size);

  gpu_run(cudaMemcpyAsync(m_device_compressed_ptrs, m_host_compressed_ptrs,
                          sizeof(void *) * batch_size, cudaMemcpyHostToDevice,
                          stream));

  return m_device_compressed_ptrs;
}


[[nodiscard]] int GpuDecompressor::batch_decompress_rfor(
    const std::vector<std::span<const uint32_t>>& block_chunk_sizes,
    const std::vector<std::span<char>>& block_compressed_buffers,
    const std::vector<std::span<char>>& block_output_buffers,
    cudaStream_t stream,
    const std::vector<std::span<const uint32_t>>& block_val_section_sizes) {

    if (block_chunk_sizes.size() != block_compressed_buffers.size() ||
        block_chunk_sizes.size() != block_output_buffers.size()) {
        LOG(ERROR) << "RFOR batch input size mismatch: chunk_sizes="
                   << block_chunk_sizes.size() << " compressed="
                   << block_compressed_buffers.size() << " output="
                   << block_output_buffers.size();
        return -1;
    }

    if (!block_val_section_sizes.empty() &&
        block_val_section_sizes.size() != block_chunk_sizes.size()) {
        LOG(ERROR) << "RFOR val_section_sizes size mismatch: val_sections="
                   << block_val_section_sizes.size()
                   << " blocks=" << block_chunk_sizes.size();
        return -1;
    }

    const bool try_pbmd_fast_path =
        FLAGS_use_metadata_in_block && block_val_section_sizes.empty();
    if (block_val_section_sizes.empty() && !try_pbmd_fast_path) {
        LOG(ERROR) << "RFOR decompression requires val_section_sizes from JSON metadata";
        return -1;
    }

    set_device_on_scope d(topology::getInstance().getGpus()[m_gpu_index_in_topo]);

    const bool synchronous = (stream == nullptr);
    if (synchronous) {
        stream = createNonBlockingStream();
    }
    if (m_max_decomp_chunk_size % sizeof(int) != 0) {
        LOG(ERROR) << "Invalid RFOR max_decomp_chunk_size="
                   << m_max_decomp_chunk_size
                   << " (must be a multiple of sizeof(int))";
        return -1;
    }
    const int rfor_tile_size =
        static_cast<int>(m_max_decomp_chunk_size / sizeof(int));
    const auto rfor_shape = resolveRforKernelShape(
        rfor_tile_size, FLAGS_threads_per_block, FLAGS_items_per_thread);
    if (rfor_shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported RFOR pre-decompress shape "
                   << FLAGS_threads_per_block << "x"
                   << FLAGS_items_per_thread
                   << " for tile_size=" << rfor_tile_size
                   << " derived from max_decomp_chunk_size="
                   << m_max_decomp_chunk_size;
        return -1;
    }
    const size_t rfor_tile_bytes = static_cast<size_t>(rfor_tile_size) * sizeof(int);

    // Count total tiles across all blocks
    size_t total_tiles = 0;
    for (const auto& chunk_sizes : block_chunk_sizes) {
        total_tiles += chunk_sizes.size();
    }

    // std::cout << "\n=== RFOR GPU DEBUG INFO ===" << std::endl;
    // std::cout << "Total tiles to decompress: " << total_tiles << std::endl;
    // std::cout << "GPU index: " << m_gpu_index_in_topo << std::endl;

    if (total_tiles == 0) {
        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
        }
        return 0;
    }

    last_batch_num_chunks = total_tiles;

    if (try_pbmd_fast_path) {
        std::vector<RFORPBMDBlockDesc> h_block_descs;
        h_block_descs.reserve(block_chunk_sizes.size());

        uint32_t tile_prefix = 0;
        constexpr size_t kPbmdFixedHeaderBytes =
            3 * sizeof(uint32_t);  // magic, version, chunk_count
        bool all_pbmd_layout = true;

        for (size_t block_idx = 0; block_idx < block_chunk_sizes.size();
             ++block_idx) {
            const auto& chunk_sizes = block_chunk_sizes[block_idx];
            const auto& compressed_buffer = block_compressed_buffers[block_idx];
            const auto& output_buffer = block_output_buffers[block_idx];

            const size_t num_chunks = chunk_sizes.size();
            const size_t required_output_bytes =
                num_chunks * rfor_tile_bytes;
            if (required_output_bytes > output_buffer.size()) {
                LOG(ERROR) << "Output buffer too small for RFOR block "
                           << block_idx << ": need " << required_output_bytes
                           << " bytes, have " << output_buffer.size();
                return -1;
            }

            size_t total_chunk_bytes = 0;
            for (uint32_t chunk_size : chunk_sizes) {
                total_chunk_bytes += chunk_size;
            }
            if (total_chunk_bytes > compressed_buffer.size()) {
                LOG(ERROR) << "Compressed buffer too small for RFOR block "
                           << block_idx << ": chunk bytes "
                           << total_chunk_bytes << " > block bytes "
                           << compressed_buffer.size();
                return -1;
            }

            // PBMD layout: [magic][algo][num_chunks][chunk_offsets...][payload chunks...]
            // Each chunk is self-describing (begins with its native codec header).
            const size_t pbmd_prefix_bytes =
                kPbmdFixedHeaderBytes + (num_chunks * sizeof(uint32_t));
            const bool has_pbmd_layout =
                compressed_buffer.size() >=
                (pbmd_prefix_bytes + total_chunk_bytes);
            if (!has_pbmd_layout) {
                all_pbmd_layout = false;
                break;
            }

            h_block_descs.push_back(
                RFORPBMDBlockDesc{compressed_buffer.data(),
                                  reinterpret_cast<int *>(
                                      const_cast<char *>(output_buffer.data())),
                                  static_cast<uint32_t>(num_chunks),
                                  tile_prefix});
            tile_prefix += static_cast<uint32_t>(num_chunks);
        }

        if (!all_pbmd_layout || h_block_descs.empty()) {
            LOG(ERROR)
                << "RFOR metadata-in-block mode is enabled, but PBMD layout "
                   "validation failed. Refusing legacy metadata-copy fallback.";
            return -1;
        }

        const size_t desc_bytes =
            sizeof(RFORPBMDBlockDesc) * h_block_descs.size();
        if (desc_bytes > m_device_decomp_workspace_size) {
            LOG(ERROR) << "Workspace too small for PBMD descriptors: need "
                       << desc_bytes << " bytes, have "
                       << m_device_decomp_workspace_size << " bytes";
            return -1;
        }

        auto *device_block_descs =
            static_cast<RFORPBMDBlockDesc *>(m_device_decomp_workspace);
        cudaError_t error = cudaMemcpyAsync(
            device_block_descs, h_block_descs.data(), desc_bytes,
            cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy PBMD descriptors: "
                       << cudaGetErrorString(error);
            return -1;
        }

        int decomp_result = launchRFORDecompressionKernelPBMD(
            device_block_descs, h_block_descs.size(), total_tiles,
            rfor_tile_size, rfor_shape.block_threads,
            rfor_shape.items_per_thread, stream);
        if (decomp_result < 0) {
            LOG(ERROR) << "RFOR PBMD decompression failed with error code: "
                       << decomp_result;
            if (synchronous) {
                syncAndDestroyStream(stream);
                last_batch_num_chunks = 0;
            }
            return decomp_result;
        }

        for (size_t i = 0; i < total_tiles; ++i) {
            m_host_compressed_chunk_sizes[i] = rfor_tile_bytes;
        }
        gpu_run(cudaMemcpyAsync(m_device_actual_uncompressed_bytes,
                                m_host_compressed_chunk_sizes,
                                sizeof(size_t) * total_tiles,
                                cudaMemcpyHostToDevice, stream));

        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
            // Byte count would overflow int above 2.15 GB; caller only checks sign.
            return 0;
        }

        return 0;
    }

    if (block_val_section_sizes.empty()) {
        LOG(ERROR) << "RFOR metadata-in-block fast path unavailable and "
                      "no val_section_sizes provided from JSON metadata";
        return -1;
    }

    // ✅ SIMPLIFIED: Use smaller, safer allocations
    std::vector<RFORTileMetadata*> h_metadata_ptrs;
    std::vector<uint32_t*> h_val_data_ptrs, h_rl_data_ptrs;
    std::vector<int*> h_output_ptrs;

    h_metadata_ptrs.reserve(total_tiles);
    h_val_data_ptrs.reserve(total_tiles);
    h_rl_data_ptrs.reserve(total_tiles);
    h_output_ptrs.reserve(total_tiles);


        for (size_t block_idx = 0; block_idx < block_chunk_sizes.size(); ++block_idx) {
            const auto& chunk_sizes = block_chunk_sizes[block_idx];
            const auto& compressed_buffer = block_compressed_buffers[block_idx];
            const auto& output_buffer = block_output_buffers[block_idx];
            const auto& val_section_sizes = block_val_section_sizes[block_idx];

            if (val_section_sizes.size() != chunk_sizes.size()) {
                LOG(ERROR) << "Mismatch between chunk_sizes and val_section_sizes for block " << block_idx;
                return -1;
            }

            const size_t num_chunks = chunk_sizes.size();
            const size_t required_output_bytes =
                num_chunks * rfor_tile_bytes;
            if (required_output_bytes > output_buffer.size()) {
                LOG(ERROR) << "Output buffer too small for RFOR block " << block_idx
                           << ": need " << required_output_bytes
                           << " bytes, have " << output_buffer.size();
                return -1;
            }

            size_t total_chunk_bytes = 0;
            for (uint32_t chunk_size : chunk_sizes) {
                total_chunk_bytes += chunk_size;
            }
            if (total_chunk_bytes > compressed_buffer.size()) {
                LOG(ERROR) << "Compressed buffer too small for RFOR block " << block_idx
                           << ": chunk bytes " << total_chunk_bytes
                           << " > block bytes " << compressed_buffer.size();
                return -1;
            }

            // RFOR has two on-disk layouts:
            // 1) legacy: [tile0(meta+val+rl)][tile1(meta+val+rl)]...
            // 2) PBMD:   [header + chunk_sizes + val_section_sizes + tile_metadata][chunk_data...]
            //
            // In in_gpu+RFOR preload mode we require metadata-in-block (PBMD).
            // Avoid size-based heuristics here; force deterministic PBMD parsing
            // when the flag is enabled.
            constexpr size_t kPbmdFixedHeaderBytes = 3 * sizeof(uint32_t);  // magic, version, chunk_count
            const size_t pbmd_prefix_bytes =
                kPbmdFixedHeaderBytes + (2 * num_chunks * sizeof(uint32_t)) +
                (num_chunks * sizeof(RFORTileMetadata));
            const bool has_pbmd_layout = FLAGS_use_metadata_in_block;
            if (has_pbmd_layout &&
                compressed_buffer.size() < (pbmd_prefix_bytes + total_chunk_bytes)) {
                LOG(ERROR) << "PBMD block too small for prefix+payload in block "
                           << block_idx << ": have " << compressed_buffer.size()
                           << ", need at least "
                           << (pbmd_prefix_bytes + total_chunk_bytes);
                return -1;
            }

            const char* chunk_data_base = nullptr;
            RFORTileMetadata* pbmd_metadata_base = nullptr;
            if (has_pbmd_layout) {
                const size_t metadata_offset =
                    kPbmdFixedHeaderBytes + (2 * num_chunks * sizeof(uint32_t));
                pbmd_metadata_base = reinterpret_cast<RFORTileMetadata*>(
                    const_cast<char*>(compressed_buffer.data() + metadata_offset));
                chunk_data_base = compressed_buffer.data() + pbmd_prefix_bytes;
            } else {
                // Legacy layout starts payload at block base. It may still have
                // trailing padding bytes; ignore anything past total_chunk_bytes.
                chunk_data_base = compressed_buffer.data();
            }

            size_t buffer_offset = 0;
            for (size_t tile_idx = 0; tile_idx < num_chunks; ++tile_idx) {
                if (buffer_offset + chunk_sizes[tile_idx] > total_chunk_bytes) {
                    LOG(ERROR) << "Buffer overflow in compressed RFOR payload for block "
                               << block_idx << " tile " << tile_idx;
                    return -1;
                }

                const char* chunk_start = chunk_data_base + buffer_offset;
                const size_t val_section_size = val_section_sizes[tile_idx];
                if (val_section_size < sizeof(RFORTileMetadata) ||
                    val_section_size > chunk_sizes[tile_idx]) {
                    LOG(ERROR) << "Invalid val_section_size " << val_section_size
                               << " for chunk size " << chunk_sizes[tile_idx]
                               << " block " << block_idx
                               << " tile " << tile_idx;
                    return -1;
                }

                RFORTileMetadata* metadata = nullptr;
                uint32_t* val_data = nullptr;
                if (has_pbmd_layout) {
                    metadata = pbmd_metadata_base + tile_idx;
                    // PBMD still stores per-chunk metadata at chunk start; decode
                    // value stream begins after that metadata.
                    val_data = reinterpret_cast<uint32_t*>(
                        const_cast<char*>(chunk_start + sizeof(RFORTileMetadata)));
                } else {
                    metadata = reinterpret_cast<RFORTileMetadata*>(
                        const_cast<char*>(chunk_start));
                    val_data = reinterpret_cast<uint32_t*>(
                        const_cast<char*>(chunk_start + sizeof(RFORTileMetadata)));
                }
                uint32_t* rl_data = reinterpret_cast<uint32_t*>(
                    const_cast<char*>(chunk_start + val_section_size));

                h_metadata_ptrs.push_back(metadata);
                h_val_data_ptrs.push_back(val_data);
                h_rl_data_ptrs.push_back(rl_data);
                buffer_offset += chunk_sizes[tile_idx];
            }

            // Set up output pointers for this block.
            for (size_t tile_idx = 0; tile_idx < num_chunks; ++tile_idx) {
                int* output_ptr = reinterpret_cast<int*>(
                    const_cast<char*>(output_buffer.data() +
                                      tile_idx * rfor_tile_bytes));
                h_output_ptrs.push_back(output_ptr);
            }
        }

        if (h_metadata_ptrs.size() != total_tiles ||
            h_val_data_ptrs.size() != total_tiles ||
            h_rl_data_ptrs.size() != total_tiles ||
            h_output_ptrs.size() != total_tiles) {
            LOG(ERROR) << "Pointer count mismatch";
            return -1;
        }

        size_t ptr_arrays_size = sizeof(void*) * total_tiles * 4; // metadata, val, rl, output

        // std::cout << "Required workspace: " << ptr_arrays_size
        //           << " bytes, Available: " << m_device_decomp_workspace_size << " bytes" << std::endl;

        if (ptr_arrays_size > m_device_decomp_workspace_size) {
            LOG(ERROR) << "Workspace too small for pointer arrays: need " << ptr_arrays_size
                       << " bytes, have " << m_device_decomp_workspace_size << " bytes";
            return -1;
        }


        // TODO: if has_pbmd_layout, metadata must not be sent from CPU
        char* workspace_ptr = static_cast<char*>(m_device_decomp_workspace);

        RFORTileMetadata** device_metadata_ptrs = reinterpret_cast<RFORTileMetadata**>(workspace_ptr);
        uint32_t** device_val_data_ptrs = reinterpret_cast<uint32_t**>(workspace_ptr + sizeof(void*) * total_tiles);
        uint32_t** device_rl_data_ptrs = reinterpret_cast<uint32_t**>(workspace_ptr + sizeof(void*) * total_tiles * 2);
        int** device_output_ptrs = reinterpret_cast<int**>(workspace_ptr + sizeof(void*) * total_tiles * 3);

        // std::cout << "Copying pointer arrays to GPU..." << std::endl;

        cudaError_t error = cudaMemcpyAsync(device_metadata_ptrs, h_metadata_ptrs.data(),
                            sizeof(RFORTileMetadata*) * total_tiles, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy metadata pointers: " << cudaGetErrorString(error);
            return -1;
        }

        error = cudaMemcpyAsync(device_val_data_ptrs, h_val_data_ptrs.data(),
                            sizeof(uint32_t*) * total_tiles, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy val_data pointers: " << cudaGetErrorString(error);
            return -1;
        }

        error = cudaMemcpyAsync(device_rl_data_ptrs, h_rl_data_ptrs.data(),
                            sizeof(uint32_t*) * total_tiles, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy rl_data pointers: " << cudaGetErrorString(error);
            return -1;
        }

        error = cudaMemcpyAsync(device_output_ptrs, h_output_ptrs.data(),
                            sizeof(int*) * total_tiles, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy output pointers: " << cudaGetErrorString(error);
            return -1;
        }

        // std::cout << "Launching RFOR kernel with " << total_tiles << " tiles..." << std::endl;


        int decomp_result = launchRFORDecompressionKernel(
            device_metadata_ptrs,
            device_val_data_ptrs,
            device_rl_data_ptrs,
            device_output_ptrs,
            total_tiles,
            rfor_tile_size,
            rfor_shape.block_threads,
            rfor_shape.items_per_thread,
            stream
        );

        // Check for errors
        if (decomp_result < 0) {
            LOG(ERROR) << "RFOR decompression failed with error code: " << decomp_result;
            if (synchronous) {
                syncAndDestroyStream(stream);
                last_batch_num_chunks = 0;
            }
            return decomp_result;
        }

        // Set actual uncompressed bytes
        for (size_t i = 0; i < total_tiles; ++i) {
            m_host_compressed_chunk_sizes[i] = rfor_tile_bytes;
        }
        gpu_run(cudaMemcpyAsync(m_device_actual_uncompressed_bytes, m_host_compressed_chunk_sizes,
                                sizeof(size_t) * total_tiles, cudaMemcpyHostToDevice, stream));

        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
            // Byte count would overflow int above 2.15 GB; caller only checks sign.
            return 0;
        }

        return 0;
}

[[nodiscard]] int GpuDecompressor::batch_decompress_flbp(
    const std::vector<std::span<const uint32_t>>& block_chunk_sizes,
    const std::vector<std::span<char>>& block_compressed_buffers,
    const std::vector<std::span<char>>& block_output_buffers,
    cudaStream_t stream,
    const std::vector<std::span<const uint32_t>>& block_val_section_sizes) {
    if (block_chunk_sizes.size() != block_compressed_buffers.size() ||
        block_chunk_sizes.size() != block_output_buffers.size()) {
        LOG(ERROR) << "FLBP batch input size mismatch: chunk_sizes="
                   << block_chunk_sizes.size() << " compressed="
                   << block_compressed_buffers.size() << " output="
                   << block_output_buffers.size();
        return -1;
    }

    if (!block_val_section_sizes.empty() &&
        block_val_section_sizes.size() != block_chunk_sizes.size()) {
        LOG(ERROR) << "FLBP val_section_sizes size mismatch: val_sections="
                   << block_val_section_sizes.size()
                   << " blocks=" << block_chunk_sizes.size();
        return -1;
    }

    const bool try_pbmd_fast_path =
        FLAGS_use_metadata_in_block && block_val_section_sizes.empty();
    if (block_val_section_sizes.empty() && !try_pbmd_fast_path) {
        LOG(ERROR) << "FLBP decompression requires aux metadata sizes "
                      "(val_section_sizes) from JSON metadata";
        return -1;
    }

    set_device_on_scope d(topology::getInstance().getGpus()[m_gpu_index_in_topo]);

    const bool synchronous = (stream == nullptr);
    if (synchronous) {
        stream = createNonBlockingStream();
    }

    size_t total_vectors = 0;
    for (const auto& chunk_sizes : block_chunk_sizes) {
        total_vectors += chunk_sizes.size();
    }
    if (total_vectors == 0) {
        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
        }
        return 0;
    }

    last_batch_num_chunks = total_vectors;

    if (try_pbmd_fast_path) {
        std::vector<FLBPPBMDBlockDesc> h_block_descs;
        h_block_descs.reserve(block_chunk_sizes.size());

        uint32_t vector_prefix = 0;
        constexpr size_t kPbmdFixedHeaderBytes =
            3 * sizeof(uint32_t);  // magic, version, chunk_count
        bool all_pbmd_layout = true;

        for (size_t block_idx = 0; block_idx < block_chunk_sizes.size();
             ++block_idx) {
            const auto& chunk_sizes = block_chunk_sizes[block_idx];
            const auto& compressed_buffer = block_compressed_buffers[block_idx];
            const auto& output_buffer = block_output_buffers[block_idx];

            const size_t num_chunks = chunk_sizes.size();
            const size_t required_output_bytes =
                num_chunks * static_cast<size_t>(FLBP_VECTOR_SIZE) * sizeof(int);
            if (required_output_bytes > output_buffer.size()) {
                LOG(ERROR) << "Output buffer too small for FLBP block "
                           << block_idx << ": need " << required_output_bytes
                           << " bytes, have " << output_buffer.size();
                return -1;
            }

            size_t total_chunk_bytes = 0;
            for (uint32_t chunk_size : chunk_sizes) {
                total_chunk_bytes += chunk_size;
            }

            // PBMD layout: [magic][algo][num_chunks][chunk_offsets...][payload chunks...]
            // Each chunk begins with its FLBPVectorMetadata header.
            const size_t pbmd_prefix_bytes =
                kPbmdFixedHeaderBytes + (num_chunks * sizeof(uint32_t));
            const bool has_pbmd_layout =
                compressed_buffer.size() >=
                (pbmd_prefix_bytes + total_chunk_bytes);
            if (!has_pbmd_layout) {
                all_pbmd_layout = false;
                break;
            }

            h_block_descs.push_back(
                FLBPPBMDBlockDesc{
                    compressed_buffer.data(),
                    reinterpret_cast<int*>(const_cast<char*>(output_buffer.data())),
                    static_cast<uint32_t>(num_chunks),
                    vector_prefix,
                });
            vector_prefix += static_cast<uint32_t>(num_chunks);
        }

        if (!all_pbmd_layout || h_block_descs.empty()) {
            LOG(ERROR)
                << "FLBP metadata-in-block mode is enabled, but PBMD layout "
                   "validation failed. Refusing legacy metadata-copy fallback.";
            return -1;
        }

        const size_t desc_bytes =
            sizeof(FLBPPBMDBlockDesc) * h_block_descs.size();
        if (desc_bytes > m_device_decomp_workspace_size) {
            LOG(ERROR) << "Workspace too small for FLBP PBMD descriptors: need "
                       << desc_bytes << " bytes, have "
                       << m_device_decomp_workspace_size << " bytes";
            return -1;
        }

        auto* device_block_descs =
            static_cast<FLBPPBMDBlockDesc*>(m_device_decomp_workspace);
        cudaError_t error = cudaMemcpyAsync(
            device_block_descs, h_block_descs.data(), desc_bytes,
            cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy FLBP PBMD descriptors: "
                       << cudaGetErrorString(error);
            return -1;
        }

        const auto flbp_shape =
            proteus::flbp::predecompress::resolveRuntimeShape(
                FLAGS_threads_per_block, FLAGS_items_per_thread);
        if (flbp_shape.tile_size == 0) {
            LOG(ERROR) << "Unsupported FLBP pre-decompress shape "
                       << FLAGS_threads_per_block << "x"
                       << FLAGS_items_per_thread
                       << "; require block_threads * items_per_thread to divide "
                       << proteus::flbp::predecompress::kVectorSize;
            return -1;
        }

        int decomp_result = launchFLBPDecompressionKernelPBMD(
            device_block_descs, h_block_descs.size(), total_vectors,
            flbp_shape.block_threads, flbp_shape.items_per_thread, stream);
        if (decomp_result < 0) {
            LOG(ERROR) << "FLBP PBMD decompression failed with error code: "
                       << decomp_result;
            if (synchronous) {
                syncAndDestroyStream(stream);
                last_batch_num_chunks = 0;
            }
            return decomp_result;
        }

        for (size_t i = 0; i < total_vectors; ++i) {
            m_host_compressed_chunk_sizes[i] = FLBP_VECTOR_SIZE * sizeof(int);
        }
        gpu_run(cudaMemcpyAsync(m_device_actual_uncompressed_bytes,
                                m_host_compressed_chunk_sizes,
                                sizeof(size_t) * total_vectors,
                                cudaMemcpyHostToDevice, stream));

        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
            // Byte count would overflow int above 2.15 GB; caller only checks sign.
            return 0;
        }

        return 0;
    }

    std::vector<FLBPVectorMetadata*> h_metadata_ptrs;
    std::vector<uint32_t*> h_packed_data_ptrs;
    std::vector<int*> h_output_ptrs;
    h_metadata_ptrs.reserve(total_vectors);
    h_packed_data_ptrs.reserve(total_vectors);
    h_output_ptrs.reserve(total_vectors);

    for (size_t block_idx = 0; block_idx < block_chunk_sizes.size(); ++block_idx) {
        const auto& chunk_sizes = block_chunk_sizes[block_idx];
        const auto& compressed_buffer = block_compressed_buffers[block_idx];
        const auto& output_buffer = block_output_buffers[block_idx];

        const size_t num_chunks = chunk_sizes.size();
        const size_t required_output_bytes =
            num_chunks * static_cast<size_t>(FLBP_VECTOR_SIZE) * sizeof(int);
        if (required_output_bytes > output_buffer.size()) {
            LOG(ERROR) << "Output buffer too small for FLBP block " << block_idx
                       << ": need " << required_output_bytes
                       << " bytes, have " << output_buffer.size();
            return -1;
        }

        size_t total_chunk_bytes = 0;
        for (uint32_t chunk_size : chunk_sizes) {
            total_chunk_bytes += chunk_size;
        }

        const size_t kPbmdFixedHeaderBytes = 3 * sizeof(uint32_t);  // magic, version, chunk_count
        const size_t pbmd_metadata_offset =
            kPbmdFixedHeaderBytes + (2 * num_chunks * sizeof(uint32_t));
        const size_t pbmd_prefix_bytes =
            pbmd_metadata_offset + (num_chunks * sizeof(FLBPVectorMetadata));
        const bool has_pbmd_layout =
            try_pbmd_fast_path &&
            compressed_buffer.size() >= (pbmd_prefix_bytes + total_chunk_bytes);
        if (try_pbmd_fast_path && !has_pbmd_layout) {
            LOG(ERROR) << "FLBP PBMD block too small for prefix+payload in block "
                       << block_idx << ": have " << compressed_buffer.size()
                       << ", need at least "
                       << (pbmd_prefix_bytes + total_chunk_bytes);
            return -1;
        }

        if (!has_pbmd_layout) {
            CHECK_LT(block_idx, block_val_section_sizes.size())
                << "Missing val_section_sizes for FLBP block " << block_idx;
            CHECK_EQ(chunk_sizes.size(), block_val_section_sizes[block_idx].size())
                << "Mismatch between chunk_sizes and val_section_sizes for FLBP block "
                << block_idx;
        }

        if (total_chunk_bytes >
            (has_pbmd_layout ? compressed_buffer.size() - pbmd_prefix_bytes
                             : compressed_buffer.size())) {
            LOG(ERROR) << "Compressed buffer too small for FLBP block "
                       << block_idx << ": chunk bytes " << total_chunk_bytes
                       << " exceeds payload bytes available";
            return -1;
        }

        char* chunk_data_base =
            const_cast<char*>(compressed_buffer.data()) +
            (has_pbmd_layout ? pbmd_prefix_bytes : 0);
        auto* pbmd_metadata_base = has_pbmd_layout
                                       ? reinterpret_cast<FLBPVectorMetadata*>(
                                             const_cast<char*>(compressed_buffer.data() +
                                                               pbmd_metadata_offset))
                                       : nullptr;

        size_t buffer_offset = 0;
        for (size_t vector_idx = 0; vector_idx < num_chunks; ++vector_idx) {
            if (buffer_offset + chunk_sizes[vector_idx] > total_chunk_bytes) {
                LOG(ERROR) << "Buffer overflow in compressed FLBP payload for block "
                           << block_idx << " vector " << vector_idx;
                return -1;
            }

            char* chunk_start = chunk_data_base + buffer_offset;
            const uint32_t aux_size =
                has_pbmd_layout
                    ? static_cast<uint32_t>(sizeof(FLBPVectorMetadata))
                    : block_val_section_sizes[block_idx][vector_idx];
            if (aux_size < sizeof(FLBPVectorMetadata) ||
                aux_size > chunk_sizes[vector_idx]) {
                LOG(ERROR) << "Invalid FLBP aux size " << aux_size
                           << " for chunk size " << chunk_sizes[vector_idx]
                           << " block " << block_idx
                           << " vector " << vector_idx;
                return -1;
            }

            FLBPVectorMetadata* metadata = has_pbmd_layout
                                               ? (pbmd_metadata_base + vector_idx)
                                               : reinterpret_cast<FLBPVectorMetadata*>(chunk_start);
            uint32_t* packed_data =
                reinterpret_cast<uint32_t*>(chunk_start + aux_size);
            int* output_ptr = reinterpret_cast<int*>(
                const_cast<char*>(output_buffer.data()) +
                vector_idx * FLBP_VECTOR_SIZE * sizeof(int));

            h_metadata_ptrs.push_back(metadata);
            h_packed_data_ptrs.push_back(packed_data);
            h_output_ptrs.push_back(output_ptr);
            buffer_offset += chunk_sizes[vector_idx];
        }
    }

    if (h_metadata_ptrs.size() != total_vectors ||
        h_packed_data_ptrs.size() != total_vectors ||
        h_output_ptrs.size() != total_vectors) {
        LOG(ERROR) << "FLBP pointer count mismatch";
        return -1;
    }

    const size_t ptr_arrays_size = sizeof(void*) * total_vectors * 3;
    if (ptr_arrays_size > m_device_decomp_workspace_size) {
        LOG(ERROR) << "Workspace too small for FLBP pointer arrays: need "
                   << ptr_arrays_size << " bytes, have "
                   << m_device_decomp_workspace_size << " bytes";
        return -1;
    }

    char* workspace_ptr = static_cast<char*>(m_device_decomp_workspace);
    auto** device_metadata_ptrs =
        reinterpret_cast<FLBPVectorMetadata**>(workspace_ptr);
    auto** device_packed_data_ptrs =
        reinterpret_cast<uint32_t**>(workspace_ptr +
                                     sizeof(void*) * total_vectors);
    auto** device_output_ptrs =
        reinterpret_cast<int**>(workspace_ptr +
                                sizeof(void*) * total_vectors * 2);

    cudaError_t error = cudaMemcpyAsync(
        device_metadata_ptrs, h_metadata_ptrs.data(),
        sizeof(FLBPVectorMetadata*) * total_vectors, cudaMemcpyHostToDevice,
        stream);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to copy FLBP metadata pointers: "
                   << cudaGetErrorString(error);
        return -1;
    }

    error = cudaMemcpyAsync(device_packed_data_ptrs, h_packed_data_ptrs.data(),
                            sizeof(uint32_t*) * total_vectors,
                            cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to copy FLBP packed-data pointers: "
                   << cudaGetErrorString(error);
        return -1;
    }

    error = cudaMemcpyAsync(device_output_ptrs, h_output_ptrs.data(),
                            sizeof(int*) * total_vectors, cudaMemcpyHostToDevice,
                            stream);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to copy FLBP output pointers: "
                   << cudaGetErrorString(error);
        return -1;
    }

    const auto flbp_shape =
        proteus::flbp::predecompress::resolveRuntimeShape(
            FLAGS_threads_per_block, FLAGS_items_per_thread);
    if (flbp_shape.tile_size == 0) {
        LOG(ERROR) << "Unsupported FLBP pre-decompress shape "
                   << FLAGS_threads_per_block << "x"
                   << FLAGS_items_per_thread
                   << "; require block_threads * items_per_thread to divide "
                   << proteus::flbp::predecompress::kVectorSize;
        return -1;
    }

    int decomp_result = launchFLBPDecompressionKernel(
        device_metadata_ptrs, device_packed_data_ptrs, device_output_ptrs,
        total_vectors, flbp_shape.block_threads, flbp_shape.items_per_thread,
        stream);
    if (decomp_result < 0) {
        LOG(ERROR) << "FLBP decompression failed with error code: "
                   << decomp_result;
        if (synchronous) {
            syncAndDestroyStream(stream);
            last_batch_num_chunks = 0;
        }
        return decomp_result;
    }

    for (size_t i = 0; i < total_vectors; ++i) {
        m_host_compressed_chunk_sizes[i] = FLBP_VECTOR_SIZE * sizeof(int);
    }
    gpu_run(cudaMemcpyAsync(m_device_actual_uncompressed_bytes,
                            m_host_compressed_chunk_sizes,
                            sizeof(size_t) * total_vectors,
                            cudaMemcpyHostToDevice, stream));

    if (synchronous) {
        syncAndDestroyStream(stream);
        last_batch_num_chunks = 0;
        // Byte count would overflow int above 2.15 GB; caller only checks sign.
        return 0;
    }

    return 0;
}


int GpuDecompressor::decompress_gpu(const size_t batch_size,
                                    void **device_uncompressed_ptrs,
                                    void **device_compressed_ptrs,
                                    size_t *device_compressed_bytes,
                                    size_t output_buffer_size,
                                    cudaStream_t stream) {
  // to ensure kernels launched on the correct GPU
  set_device_on_scope d(topology::getInstance().getGpus()[m_gpu_index_in_topo]);

  const bool synchronous = (stream == nullptr);
  if (synchronous) {
    stream = createNonBlockingStream();
  }
  DCHECK_GT(batch_size, 0) << "Cannot have an empty batch.";
  last_batch_num_chunks = batch_size;

  nvcompStatus_t status;

  size_t *device_uncompressed_bytes =
      get_device_uncompressed_chunk_sizes(batch_size, stream);

  // Run decompression
  switch (m_comp_algo) {
    case CompressionAlgorithm::LZ4:
      LOG(INFO) << "WE ARE USING LZ4========================================================================";
      status = nvcompBatchedLZ4DecompressAsync(
          device_compressed_ptrs, device_compressed_bytes,
          device_uncompressed_bytes, m_device_actual_uncompressed_bytes,
          batch_size, m_device_decomp_workspace,
          m_device_decomp_workspace_size, device_uncompressed_ptrs,
          m_device_status_ptrs, stream);
      break;
    case CompressionAlgorithm::CASCADED:
      // LOG(INFO) << "WE ARE USING CASCADED========================================================================";
      status = nvcompBatchedCascadedDecompressAsync(
          device_compressed_ptrs, device_compressed_bytes,
          device_uncompressed_bytes, m_device_actual_uncompressed_bytes,
          batch_size, m_device_decomp_workspace,
          m_device_decomp_workspace_size, device_uncompressed_ptrs,
          m_device_status_ptrs, stream);
      break;
    case CompressionAlgorithm::GDEFLATE:
      LOG(INFO) << "WE ARE USING GDEFLATE========================================================================";
      status = nvcompBatchedGdeflateDecompressAsync(
          device_compressed_ptrs, device_compressed_bytes,
          device_uncompressed_bytes, m_device_actual_uncompressed_bytes,
          batch_size, m_device_decomp_workspace,
          m_device_decomp_workspace_size, device_uncompressed_ptrs,
          m_device_status_ptrs, stream);
      if (status != nvcompSuccess) {
        LOG(ERROR) << "decompress_gpu failed on nvcompBatchedGdeflateDecompressAsync with "
                   << magic_enum::enum_name(status);
        return -status;
      }
      break;
    case CompressionAlgorithm::RFOR:
      // RFOR decompression is handled in batch_decompress_block_gpu
      LOG(ERROR) << "RFOR decompression should not reach decompress_gpu method";
      return -1;
    case CompressionAlgorithm::FLBP:
      // FLBP decompression is handled in batch_decompress_block_gpu
      LOG(ERROR) << "FLBP decompression should not reach decompress_gpu method";
      return -1;
    default:
      throw std::runtime_error("Unsupported compression algorithm");
  }
  if (status != nvcompSuccess && m_comp_algo != CompressionAlgorithm::RFOR &&
      m_comp_algo != CompressionAlgorithm::FLBP) {
    LOG(ERROR)
        << "decompress_gpu failed on nvcompBatchedLZ4DecompressAsync with "
        << magic_enum::enum_name(status);
    return -status;
  }

  if (synchronous) {
    syncAndDestroyStream(stream);
    // Assume device_status_ptrs is an array of nvcompStatus_t on the device
    std::vector<nvcompStatus_t> host_status_ptrs(batch_size);

    // Copy the status from device to host
    gpu_run(cudaMemcpy(host_status_ptrs.data(), m_device_status_ptrs,
                       batch_size * sizeof(nvcompStatus_t),
                       cudaMemcpyDeviceToHost));

    // Check the status of each decompression task
    for (size_t i = 0; i < batch_size; i++) {
      if (host_status_ptrs[i] != nvcompSuccess) {
        LOG(ERROR) << "GPU decompression failed for chunk " << i
                   << " with status "
                   << magic_enum::enum_name(host_status_ptrs[i]);
        return -host_status_ptrs[i];
      }
    }

    std::vector<size_t> host_actual_uncompressed_bytes(batch_size);
    gpu_run(cudaMemcpy(host_actual_uncompressed_bytes.data(),
                       m_device_actual_uncompressed_bytes,
                       batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t decompressed_size = 0;
    for (size_t i = 0; i < batch_size; i++) {
      decompressed_size += host_actual_uncompressed_bytes[i];
    }
    DCHECK_LE(decompressed_size, output_buffer_size);
    last_batch_num_chunks = 0;  // we use last_batch_num_chunks to check if we
                                // have an outstanding asynchronous call
    return decompressed_size;
  }

  return (int)0;
}

/* Decompress a batch of compressed data chunks on the GPU. */
[[nodiscard]] int GpuDecompressor::batch_decompress_block_gpu(
    const std::vector<std::vector<uint32_t>> &block_chunk_sizes,
    const std::vector<std::span<char>> &block_compressed_buffers,
    const std::vector<std::span<char>> &block_output_buffers,
    cudaStream_t stream,
    const std::vector<std::vector<uint32_t>> &block_val_section_sizes) {
  std::vector<std::span<const uint32_t>> chunk_size_spans;
  chunk_size_spans.reserve(block_chunk_sizes.size());
  for (const auto &chunk_sizes : block_chunk_sizes) {
    chunk_size_spans.emplace_back(chunk_sizes.data(), chunk_sizes.size());
  }

  std::vector<std::span<const uint32_t>> val_section_size_spans;
  val_section_size_spans.reserve(block_val_section_sizes.size());
  for (const auto &val_sizes : block_val_section_sizes) {
    val_section_size_spans.emplace_back(val_sizes.data(), val_sizes.size());
  }

  return batch_decompress_block_gpu(chunk_size_spans, block_compressed_buffers,
                                    block_output_buffers, stream,
                                    val_section_size_spans);
}

/* Decompress a batch of compressed data chunks on the GPU (span metadata). */
[[nodiscard]] int GpuDecompressor::batch_decompress_block_gpu(
    const std::vector<std::span<const uint32_t>> &block_chunk_sizes,
    const std::vector<std::span<char>> &block_compressed_buffers,
    const std::vector<std::span<char>> &block_output_buffers,
    cudaStream_t stream,
    const std::vector<std::span<const uint32_t>> &block_val_section_sizes) {

  DCHECK_EQ(block_compressed_buffers.size(), block_chunk_sizes.size());
  DCHECK_EQ(block_output_buffers.size(), block_chunk_sizes.size());

  // Handle RFOR either via JSON val_section_sizes (legacy) or via PBMD
  // metadata-in-block fast path when enabled.
  if (m_comp_algo == CompressionAlgorithm::RFOR) {
    if (!block_val_section_sizes.empty() || FLAGS_use_metadata_in_block) {
      return batch_decompress_rfor(block_chunk_sizes, block_compressed_buffers,
                                 block_output_buffers, stream, block_val_section_sizes);
    } else {
      LOG(ERROR) << "RFOR algorithm selected but no val_section_sizes provided from JSON metadata";
      return -1;
    }
  }
  if (m_comp_algo == CompressionAlgorithm::FLBP) {
    if (!block_val_section_sizes.empty() || FLAGS_use_metadata_in_block) {
      return batch_decompress_flbp(block_chunk_sizes, block_compressed_buffers,
                                   block_output_buffers, stream,
                                   block_val_section_sizes);
    } else {
      LOG(ERROR) << "FLBP algorithm selected but no val_section_sizes provided "
                    "from JSON metadata";
      return -1;
    }
  }

  // Handle other compression algorithms (LZ4, CASCADED, GDEFLATE)
  size_t batch_size = 0;
  for (const auto &chunk_sizes : block_chunk_sizes) {
    batch_size += chunk_sizes.size();
  }

  size_t output_buffer_size = 0;
  for (const auto &output_buffer : block_output_buffers) {
    output_buffer_size += output_buffer.size();
  }

  void **device_uncompressed_ptrs = get_device_uncompressed_ptrs(
      block_output_buffers, block_chunk_sizes, batch_size, stream);

  void **device_compressed_ptrs = get_device_compressed_ptrs(
      block_compressed_buffers, block_chunk_sizes, batch_size, stream);

  size_t *device_compressed_bytes =
      get_device_compressed_bytes(block_chunk_sizes, batch_size, stream);

  return decompress_gpu(batch_size, device_uncompressed_ptrs,
                        device_compressed_ptrs, device_compressed_bytes,
                        output_buffer_size, stream);
}

/* Assuming that compressed_buffer and output_buffer are GPU resident.  */
[[nodiscard]] int GpuDecompressor::decompress_block_gpu(
    std::vector<uint32_t> &chunk_sizes, std::span<char> compressed_buffer,
    std::span<char> output_buffer, cudaStream_t stream) {
  DCHECK_GT(chunk_sizes.size(), 0) << "Cannot have an empty batch.";
  std::vector<std::span<const uint32_t>> block_chunk_sizes(1);
  block_chunk_sizes[0] =
      std::span<const uint32_t>(chunk_sizes.data(), chunk_sizes.size());

  std::vector<std::span<char>> block_compressed_buffers(1);
  block_compressed_buffers[0] = compressed_buffer;

  std::vector<std::span<char>> block_output_buffers(1);
  block_output_buffers[0] = output_buffer;

  return batch_decompress_block_gpu(block_chunk_sizes, block_compressed_buffers,
                                    block_output_buffers, stream, {});
}


//
// [[nodiscard]] int decompress_rfor_block(const std::vector<uint32_t>& chunk_sizes,
//                                         std::span<const char> compressed_buffer,
//                                         std::span<char> output_buffer,
//                                         int max_decomp_chunk_size,
//                                         const std::vector<uint32_t>& val_section_sizes) {
//
//     if (chunk_sizes.empty()) {
//         LOG(ERROR) << "Empty chunk_sizes vector";
//         return -1;
//     }
//
//     if (val_section_sizes.empty()) {
//         LOG(ERROR) << "RFOR decompression requires val_section_sizes from JSON metadata";
//         return -1;
//     }
//
//     if (chunk_sizes.size() != val_section_sizes.size()) {
//         LOG(ERROR) << "Mismatch between chunk_sizes and val_section_sizes";
//         return -1;
//     }
//
//     size_t buffer_offset = 0;
//     size_t output_offset = 0;
//
//     for (size_t i = 0; i < chunk_sizes.size(); ++i) {
//         const char* chunk_start = compressed_buffer.data() + buffer_offset;
//         const RFORTileMetadata* metadata = reinterpret_cast<const RFORTileMetadata*>(chunk_start);
//
//         // ✅ KEY FIX: Use JSON metadata val_section_sizes directly
//         size_t val_section_size = val_section_sizes[i];
//         size_t rl_data_offset = val_section_size;
//
//         // ✅ Verify chunk size consistency with JSON metadata
//         size_t rl_data_size = chunk_sizes[i] - val_section_size;
//         size_t expected_chunk_size = val_section_size + rl_data_size;
//         if (chunk_sizes[i] != expected_chunk_size) {
//             LOG(ERROR) << "Chunk size mismatch with JSON metadata: expected " << expected_chunk_size
//                        << ", got " << chunk_sizes[i];
//             return -1;
//         }
//
//         const uint32_t* val_data = reinterpret_cast<const uint32_t*>(chunk_start + sizeof(RFORTileMetadata));
//         const uint32_t* rl_data = reinterpret_cast<const uint32_t*>(chunk_start + rl_data_offset);
//
//         int* output_tile = reinterpret_cast<int*>(output_buffer.data() + output_offset);
//         int result = decompressRFORTile(metadata, val_data, rl_data, output_tile);
//
//         if (result < 0) {
//             LOG(ERROR) << "Failed to decompress RFOR tile " << i;
//             return result;
//         }
//
//         buffer_offset += chunk_sizes[i];
//         output_offset += RFOR_TILE_SIZE * sizeof(int);
//     }
//
//     return static_cast<int>(output_offset);
// }

int decompress_block(
    const std::vector<uint32_t> &chunk_sizes,
    const std::span<const char> compressed_buffer,
    std::span<char> output_buffer, int max_decomp_chunk_size) {
  int comp_idx = 0;
  int decomp_idx = 0;
  for (const auto &chunk_size : chunk_sizes) {
    auto decompressed_size = LZ4_decompress_safe(
        &compressed_buffer[comp_idx], &output_buffer[decomp_idx],
        (int)chunk_size, max_decomp_chunk_size);
    if (decompressed_size < 0) [[unlikely]] {
      LOG(ERROR) << "failed to decompress block. LZ4 return value: "
                 << decompressed_size << " for chunk size: " << chunk_size
                 << " at comp index: " << comp_idx
                 << " decomp index: " << decomp_idx;
      return decompressed_size;
    }

    comp_idx += chunk_size;
    decomp_idx += decompressed_size;
    DCHECK_LE(decomp_idx, output_buffer.size());
  }
  return decomp_idx;
}

void debug_cpu_decompress_first_tile(const std::vector<std::vector<uint32_t>>& block_chunk_sizes,
                                     const std::vector<std::span<char>>& block_compressed_buffers,
                                     const std::vector<std::vector<uint32_t>>& block_val_section_sizes) {

    if (block_compressed_buffers.empty() || block_val_section_sizes.empty()) {
        std::cout << "No data to decompress for CPU reference" << std::endl;
        return;
    }

    std::cout << "\n=== CPU REFERENCE DECOMPRESSION - FIRST TILE ===" << std::endl;

    const auto& chunk_sizes = block_chunk_sizes[0];
    const auto& compressed_buffer = block_compressed_buffers[0];
    const auto& val_section_sizes = block_val_section_sizes[0];

    if (chunk_sizes.empty()) {
        std::cout << "No chunks to decompress" << std::endl;
        return;
    }

    // Decompress first tile only
    const char* chunk_start = compressed_buffer.data();

    // Parse metadata
    const RFORTileMetadata* metadata = reinterpret_cast<const RFORTileMetadata*>(chunk_start);

    std::cout << "CPU: Metadata - val_ref=" << metadata->val_reference
              << ", val_bw=" << metadata->val_bitwidth
              << ", val_count=" << metadata->val_count << std::endl;
    std::cout << "CPU: Metadata - rl_ref=" << metadata->rl_reference
              << ", rl_bw=" << metadata->rl_bitwidth
              << ", rl_count=" << metadata->rl_count << std::endl;

    // Use val_section_sizes from JSON metadata
    size_t val_section_size = val_section_sizes[0];
    size_t rl_data_offset = val_section_size;

    const uint32_t* val_data = reinterpret_cast<const uint32_t*>(chunk_start + sizeof(RFORTileMetadata));
    const uint32_t* rl_data = reinterpret_cast<const uint32_t*>(chunk_start + rl_data_offset);

    // Calculate available words for bounds checking
    uint32_t val_data_size = val_section_size - sizeof(RFORTileMetadata);
    uint32_t val_available_words = val_data_size / sizeof(uint32_t);
    uint32_t rl_data_size = chunk_sizes[0] - val_section_size;
    uint32_t rl_available_words = rl_data_size / sizeof(uint32_t);

    std::cout << "CPU: val_available_words=" << val_available_words
              << ", rl_available_words=" << rl_available_words << std::endl;

    // Validate metadata
    if (metadata->val_count != metadata->rl_count || metadata->val_count == 0 ||
        metadata->val_count > RFOR_TILE_SIZE) {
        std::cout << "CPU: Invalid metadata detected!" << std::endl;
        return;
    }

    // Decode values using safer approach
    std::vector<int> values(metadata->val_count);
    for (uint32_t i = 0; i < metadata->val_count; ++i) {
        if (metadata->val_bitwidth == 0) {
            values[i] = metadata->val_reference;
            continue;
        }

        uint32_t start_bitindex = (metadata->val_bitwidth * i);
        uint32_t start_intindex = (start_bitindex >> 5);
        uint32_t bit_offset_in_word = start_bitindex & 31;

        if (start_intindex >= val_available_words) {
            std::cout << "CPU: val bounds error at i=" << i << ", start_intindex=" << start_intindex << std::endl;
            values[i] = metadata->val_reference;
            continue;
        }

        uint32_t bits_in_first_word = 32 - bit_offset_in_word;
        bool needs_two_words = metadata->val_bitwidth > bits_in_first_word;

        if (needs_two_words && start_intindex + 1 >= val_available_words) {
            // Can only read from first word
            uint32_t element = (val_data[start_intindex] >> bit_offset_in_word) & ((1U << metadata->val_bitwidth) - 1U);
            values[i] = element + metadata->val_reference;
        } else if (needs_two_words) {
            // Read across word boundary
            unsigned long long element_block = (((unsigned long long)val_data[start_intindex + 1]) << 32) | val_data[start_intindex];
            uint32_t element = (element_block >> bit_offset_in_word) & ((1ULL << metadata->val_bitwidth) - 1ULL);
            values[i] = element + metadata->val_reference;
        } else {
            // Single word read
            uint32_t element = (val_data[start_intindex] >> bit_offset_in_word) & ((1U << metadata->val_bitwidth) - 1U);
            values[i] = element + metadata->val_reference;
        }
    }

    // Decode run lengths using safer approach
    std::vector<int> run_lengths(metadata->rl_count);
    for (uint32_t i = 0; i < metadata->rl_count; ++i) {
        if (metadata->rl_bitwidth == 0) {
            run_lengths[i] = metadata->rl_reference;
            continue;
        }

        uint32_t start_bitindex = (metadata->rl_bitwidth * i);
        uint32_t start_intindex = (start_bitindex >> 5);
        uint32_t bit_offset_in_word = start_bitindex & 31;

        if (start_intindex >= rl_available_words) {
            std::cout << "CPU: rl bounds error at i=" << i << ", start_intindex=" << start_intindex << std::endl;
            run_lengths[i] = metadata->rl_reference;
            continue;
        }

        uint32_t bits_in_first_word = 32 - bit_offset_in_word;
        bool needs_two_words = metadata->rl_bitwidth > bits_in_first_word;

        if (needs_two_words && start_intindex + 1 >= rl_available_words) {
            uint32_t element = (rl_data[start_intindex] >> bit_offset_in_word) & ((1U << metadata->rl_bitwidth) - 1U);
            run_lengths[i] = element + metadata->rl_reference;
        } else if (needs_two_words) {
            unsigned long long element_block = (((unsigned long long)rl_data[start_intindex + 1]) << 32) | rl_data[start_intindex];
            uint32_t element = (element_block >> bit_offset_in_word) & ((1ULL << metadata->rl_bitwidth) - 1ULL);
            run_lengths[i] = element + metadata->rl_reference;
        } else {
            uint32_t element = (rl_data[start_intindex] >> bit_offset_in_word) & ((1U << metadata->rl_bitwidth) - 1U);
            run_lengths[i] = element + metadata->rl_reference;
        }
    }

    // Print decoded values and run lengths
    std::cout << "CPU: First 10 decoded values: ";
    for (int i = 0; i < std::min(10u, metadata->val_count); i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "CPU: First 10 decoded run_lengths: ";
    for (int i = 0; i < std::min(10u, metadata->rl_count); i++) {
        std::cout << run_lengths[i] << " ";
    }
    std::cout << std::endl;

    // Reconstruct original data
    std::vector<int> cpu_output(RFOR_TILE_SIZE, 0);
    int output_pos = 0;
    for (uint32_t i = 0; i < metadata->val_count && output_pos < RFOR_TILE_SIZE; ++i) {
        int value = values[i];
        int run_length = run_lengths[i];

        if (run_length <= 0) {
            std::cout << "CPU: Invalid run_length " << run_length << " at position " << i << std::endl;
            continue;
        }

        for (int j = 0; j < run_length && output_pos < RFOR_TILE_SIZE; ++j) {
            cpu_output[output_pos] = value;
            output_pos++;
        }
    }

    std::cout << "CPU: Reconstructed " << output_pos << " elements" << std::endl;
    std::cout << "CPU: First 40 output elements: ";
    for (int i = 0; i < 40; i++) {
        std::cout << cpu_output[i];
        if (i < 39) std::cout << ", ";
    }
    std::cout << std::endl;

    // Calculate statistics
    int min_val = cpu_output[0], max_val = cpu_output[0];
    long long sum = 0;
    for (int i = 0; i < RFOR_TILE_SIZE; i++) {
        min_val = std::min(min_val, cpu_output[i]);
        max_val = std::max(max_val, cpu_output[i]);
        sum += cpu_output[i];
    }
    std::cout << "CPU: Tile stats - min=" << min_val << ", max=" << max_val
              << ", sum=" << sum << ", avg=" << (sum / RFOR_TILE_SIZE) << std::endl;
}

[[nodiscard]] int decompress_block(const std::vector<uint32_t> &chunk_sizes,
                                   std::span<char> compressed_buffer,
                                   std::span<char> output_buffer,
                                   int max_decomp_chunk_size) {
  return decompress_block(
      chunk_sizes,
      std::span<const char>(compressed_buffer.data(), compressed_buffer.size()),
      output_buffer, max_decomp_chunk_size);
}
