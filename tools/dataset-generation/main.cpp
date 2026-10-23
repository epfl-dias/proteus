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

#include <gflags/gflags.h>
#include <lz4.h>
#include <lz4hc.h>
#include <nvcomp.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/snappy.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <olap/operators/decompression-config.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/glog.hpp>
#include <span>
#include <unordered_set>
#include <vector>

#include "platform/util/timing.hpp"
#include "flbp-compression.hpp"
#include "rfor-compression.hpp"

enum CompressionAlgorithm { LZ4, CASCADED, GDEFLATE, SNAPPY, RFOR, FLBP};

// #define NDEBUG

namespace fs = std::filesystem;
[[nodiscard]] int decompress_block(const std::vector<uint32_t>& chunk_sizes,
                                   std::span<char> compressed_buffer,
                                   std::span<char> output_buffer,
                                   int max_decomp_chunk_size) {
  int comp_idx = 0;
  int decomp_idx = 0;
  for (const auto& chunk_size : chunk_sizes) {
    auto decompressed_size = LZ4_decompress_safe(
        &compressed_buffer[comp_idx], &output_buffer[decomp_idx], chunk_size,
        max_decomp_chunk_size);
    if (decompressed_size < 0) [[unlikely]] {
      LOG(ERROR) << "failed to decompress block. LZ4 return value: "
                 << decompressed_size << " for chunk size: " << chunk_size
                 << " at comp index: " << comp_idx
                 << " decomp index: " << decomp_idx;
      return decompressed_size;
    }

    comp_idx += chunk_size;
    decomp_idx += decompressed_size;
  }
  DCHECK_LE(decomp_idx, output_buffer.size());
  return 0;
}

DEFINE_string(input_file, "", "Path to the input file");

DEFINE_string(input_directory, "", "Path to the input dataset");
DEFINE_string(columns_file, "",
              "Optional path to a file with one column name per line. If empty "
              "(default), the column set is read from <input_directory>/"
              "catalog.json -- the partitioner carries no dataset-specific "
              "hardcoded column list.");
DECLARE_int32(
    num_files);  // shouldn't really need this, but it silences a warning
DEFINE_int32(num_files, 2, "Number of files to split the dataset into");
DECLARE_bool(
    compress_data);  // shouldn't really need this, but it silences a warning
DEFINE_bool(compress_data, false, "Compress data with \" compression_algorithm \"");
DECLARE_int32(compress_level);
DEFINE_int32(compress_level, -1, "Compression level");
DEFINE_string(output_directories, "",
              "Comma delimited list of output directories");
DECLARE_string(compression_algorithm);
DEFINE_string(compression_algorithm, "none", "Compression algorithm");

DECLARE_bool(metadata_in_block);
DEFINE_bool(metadata_in_block, false, "If false (default), If true it writes down compression metadata directly in the blocks.");

DECLARE_int32(rfor_tile_size);
DEFINE_int32(
    rfor_tile_size, RFOR_TILE_SIZE,
    "RFOR dataset tile size in elements. Should match runtime decode contract "
    "(typically 512 for CB2T in-GPU decode).");

DECLARE_bool(precompute_rl);

// Align to 4KiB for GDS
// https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#cufile-bufregister-fileread-filewrite
// 512 is sufficient of IO_DIRECT
static constexpr size_t kAlignment = 4 * 1024;

std::vector<std::vector<char>> splitFileIntoBlocks(const fs::path& filePath,
                                                   const size_t blockSize) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<std::vector<char>> blocks;
  blocks.reserve((fileSize + blockSize - 1) / blockSize);

  while (fileSize > blockSize) {
    std::vector<char> block(blockSize);
    if (!file.read(block.data(), blockSize)) {
      LOG(FATAL) << "Failed to read block from file";
    }
    blocks.push_back(std::move(block));
    fileSize -= blockSize;
  }

  if (fileSize > 0) {
    CHECK_EQ(fileSize % 4, 0)
        << "sub block size remainder is not a multiple of 4 bytes: " << fileSize
        << "for file: " << filePath;
    std::vector<char> block(fileSize);
    if (file.read(block.data(), fileSize)) {
      blocks.push_back(std::move(block));
    } else {
      LOG(FATAL) << "Failed to read last block from file. " << filePath;
    }
  }

  size_t total_size_bytes = 0;
  for (const auto& block : blocks) {
    total_size_bytes += block.size();
  }
  CHECK_EQ(total_size_bytes, fs::file_size(filePath));
  //  fsize = total_size_bytes;

  return blocks;
}

/**
 * @param blocks vector of uncompressed blocks to write to files
 * @param output_directories A vector of directories. The file is split into
 * size() parts and one part is written to each directory
 * @param base_file_Name The base file name to write to. The file name will be
 * appended with _i_numFiles where i is the index of the file and numFiles is
 * the total number of files
 */
void writeUncompressedChunksToFiles(
    const std::vector<std::vector<char>>& blocks,
    const std::vector<fs::path> output_directories,
    const std::string& base_file_Name) {
  // ceiling division
  const int numFiles = output_directories.size();
  const int blocksPerFile = (blocks.size() + numFiles - 1) / numFiles;

  for (int i = 0; i < numFiles; ++i) {
    const fs::path filePath =
        output_directories[i] / (base_file_Name + "_" + std::to_string(i) +
                                 "_" + std::to_string(numFiles));
    std::ofstream file(filePath, std::ios::binary);

    rapidjson::Document metadata;
    metadata.SetObject();
    rapidjson::Document::AllocatorType& allocator = metadata.GetAllocator();

    metadata.AddMember("data_format", "UNCOMPRESSED", allocator);
    metadata.AddMember("_comment", "generated with proteus-generate-cb2t-dataset",
                       allocator);
    metadata.AddMember("data_file",
                       rapidjson::Value(filePath.filename().c_str(), allocator),
                       allocator);
    metadata.AddMember("num_blocks", 0, allocator);

    rapidjson::Value block_sizes(rapidjson::kArrayType);
    rapidjson::Value block_offsets(rapidjson::kArrayType);
    rapidjson::Value value_counts(rapidjson::kArrayType);

    size_t offset = 0;
    for (int j = 0; j < blocksPerFile && i * blocksPerFile + j < blocks.size();
         ++j) {
      const auto& block = blocks[i * blocksPerFile + j];
      file.write(block.data(), block.size());

      metadata["num_blocks"] = j + 1;
      block_sizes.PushBack(rapidjson::Value().SetUint(block.size()), allocator);
      block_offsets.PushBack(rapidjson::Value().SetUint64(offset), allocator);
      // FIXME non 4 byte types
      CHECK_EQ(block.size() % 4, 0) << "fixme only 4 byte types";
      value_counts.PushBack(rapidjson::Value().SetUint(block.size() / 4),
                            allocator);

      // write padding to align to 4096
      if (block.size() % kAlignment != 0) {
        for (int k = 0; k < kAlignment - (block.size() % kAlignment); ++k) {
          file.write("\0", 1);
          offset += 1;
        }
      }

      offset += block.size();
    }

    metadata.AddMember("block_sizes", block_sizes, allocator);
    metadata.AddMember("block_offsets", block_offsets, allocator);
    metadata.AddMember("value_counts", value_counts, allocator);

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    metadata.Accept(writer);

    std::ofstream metaFile(filePath.string() + ".metadata.json");
    metaFile << strbuf.GetString();
    metaFile.flush();
  }
}

// struct CompressedBlock {
//   std::vector<std::vector<char>> compressed_chunks;
//   size_t num_values;
// };

/**
 * @param chunk uncompressed data
 * @return vector of compressed bytes
 */
std::vector<char> compressChunk(std::span<const char>& chunk,
                                int32_t compress_level) {
  auto max_compressed_chunk_size = LZ4_compressBound(chunk.size());
  std::vector<char> compressed_chunk(max_compressed_chunk_size);

  const int compressed_size =
      compress_level == -1
          ? LZ4_compress_default(chunk.data(), compressed_chunk.data(),
                                 chunk.size(), max_compressed_chunk_size)
          : LZ4_compress_HC(chunk.data(), compressed_chunk.data(), chunk.size(),
                            max_compressed_chunk_size, compress_level);

  CHECK_GT(compressed_size, 0)
      << "Failed to compress chunk with size " << chunk.size();
  compressed_chunk.resize(compressed_size);

  return compressed_chunk;
}

/**
 * @param block uncompressed data block
 * @param chunk_size size in bytes of each uncompressed chunk
 * @return vector of compressed chunks
 */
CompressedBlock compressBlockLZ4(const std::vector<char>& block,
                                 size_t chunk_size, int32_t compress_level) {
  size_t num_chunks =
      (block.size() + chunk_size - 1) / chunk_size;  // ceiling division
  std::vector<std::vector<char>> compressed_chunks;
  compressed_chunks.reserve(num_chunks);

  for (size_t i = 0; i < block.size(); i += chunk_size) {
    size_t end = std::min(i + chunk_size, block.size());
    auto chunk = std::span(block.data() + i, end - i);
    CHECK_LE(chunk.size(), chunk_size);
    compressed_chunks.emplace_back(compressChunk(chunk, compress_level));
    // #ifndef NDEBUG
    //     // validate
    //     char* decompressed_chunk = new char[chunk.size()];
    //     auto res =
    //         LZ4_decompress_safe(compressed_chunks.back().data(),
    //         decompressed_chunk,
    //                             compressed_chunks.back().size(),
    //                             chunk.size());
    //     CHECK_EQ(res, chunk.size()) << "failed to decompress chunk";
    //     auto cmp_res = memcmp(chunk.data(), decompressed_chunk,
    //     chunk.size()); CHECK_EQ(cmp_res, 0) << "decompressed chunk does not
    //     match original chunk"; delete[] decompressed_chunk;
    // #endif
  }
  CHECK_EQ(compressed_chunks.size(), num_chunks);
  // fixme: assuming 4 byte types everywhere
  CHECK_EQ(block.size() % 4, 0)
      << "fixme only 4 byte types. block size: " << block.size();
  return {compressed_chunks, block.size() / 4};
}

void threadCompressBlocks(
    const std::vector<std::vector<char>>& uncompressed_blocks,
    std::vector<CompressedBlock>& compressed_blocks, size_t chunk_size) {
  //  CHECK_EQ(uncompressed_blocks.size(), compressed_blocks.size());
  for (size_t i = 0; i < uncompressed_blocks.size(); i++) {
    auto& uncompressed_block = uncompressed_blocks[i];
    auto& compressed_block = compressed_blocks[i];

    size_t num_chunks = (uncompressed_block.size() + chunk_size - 1) /
                        chunk_size;  // ceiling division

    size_t chunk_idx = 0;
    for (size_t j = 0; j < uncompressed_block.size(); j += chunk_size) {
      size_t end = std::min(j + chunk_size, uncompressed_block.size());
      auto chunk = std::span(uncompressed_block.data() + j, end - j);
      CHECK_LE(chunk.size(), chunk_size);
      auto max_compressed_chunk_size = LZ4_compressBound(chunk.size());

      const int compressed_size = LZ4_compress_default(
          chunk.data(), compressed_block.compressed_chunks[chunk_idx].data(),
          chunk.size(), max_compressed_chunk_size);
      CHECK_GT(compressed_size, 0)
          << "Failed to compress chunk with size " << chunk.size();
      compressed_block.compressed_chunks[chunk_idx].resize(compressed_size);
      chunk_idx += 1;
    }
    CHECK_EQ(chunk_idx, num_chunks);
  }
}

void threadCompressBlocks1(
    const std::vector<std::vector<char>>& uncompressed_blocks,
    std::vector<CompressedBlock>& compressed_blocks, size_t chunk_size,
    int32_t compress_level) {
  for (const auto& uncompressed_block : uncompressed_blocks) {
    compressed_blocks.push_back(
        compressBlockLZ4(uncompressed_block, chunk_size, compress_level));
  }
}

std::chrono::milliseconds compressBlocks2(
    ThreadPool& tp, std::vector<std::vector<char>>& uncompressed_blocks,
    size_t chunk_size, size_t n_threads, int32_t compress_level) {
  std::vector<std::vector<std::vector<char>>> threads_uncompressed_blocks(
      n_threads);
  for (size_t i = 0; i < uncompressed_blocks.size(); i++) {
    threads_uncompressed_blocks[i % n_threads].push_back(
        uncompressed_blocks[i]);
  }

  std::vector<std::vector<CompressedBlock>> threads_compressed_blocks(
      n_threads);

  std::vector<std::future<void>> futures;
  for (uint32_t i = 0; i < n_threads; i++) {
    futures.emplace_back(tp.enqueue(
        threadCompressBlocks1, std::ref(threads_uncompressed_blocks[i]),
        std::ref(threads_compressed_blocks[i]), chunk_size, compress_level));
  }

  std::chrono::milliseconds comp_time;
  {
    time_block t{[&](const auto& time) { comp_time = time; }};
    for (auto& thread : futures) {
      thread.get();
    }
  }

  return comp_time;
}

std::chrono::milliseconds compressBlocks1(
    ThreadPool& tp, std::vector<std::vector<char>>& uncompressed_blocks,
    size_t chunk_size, size_t n_threads) {
  // Allocate memory for input.
  std::vector<std::vector<std::vector<char>>> threads_uncompressed_blocks(
      n_threads);
  for (size_t i = 0; i < uncompressed_blocks.size(); i++) {
    threads_uncompressed_blocks[i % n_threads].push_back(
        uncompressed_blocks[i]);
  }

  // Allocate memory for output.
  std::vector<std::vector<CompressedBlock>> threads_compressed_blocks(
      n_threads);
  for (size_t i = 0; i < uncompressed_blocks.size(); i++) {
    auto& block = uncompressed_blocks[i];
    size_t num_chunks = (block.size() + chunk_size - 1) / chunk_size;
    std::vector<std::vector<char>> compressed_chunks;
    compressed_chunks.reserve(num_chunks);
    for (size_t j = 0; j < block.size(); j += chunk_size) {
      size_t end = std::min(j + chunk_size, block.size());
      auto chunk = std::span(block.data() + j, end - j);
      auto max_compressed_chunk_size = LZ4_compressBound(chunk.size());
      std::vector<char> compressed_chunk(max_compressed_chunk_size);
      compressed_chunks.emplace_back(compressed_chunk);
    }
    //    CHECK_EQ(compressed_chunks.size(), num_chunks);
    threads_compressed_blocks[i % n_threads].push_back(
        CompressedBlock{compressed_chunks, uncompressed_blocks.size() / 4});
  }

  std::vector<std::future<void>> futures;
  for (uint32_t i = 0; i < n_threads; i++) {
    futures.emplace_back(tp.enqueue(
        threadCompressBlocks, std::ref(threads_uncompressed_blocks[i]),
        std::ref(threads_compressed_blocks[i]), chunk_size));
  }

  std::chrono::milliseconds comp_time;
  {
    time_block t{[&](const auto& time) { comp_time = time; }};
    for (auto& thread : futures) {
      thread.get();
    }
  }

  return comp_time;
}

/**
 *
 * @return a vector of compressed blocks, each of which is a vector of
 * compressed chunks
 */
std::vector<CompressedBlock> compressBlocks(
    ThreadPool& tp, std::vector<std::vector<char>>& uncompressed_blocks,
    size_t chunk_size, int32_t compress_level) {
  // important, maintain the order of the blocks as we compress each column
  // seperately
  std::vector<std::future<CompressedBlock>> compressed_block_futures;
  compressed_block_futures.reserve(uncompressed_blocks.size());
  for (const auto& block : uncompressed_blocks) {
    compressed_block_futures.emplace_back(
        tp.enqueue(compressBlockLZ4, block, chunk_size, compress_level));
  }

  std::vector<CompressedBlock> compressed_blocks;
  compressed_blocks.reserve(uncompressed_blocks.size());
  for (int i = 0; i < compressed_block_futures.size(); ++i) {
    compressed_blocks.emplace_back(compressed_block_futures[i].get());
    uncompressed_blocks[i].resize(0);
  }
  CHECK_EQ(uncompressed_blocks.size(), compressed_blocks.size());

  return compressed_blocks;
}

std::vector<CompressedBlock> compressBlocksGPU(
    std::vector<std::vector<char>>& uncompressed_blocks, size_t chunk_size,
    CompressionAlgorithm compression_algo) {
  const size_t batch_size = 128;
  std::vector<CompressedBlock> compressed_blocks;
  compressed_blocks.reserve(uncompressed_blocks.size());

  // Iterate through blocks in batches
  for (size_t batch_start = 0; batch_start < uncompressed_blocks.size();
       batch_start += batch_size) {
    const size_t current_batch_size =
        std::min(batch_size, uncompressed_blocks.size() - batch_start);

    cudaStream_t stream = nullptr;
    gpu_run(cudaStreamCreate(&stream));

    size_t total_chunks = 0;
    std::vector<size_t> block_chunk_sizes(current_batch_size);

    // Compute total chunks and per-block chunk sizes
    for (size_t i = 0; i < current_batch_size; ++i) {
      block_chunk_sizes[i] =
          (uncompressed_blocks[batch_start + i].size() + chunk_size - 1) /
          chunk_size;
      total_chunks += block_chunk_sizes[i];
    }

    // Allocate GPU memory for uncompressed data and related pointers
    size_t* host_uncompressed_bytes = nullptr;
    gpu_run(cudaMallocHost(&host_uncompressed_bytes,
                           sizeof(size_t) * total_chunks));

    void** host_uncompressed_ptrs = nullptr;
    gpu_run(
        cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * total_chunks));

    char* device_input_data = nullptr;
    gpu_run(
        cudaMallocAsync(&device_input_data, chunk_size * total_chunks, stream));

    size_t chunk_offset = 0;
    for (size_t i = 0; i < current_batch_size; ++i) {
      const auto& block = uncompressed_blocks[batch_start + i];
      size_t block_size = block.size();
      size_t num_chunks = block_chunk_sizes[i];

      // Copy each block's data to GPU memory
      for (size_t j = 0; j < num_chunks; ++j) {
        size_t offset = chunk_offset * chunk_size;
        size_t copy_size =
            (j + 1 < num_chunks) ? chunk_size : block_size - j * chunk_size;

        gpu_run(cudaMemcpyAsync(device_input_data + offset,
                                block.data() + j * chunk_size, copy_size,
                                cudaMemcpyHostToDevice, stream));

        host_uncompressed_ptrs[chunk_offset] = device_input_data + offset;
        host_uncompressed_bytes[chunk_offset] = copy_size;
        chunk_offset++;
      }
    }

    void** device_uncompressed_ptrs = nullptr;
    size_t* device_uncompressed_bytes = nullptr;

    gpu_run(cudaMallocAsync(&device_uncompressed_ptrs,
                            sizeof(void*) * total_chunks, stream));
    gpu_run(cudaMallocAsync(&device_uncompressed_bytes,
                            sizeof(size_t) * total_chunks, stream));

    gpu_run(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs,
                            sizeof(void*) * total_chunks,
                            cudaMemcpyHostToDevice, stream));
    gpu_run(cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes,
                            sizeof(size_t) * total_chunks,
                            cudaMemcpyHostToDevice, stream));

    // Allocate workspace and output memory
    nvcompBatchedCascadedOpts_t opts_cascaded{chunk_size, NVCOMP_TYPE_INT, 2, 1,
                                              1};
    nvcompBatchedGdeflateOpts_t opts_gdeflate{0};
    nvcompBatchedSnappyOpts_t opts_snappy{0};
    size_t temp_bytes, max_out_bytes;
    switch (compression_algo) {
      case CompressionAlgorithm::CASCADED: {
        nvcompBatchedCascadedCompressGetTempSize(total_chunks, chunk_size,
                                                 opts_cascaded, &temp_bytes);
        nvcompBatchedCascadedCompressGetMaxOutputChunkSize(
            chunk_size, opts_cascaded, &max_out_bytes);
        break;
      }
      case CompressionAlgorithm::GDEFLATE: {
        nvcompBatchedGdeflateCompressGetTempSize(total_chunks, chunk_size,
                                                 opts_gdeflate, &temp_bytes);
        nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
            chunk_size, opts_gdeflate, &max_out_bytes);
        break;
      }
      case CompressionAlgorithm::SNAPPY: {
        nvcompBatchedSnappyCompressGetTempSize(total_chunks, chunk_size,
                                               opts_snappy, &temp_bytes);
        nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
            chunk_size, opts_snappy, &max_out_bytes);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported compression algorithm: " << compression_algo;
    }

    void* device_temp_ptr = nullptr;
    gpu_run(cudaMallocAsync(&device_temp_ptr, temp_bytes, stream));

    void** host_compressed_ptrs = nullptr;
    gpu_run(
        cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * total_chunks));

    for (size_t i = 0; i < total_chunks; ++i) {
      gpu_run(cudaMallocAsync(&host_compressed_ptrs[i], max_out_bytes, stream));
    }

    void** device_compressed_ptrs = nullptr;
    gpu_run(cudaMallocAsync(&device_compressed_ptrs,
                            sizeof(void*) * total_chunks, stream));
    gpu_run(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs,
                            sizeof(void*) * total_chunks,
                            cudaMemcpyHostToDevice, stream));

    size_t* device_compressed_bytes = nullptr;
    gpu_run(cudaMallocAsync(&device_compressed_bytes,
                            sizeof(size_t) * total_chunks, stream));

    // Compress data
    nvcompStatus_t comp_res;
    switch (compression_algo) {
      case CompressionAlgorithm::CASCADED: {
        comp_res = nvcompBatchedCascadedCompressAsync(
            device_uncompressed_ptrs, device_uncompressed_bytes, chunk_size,
            total_chunks, device_temp_ptr, temp_bytes, device_compressed_ptrs,
            device_compressed_bytes, opts_cascaded, stream);
        break;
      }
      case CompressionAlgorithm::GDEFLATE: {
        comp_res = nvcompBatchedGdeflateCompressAsync(
            device_uncompressed_ptrs, device_uncompressed_bytes, chunk_size,
            total_chunks, device_temp_ptr, temp_bytes, device_compressed_ptrs,
            device_compressed_bytes, opts_gdeflate, stream);
        break;
      }
      case CompressionAlgorithm::SNAPPY: {
        comp_res = nvcompBatchedSnappyCompressAsync(
            device_uncompressed_ptrs, device_uncompressed_bytes, chunk_size,
            total_chunks, device_temp_ptr, temp_bytes, device_compressed_ptrs,
            device_compressed_bytes, opts_snappy, stream);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported compression algorithm: ";
    }
    CHECK_EQ(comp_res, nvcompSuccess);

    gpu_run(cudaStreamSynchronize(stream));

    // Retrieve compressed data
    size_t chunk_idx = 0;
    for (size_t i = 0; i < current_batch_size; ++i) {
      std::vector<std::vector<char>> compressed_chunks;
      size_t num_chunks = block_chunk_sizes[i];
      for (size_t j = 0; j < num_chunks; ++j) {
        size_t compressed_size = 0;
        gpu_run(cudaMemcpy(&compressed_size,
                           device_compressed_bytes + chunk_idx, sizeof(size_t),
                           cudaMemcpyDeviceToHost));

        std::vector<char> compressed_chunk(compressed_size);
        gpu_run(cudaMemcpy(compressed_chunk.data(),
                           host_compressed_ptrs[chunk_idx], compressed_size,
                           cudaMemcpyDeviceToHost));
        compressed_chunks.push_back(compressed_chunk);
        chunk_idx++;
      }
      compressed_blocks.push_back(
          {compressed_chunks, uncompressed_blocks[batch_start + i].size() / 4});
    }

    // Free GPU and host memory for the batch
    gpu_run(cudaFree(device_input_data));
    gpu_run(cudaFree(device_uncompressed_ptrs));
    gpu_run(cudaFree(device_uncompressed_bytes));
    gpu_run(cudaFree(device_temp_ptr));
    gpu_run(cudaFree(device_compressed_ptrs));
    gpu_run(cudaFree(device_compressed_bytes));

    for (size_t i = 0; i < total_chunks; ++i) {
      gpu_run(cudaFree(host_compressed_ptrs[i]));
    }

    gpu_run(cudaFreeHost(host_uncompressed_bytes));
    gpu_run(cudaFreeHost(host_uncompressed_ptrs));
    gpu_run(cudaFreeHost(host_compressed_ptrs));

    gpu_run(cudaStreamDestroy(stream));
  }

  return compressed_blocks;
}

std::vector<CompressedBlock> compressBlocksLZ4(
    ThreadPool& tp, std::vector<std::vector<char>>& uncompressed_blocks,
    size_t chunk_size, int32_t compress_level, uint32_t n_threads) {
  std::vector<CompressedBlock> compressed_blocks(uncompressed_blocks.size());
  size_t num_blocks = uncompressed_blocks.size();
  size_t blocks_per_thread = (num_blocks + n_threads - 1) / n_threads;

  std::vector<std::future<void>> futures;

  for (int t = 0; t < n_threads; ++t) {
    futures.emplace_back(tp.enqueue([&, t]() {
      size_t start_index = t * blocks_per_thread;
      size_t end_index = std::min(start_index + blocks_per_thread, num_blocks);

      for (size_t i = start_index; i < end_index; ++i) {
        compressed_blocks[i] = compressBlockLZ4(uncompressed_blocks[i],
                                                chunk_size, compress_level);
        uncompressed_blocks[i].resize(0);
      }
    }));
  }

  LOG(INFO) << "Finished enqueuing compress_blocks tasks";

  for (auto& future : futures) {
    future.get();
  }

  uint64_t bytes_compressed = 0;
  for (const auto& block : compressed_blocks) {
    for (const auto& chunk : block.compressed_chunks) {
      bytes_compressed += chunk.size();
    }
  }
  LOG(INFO) << "All threads finished. Compressed " << bytes_compressed
            << " bytes.";

  CHECK_EQ(uncompressed_blocks.size(), compressed_blocks.size());

  return compressed_blocks;
}

std::vector<CompressedBlock> compressBlocks(
    ThreadPool& tp, std::vector<std::vector<char>>& uncompressed_blocks,
    size_t chunk_size, int32_t compress_level,
    CompressionAlgorithm compression_algo, uint32_t n_threads,
    size_t rfor_tile_size_elems, bool rfor_precompute_rl) {
  switch (compression_algo) {
    case CompressionAlgorithm::LZ4:
      LOG(INFO) << "Compressing blocks using LZ4";
      return compressBlocksLZ4(tp, uncompressed_blocks, chunk_size,
                               compress_level, n_threads);
    case CompressionAlgorithm::CASCADED:
      LOG(INFO) << "Compressing blocks using Cascaded";
      return compressBlocksGPU(uncompressed_blocks, chunk_size,
                               compression_algo);
      //      return compressBlocksCascaded(uncompressed_blocks, chunk_size);
    case CompressionAlgorithm::GDEFLATE:
      LOG(INFO) << "Compressing blocks using GDeflate";
      return compressBlocksGPU(uncompressed_blocks, chunk_size,
                               compression_algo);
      //      return compressBlocksGDeflate(uncompressed_blocks, chunk_size);
    case CompressionAlgorithm::SNAPPY:
      LOG(INFO) << "Compressing blocks using Snappy";
      return compressBlocksGPU(uncompressed_blocks, chunk_size,
                               compression_algo);
      //      return compressBlocksSnappy(uncompressed_blocks, chunk_size)
    case CompressionAlgorithm::RFOR:
      LOG(INFO) << "Compressing blocks using RFOR";
      return compressBlocksRFOR(uncompressed_blocks, rfor_tile_size_elems,
                                rfor_precompute_rl);
    case CompressionAlgorithm::FLBP:
      // FLBP keeps the FastLanes on-disk vector format fixed at 1024 values.
      // The paper's smaller "mini-vectors" are a runtime decode strategy
      // inside the GPU kernel, not a different dataset format knob.
      LOG(INFO) << "Compressing blocks using FastLanes bitpack";
      return compressBlocksFLBP(uncompressed_blocks, FLBP_VECTOR_SIZE);
    default:
      LOG(FATAL) << "Unsupported compression algorithm: " << compression_algo;
  }
}

/**
 * @param blocks vector of compressed blocks to write to files
 * @param output_directories A vector of directories. The file is split into
 * size() parts and one part is written to each directory
 * @param base_file_Name The base file name to write to. The file name will be
 * appended with _i_numFiles where i is the index of the file and numFiles is
 * the total number of files
 * @param chunk_size size in bytes of each uncompressed chunk, previously used
 * to compress the blocks
 */

void writeCompressedBlocksToFiles(
    std::vector<CompressedBlock>& blocks,
    const std::vector<fs::path> output_directories,
    const std::string& base_file_Name, const int chunk_size,
    CompressionAlgorithm compression_algo = CompressionAlgorithm::CASCADED,
    bool metadata_in_block = false, size_t rfor_tile_size_elems = RFOR_TILE_SIZE,
    bool rfor_precompute_rl = false) {

  const int numFiles = output_directories.size();
  const int blocksPerFile = (blocks.size() + numFiles - 1) / numFiles;
  LOG(INFO) << "splitting " << blocks.size() << " blocks into " << numFiles
            << " files with " << blocksPerFile << " blocks per file";

  for (int i = 0; i < numFiles; ++i) {
    const fs::path filePath =
        output_directories[i] / (base_file_Name + "_" + std::to_string(i) +
                                 "_" + std::to_string(numFiles));
    int data_file_fd = open(filePath.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    PCHECK(data_file_fd > 0) << "failed to open: " << filePath;

    rapidjson::Document metadata;
    metadata.SetObject();
    rapidjson::Document::AllocatorType& allocator = metadata.GetAllocator();

    // Handle per-codec metadata headers.
    metadata.AddMember("data_format", "COMPRESSED", allocator);
    if (compression_algo == CompressionAlgorithm::RFOR) {
      metadata.AddMember("compression_algorithm", "RFOR", allocator);
      metadata.AddMember(
          "tile_size", rapidjson::Value().SetUint(static_cast<uint32_t>(rfor_tile_size_elems)),
          allocator);
      metadata.AddMember(
          "rl_format",
          rapidjson::Value(rfor_precompute_rl ? "CUMULATIVE" : "RUN_LENGTHS",
                           allocator),
          allocator);
      metadata.AddMember("block_format", "clean_tiles_no_headers", allocator);
    } else if (compression_algo == CompressionAlgorithm::FLBP) {
      metadata.AddMember("compression_algorithm", "FLBP", allocator);
      // Unlike RFOR, FLBP does not expose a preprocessing-time tile/vector size
      // flag here. The on-disk FastLanes vector stays 1024-wide; only the
      // runtime decode implementation may split it into smaller mini-vectors.
      metadata.AddMember("vector_size",
                         rapidjson::Value().SetUint(FLBP_VECTOR_SIZE),
                         allocator);
      metadata.AddMember("block_format", "clean_vectors_no_headers", allocator);
    } else {
      if (compression_algo == CompressionAlgorithm::LZ4) {
        metadata.AddMember("compression_algorithm", "LZ4", allocator);
      } else if (compression_algo == CompressionAlgorithm::CASCADED) {
        metadata.AddMember("compression_algorithm", "CASCADED", allocator);
      } else if (compression_algo == CompressionAlgorithm::GDEFLATE) {
        metadata.AddMember("compression_algorithm", "GDEFLATE", allocator);
      }
    }

    metadata.AddMember("_comment", "generated with proteus-generate-cb2t-dataset",
                       allocator);
    metadata.AddMember("data_file",
                       rapidjson::Value(filePath.filename().c_str(), allocator),
                       allocator);
    metadata.AddMember("num_blocks", 0, allocator);
    metadata.AddMember("decompressed_chunk_size", chunk_size, allocator);
    metadata.AddMember("pbmd_chunk_offsets", metadata_in_block, allocator);

    rapidjson::Value block_sizes(rapidjson::kArrayType);
    rapidjson::Value block_offsets(rapidjson::kArrayType);
    rapidjson::Value value_counts(rapidjson::kArrayType);
    rapidjson::Value all_chunk_sizes(rapidjson::kArrayType);

    rapidjson::Value all_cb2t_num_decode_val(rapidjson::kArrayType);
    rapidjson::Value all_cb2t_num_decode_rl(rapidjson::kArrayType);
    rapidjson::Value all_cb2t_val_words(rapidjson::kArrayType);
    rapidjson::Value all_cb2t_rl_words(rapidjson::kArrayType);

    size_t offset = 0;
    size_t bytes_written = 0;
    size_t chunks_written = 0;
    for (int j = 0; j < blocksPerFile && i * blocksPerFile + j < blocks.size(); ++j) {
      auto& block = blocks[i * blocksPerFile + j];

      uint64_t block_size = 0;
      rapidjson::Value block_chunk_sizes(rapidjson::kArrayType);

      rapidjson::Value block_cb2t_num_decode_val(rapidjson::kArrayType);
      rapidjson::Value block_cb2t_num_decode_rl(rapidjson::kArrayType);
      rapidjson::Value block_cb2t_val_words(rapidjson::kArrayType);
      rapidjson::Value block_cb2t_rl_words(rapidjson::kArrayType);

      CHECK_GE(block.compressed_chunks.size(), 1);

      // ✅ Updated RFOR validation for chunks
      if (compression_algo == CompressionAlgorithm::RFOR) {
        // LOG(INFO) << "RFOR block " << j << " has " << block.compressed_chunks.size()
        //           << " tiles";
      }

      std::vector<uint32_t> chunk_sizes;
      chunk_sizes.reserve(block.compressed_chunks.size());
      std::vector<uint32_t> chunk_offsets;
      chunk_offsets.reserve(block.compressed_chunks.size());
      uint32_t running_chunk_offset = 0;

      for (const auto& chunk : block.compressed_chunks) {
        CHECK_GT(chunk.size(), 0);
        CHECK_LE(
            chunk.size(),
            static_cast<size_t>(std::numeric_limits<uint32_t>::max() -
                                running_chunk_offset))
            << "Compressed chunk exceeds 32-bit PBMD offset range";
        chunk_offsets.push_back(running_chunk_offset);
        chunk_sizes.push_back(chunk.size());
        block_chunk_sizes.PushBack(rapidjson::Value().SetUint(chunk.size()), allocator);
        running_chunk_offset += static_cast<uint32_t>(chunk.size());
        chunks_written += 1;

        // RFOR aux-section metadata.
        if (compression_algo == CompressionAlgorithm::RFOR) {
          // ✅ Parse metadata from chunk to calculate val section size
          const RFORTileMetadata* metadata_ptr = reinterpret_cast<const RFORTileMetadata*>(chunk.data());

          // ✅ Calculate val section size exactly like microbenchmark
          uint32_t val_packed_words = ((metadata_ptr->val_count * metadata_ptr->val_bitwidth + 31) / 32);
          uint32_t val_section_size = sizeof(RFORTileMetadata) + (val_packed_words * sizeof(uint32_t));

          const uint32_t cb2t_items_per_thread =
              static_cast<uint32_t>(proteus::rfor::cb2t::kItemsPerThread);
          const uint32_t cb2t_num_decode_val =
              (metadata_ptr->val_count + cb2t_items_per_thread - 1) / cb2t_items_per_thread;
          const uint32_t cb2t_num_decode_rl =
              (metadata_ptr->rl_count + cb2t_items_per_thread - 1) / cb2t_items_per_thread;

          block_cb2t_num_decode_val.PushBack(
              rapidjson::Value().SetUint(cb2t_num_decode_val), allocator);
          block_cb2t_num_decode_rl.PushBack(
              rapidjson::Value().SetUint(cb2t_num_decode_rl), allocator);
          block_cb2t_val_words.PushBack(
              rapidjson::Value().SetUint(val_packed_words), allocator);
          block_cb2t_rl_words.PushBack(
              rapidjson::Value().SetUint(((metadata_ptr->rl_count * metadata_ptr->rl_bitwidth + 31) / 32)),
              allocator);

          // Verify chunk size consistency
          uint32_t rl_packed_words = ((metadata_ptr->rl_count * metadata_ptr->rl_bitwidth + 31) / 32);
          uint32_t rl_section_size = rl_packed_words * sizeof(uint32_t);
          size_t expected_chunk_size = val_section_size + rl_section_size;

          if (chunk.size() != expected_chunk_size) {
            LOG(WARNING) << "RFOR chunk size mismatch: expected " << expected_chunk_size
                        << ", got " << chunk.size();
          }
        } else if (compression_algo == CompressionAlgorithm::FLBP) {
          CHECK_GE(chunk.size(), sizeof(FLBPVectorMetadata))
              << "FLBP chunk is smaller than metadata header";
          const FLBPVectorMetadata* metadata_ptr =
              reinterpret_cast<const FLBPVectorMetadata*>(chunk.data());

          // For FLBP, the fixed header is self-describing; no aux write needed.
        }
      }

      if (metadata_in_block) {
        // Serialize per-block metadata before the compressed chunks.
        // Layout: [magic][algo][num_chunks][chunk_offsets...][payload chunks...]
        // Each payload chunk is self-describing (begins with its native codec header).
        constexpr uint32_t kMetadataMagic = 0x50424D44;  // "PBMD"
        uint32_t algo_id = static_cast<uint32_t>(compression_algo);
        uint32_t num_chunks = block.compressed_chunks.size();

        auto wrote_bytes = write(data_file_fd, &kMetadataMagic, sizeof(uint32_t));
        PCHECK(wrote_bytes == sizeof(uint32_t)) << "failed to write metadata magic";
        wrote_bytes = write(data_file_fd, &algo_id, sizeof(uint32_t));
        PCHECK(wrote_bytes == sizeof(uint32_t)) << "failed to write metadata algo id";
        wrote_bytes = write(data_file_fd, &num_chunks, sizeof(uint32_t));
        PCHECK(wrote_bytes == sizeof(uint32_t)) << "failed to write num_chunks";

        bytes_written += sizeof(uint32_t) * 3;
        block_size += sizeof(uint32_t) * 3;

        if (!chunk_offsets.empty()) {
          wrote_bytes = write(data_file_fd, chunk_offsets.data(),
                              sizeof(uint32_t) * chunk_offsets.size());
          PCHECK(wrote_bytes == static_cast<ssize_t>(sizeof(uint32_t) *
                                                    chunk_offsets.size()))
              << "failed to write chunk offsets";
          bytes_written += sizeof(uint32_t) * chunk_offsets.size();
          block_size += sizeof(uint32_t) * chunk_offsets.size();
        }

      }

      for (const auto& chunk : block.compressed_chunks) {
        auto wrote_bytes = write(data_file_fd, chunk.data(), chunk.size());
        PCHECK(wrote_bytes == chunk.size())
            << "failed to write to file. wrote_bytes: " << wrote_bytes
            << " to file " << filePath << " chunk.size() = " << chunk.size();
        bytes_written += chunk.size();
        block_size += chunk.size();
      }


      if (compression_algo == CompressionAlgorithm::RFOR) {
        // LOG(INFO) << "RFOR block " << j << " contains " << block.compressed_chunks.size()
        //           << " tiles with total compressed size " << block_size << " bytes";

        // ✅ Sample validation: check first chunk structure if available
        if (!block.compressed_chunks.empty() && block.compressed_chunks[0].size() >= sizeof(RFORTileMetadata)) {
          const RFORTileMetadata* first_metadata = reinterpret_cast<const RFORTileMetadata*>(
              block.compressed_chunks[0].data());

          // LOG(INFO) << "First tile: val_ref=" << first_metadata->val_reference
          //           << ", val_bitwidth=" << first_metadata->val_bitwidth
          //           << ", val_count=" << first_metadata->val_count
          //           << ", rl_ref=" << first_metadata->rl_reference
          //           << ", rl_bitwidth=" << first_metadata->rl_bitwidth
          //           << ", rl_count=" << first_metadata->rl_count;
        }
      }

      metadata["num_blocks"] = j + 1;
      block_sizes.PushBack(rapidjson::Value().SetUint(block_size), allocator);
      block_offsets.PushBack(rapidjson::Value().SetUint64(offset), allocator);
      value_counts.PushBack(rapidjson::Value().SetUint(block.num_values), allocator);
      all_chunk_sizes.PushBack(block_chunk_sizes, allocator);

      if (compression_algo == CompressionAlgorithm::RFOR) {
        all_cb2t_num_decode_val.PushBack(block_cb2t_num_decode_val, allocator);
        all_cb2t_num_decode_rl.PushBack(block_cb2t_num_decode_rl, allocator);
        all_cb2t_val_words.PushBack(block_cb2t_val_words, allocator);
        all_cb2t_rl_words.PushBack(block_cb2t_rl_words, allocator);
      }

      // ✅ write padding to align to 4096
      constexpr char padding_byte = '1';
      if (block_size % kAlignment != 0) {
        for (int k = 0; k < kAlignment - (block_size % kAlignment); ++k) {
          PCHECK(write(data_file_fd, &padding_byte, 1) > 0)
              << "failed to write padding";
          offset += 1;
        }
      }

      offset += block_size;
    }

    metadata.AddMember("block_sizes", block_sizes, allocator);
    metadata.AddMember("block_offsets", block_offsets, allocator);
    metadata.AddMember("value_counts", value_counts, allocator);
    metadata.AddMember("chunk_sizes", all_chunk_sizes, allocator);

    if (compression_algo == CompressionAlgorithm::RFOR) {
      metadata.AddMember("cb2t_num_decode_val", all_cb2t_num_decode_val, allocator);
      metadata.AddMember("cb2t_num_decode_rl", all_cb2t_num_decode_rl, allocator);
      metadata.AddMember("cb2t_val_words", all_cb2t_val_words, allocator);
      metadata.AddMember("cb2t_rl_words", all_cb2t_rl_words, allocator);
      metadata.AddMember(
          "cb2t_block_threads",
          rapidjson::Value().SetUint(static_cast<uint32_t>(proteus::rfor::cb2t::kBlockThreads)),
          allocator);
      metadata.AddMember(
          "cb2t_items_per_thread",
          rapidjson::Value().SetUint(static_cast<uint32_t>(proteus::rfor::cb2t::kItemsPerThread)),
          allocator);
    }

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    metadata.Accept(writer);
    std::ofstream metaFile(filePath.string() + ".metadata.json",
                           std::ofstream::out | std::ofstream::trunc);
    PCHECK(metaFile) << "failed to open metadata file for writing";
    metaFile << strbuf.GetString();
    PCHECK(metaFile) << "failed to write metadata to file";
    close(data_file_fd);
    LOG(INFO) << "wrote " << bytes_written << " bytes to " << filePath << " in "
              << chunks_written << " chunks and " << blocks.size() << " blocks";
  }
}

std::vector<fs::path> splitOutputDirectories(const std::string& str) {
  std::vector<fs::path> result;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    result.push_back(token);
    if (!fs::exists(result.back())) {
      if (fs::create_directories(result.back())) {
        LOG(INFO) << "Successfully created directories: " << result.back();
      } else {
        LOG(FATAL) << "Failed to create directories or they already exist: "
                   << result.back();
      }
    } else {
      CHECK(fs::is_directory(result.back())) << "output directory is not a "
                                                "directory: "
                                             << result.back();
    }
  }

  return result;
}

// Column set is dataset-driven, not hardcoded: read every relation's columns
// from the dataset's catalog.json (each top-level member is a relation whose
// type.inner.attributes[] carry the column names as `attrName`).
std::unordered_set<std::string> columnNamesFromCatalog(
    const fs::path& catalog_path) {
  std::unordered_set<std::string> names;
  std::ifstream in(catalog_path);
  CHECK(in.good()) << "Cannot open catalog.json for the column set: "
                   << catalog_path
                   << " (pass --columns_file to supply columns explicitly)";
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  rapidjson::Document doc;
  doc.Parse(text.c_str());
  CHECK(!doc.HasParseError() && doc.IsObject())
      << "Malformed catalog.json: " << catalog_path;
  for (auto rel = doc.MemberBegin(); rel != doc.MemberEnd(); ++rel) {
    if (!rel->value.IsObject() || !rel->value.HasMember("type")) continue;
    const auto& type = rel->value["type"];
    if (!type.IsObject() || !type.HasMember("inner")) continue;
    const auto& inner = type["inner"];
    if (!inner.IsObject() || !inner.HasMember("attributes")) continue;
    const auto& attrs = inner["attributes"];
    if (!attrs.IsArray()) continue;
    for (const auto& a : attrs.GetArray()) {
      if (a.IsObject() && a.HasMember("attrName") && a["attrName"].IsString()) {
        names.insert(a["attrName"].GetString());
      }
    }
  }
  CHECK(!names.empty()) << "No columns found in catalog.json: " << catalog_path;
  return names;
}

// One column name per line; '#' comments and blank lines skipped.
std::unordered_set<std::string> columnNamesFromFile(const fs::path& path) {
  std::unordered_set<std::string> names;
  std::ifstream in(path);
  CHECK(in.good()) << "Cannot open --columns_file: " << path;
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                             line.back() == '\t'))
      line.pop_back();
    if (!line.empty() && line[0] != '#') names.insert(line);
  }
  CHECK(!names.empty()) << "--columns_file has no column names: " << path;
  return names;
}

int main(int argc, char* argv[]) {
  // Parse the command line arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging((argv)[0]);
  FLAGS_colorlogtostderr = true;
  google::LogToStderr();

  CHECK(!FLAGS_input_directory.empty()) << "Input directory not provided";
  CHECK(!FLAGS_output_directories.empty()) << "Output directories not provided";
  CHECK_GT(FLAGS_num_files, 0) << "Number of files must be greater than 0";
  CHECK(!(FLAGS_precompute_rl && !FLAGS_compress_data))
      << "--precompute_rl requires --compress_data=true";
  CHECK(!(FLAGS_precompute_rl && FLAGS_compression_algorithm != "rfor"))
      << "--precompute_rl currently supports only --compression_algorithm=rfor";

 // assert that if not --compress_data no compression algorithm is set
  if (!FLAGS_compress_data) {
    CHECK(FLAGS_compression_algorithm == "none")
        << "Compression algorithm should not be set if --compress_data is false";
  }
  else
  {
    // check the compression algorithm is [lz4, cascaded, gdeflate, snappy, rfor]
    CHECK(FLAGS_compression_algorithm == "lz4" ||
          FLAGS_compression_algorithm == "cascaded" ||
          FLAGS_compression_algorithm == "gdeflate" ||
          FLAGS_compression_algorithm == "snappy" ||
          FLAGS_compression_algorithm == "rfor" ||
          FLAGS_compression_algorithm == "flbp")
        << "Invalid compression algorithm: "
        << FLAGS_compression_algorithm;
  }

  std::vector<fs::path> output_directories =
      splitOutputDirectories(FLAGS_output_directories);

  fs::path input_dir_path(FLAGS_input_directory);
  CHECK(fs::exists(input_dir_path))
      << "Input directory does not exist: " << input_dir_path;
  CHECK(fs::is_directory(input_dir_path))
      << "Input directory is not a directory: " << input_dir_path;

  //  LOG(INFO) << "input file: " << FLAGS_input_file;
  LOG(INFO) << "input directory: " << FLAGS_input_directory;
  LOG(INFO) << "output directories: " << FLAGS_output_directories;
  LOG(INFO) << "compress data: " << FLAGS_compress_data;
  LOG(INFO) << "Splitting each file into:" << output_directories.size()
            << "output files";
  LOG(INFO) << "compress level: " << FLAGS_compress_level;
  google::InstallFailureSignalHandler();

  topology::init();
  auto& topo = topology::getInstance();

  std::vector<fs::path> data_file_paths;
  // Dataset-driven column set: --columns_file overrides, else read every
  // relation's columns from <input_directory>/catalog.json. The partitioner
  // carries no dataset-specific hardcoded list, so it works for any workload
  // (TPC-H / SSB / Stack) without edits.
  std::unordered_set<std::string> column_names =
      FLAGS_columns_file.empty()
          ? columnNamesFromCatalog(input_dir_path / "catalog.json")
          : columnNamesFromFile(fs::path(FLAGS_columns_file));
  LOG(INFO) << "Partitioning " << column_names.size() << " columns ("
            << (FLAGS_columns_file.empty()
                    ? "from catalog.json"
                    : "from --columns_file " + FLAGS_columns_file)
            << ")";

  for (const auto& entry : fs::directory_iterator(input_dir_path)) {
    if (entry.is_regular_file() &&
        entry.path().string().find("dict") == std::string::npos &&
        entry.path().string().find("catalog") == std::string::npos) {
      std::string filename = entry.path().filename().string();
      std::string column_name = filename.substr(filename.find_last_of('.') + 1);
      if (column_names.find(column_name) != column_names.end()) {
        LOG(INFO) << "Adding file: " << entry.path();
        data_file_paths.push_back(entry.path());
      }
      //            LOG(INFO) << "Adding file: " << entry.path();
      //            data_file_paths.push_back(entry.path());
    }
  }
  constexpr size_t kBlockSize = 2 * 1024 * 1024;  // 2MiB
  size_t kChunkSize;

  CompressionAlgorithm compression_algo = CompressionAlgorithm::LZ4;
  if (!FLAGS_compress_data) {
    kChunkSize = kBlockSize;
  } else if (FLAGS_compression_algorithm == "lz4") {
    compression_algo = CompressionAlgorithm::LZ4;
    LOG(INFO) << "Using lz4 compression";
    kChunkSize = 64 * 1024;
  } else if (FLAGS_compression_algorithm == "cascaded") {
    compression_algo = CompressionAlgorithm::CASCADED;
    LOG(INFO) << "Using cascaded compression";
    kChunkSize = 64 * 1024;
  } else if (FLAGS_compression_algorithm == "gdeflate") {
    compression_algo = CompressionAlgorithm::GDEFLATE;
    LOG(INFO) << "Using gdeflate compression";
    kChunkSize = 64 * 1024;
  } else if (FLAGS_compression_algorithm == "snappy") {
    compression_algo = CompressionAlgorithm::SNAPPY;
    LOG(INFO) << "Using snappy compression";
    kChunkSize = 4 * 1024;
  } else if (FLAGS_compression_algorithm == "rfor") {
    compression_algo = CompressionAlgorithm::RFOR;
    LOG(INFO) << "Using rfor compression";
    CHECK_GT(FLAGS_rfor_tile_size, 0) << "--rfor_tile_size must be > 0";
    kChunkSize = static_cast<size_t>(FLAGS_rfor_tile_size) * sizeof(int);
    LOG(INFO) << "RFOR dataset tile_size=" << FLAGS_rfor_tile_size
              << " (default CB2T tile_size=" << RFOR_TILE_SIZE << ")";
    LOG(INFO) << "RFOR precompute cumulative RL="
              << (FLAGS_precompute_rl ? "true" : "false");
  } else if (FLAGS_compression_algorithm == "flbp") {
    compression_algo = CompressionAlgorithm::FLBP;
    // RFOR exposes tile_size because Proteus supports multiple RFOR on-disk
    // tile formats. FLBP keeps the FastLanes on-disk vector fixed at 1024
    // values; paper-style mini-vectors are an internal decode-kernel detail.
    LOG(INFO) << "Using FastLanes bitpack compression";
    kChunkSize = static_cast<size_t>(FLBP_VECTOR_SIZE) * sizeof(int);
    LOG(INFO) << "FLBP vector_size=" << FLBP_VECTOR_SIZE
              << " decompressed_chunk_size=" << kChunkSize;
  } else {
    LOG(FATAL) << "Invalid compression algorithm: "
               << FLAGS_compression_algorithm;
  }

  if (FLAGS_compress_data == true) {
    size_t core_count = topo.getCoreCount();
    ThreadPool pool(core_count);
    // Change that to use the number of files
    for (const auto& data_file_path : data_file_paths) {
      LOG(INFO) << "Splitting file: " << data_file_path;
      auto blocks = splitFileIntoBlocks(data_file_path, kBlockSize);
      auto compressed_blocks =
          compressBlocks(pool, blocks, kChunkSize, FLAGS_compress_level,
                         compression_algo, core_count,
                         static_cast<size_t>(FLAGS_rfor_tile_size),
                         FLAGS_precompute_rl);
      CHECK_EQ(blocks.size(), compressed_blocks.size());
      writeCompressedBlocksToFiles(compressed_blocks, output_directories,
                                   data_file_path.filename().string(),
                                   kChunkSize, compression_algo,
                                   FLAGS_metadata_in_block,
                                   static_cast<size_t>(FLAGS_rfor_tile_size),
                                   FLAGS_precompute_rl);
      fs::path dict_path = data_file_path.string() + ".dict";
      if (fs::exists(dict_path)) {
        for (const auto& output_dir : output_directories) {
          fs::remove(output_dir / dict_path.filename().string());
          fs::copy(dict_path, output_dir / dict_path.filename().string());
        }
      }
    }
  } else {
    ThreadPool pool(4);
    std::vector<std::future<void>> split_column_futures;
    split_column_futures.reserve(data_file_paths.size());
    for (const auto& data_file_path : data_file_paths) {
      split_column_futures.emplace_back(pool.enqueue([&, data_file_path]() {
        LOG(INFO) << "Splitting file: " << data_file_path;
        auto blocks = splitFileIntoBlocks(data_file_path, kBlockSize);
        writeUncompressedChunksToFiles(blocks, output_directories,
                                       data_file_path.filename().string());
        fs::path dict_path = data_file_path.string() + ".dict";
        if (fs::exists(dict_path)) {
          for (const auto& output_dir : output_directories) {
            fs::remove(output_dir / dict_path.filename().string());
            fs::copy(dict_path, output_dir / dict_path.filename().string());
          }
        }
      }));
    }
    LOG(INFO) << "waiting for futures to complete";
    CHECK_EQ(split_column_futures.size(), data_file_paths.size());
    for (int i = 0; i < data_file_paths.size(); i++) {
      LOG(INFO) << "waiting for future " << data_file_paths[i];
      split_column_futures[i].get();
    }
    LOG(INFO) << "done";
  }

  return 0;
}
