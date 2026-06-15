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

#include "gpu-benchmarks.hpp"

#include <numeric>
#include <platform/util/glog.hpp>
#include <platform/util/timing.hpp>

#include "compressed-file.hpp"

struct DecompressionResult {
  size_t decompressed_size;
  double decompression_throughput;
  std::chrono::milliseconds decomp_time;
};

[[maybe_unused]] static DecompressionResult benchmark_lz4_gpu_decompression(
    const std::vector<CompressedFile::CompressedBlock>& compressed_blocks,
    bool from_cpu_mem) {
  std::chrono::milliseconds decomp_time;
  const uint32_t block_batch_size = 1;
  uint32_t block_end;

  auto gpu_decompressor = GpuDecompressor(
      compressed_blocks.front().decompressed_chunk_size, block_batch_size,
      compressed_blocks.front().chunk_sizes.size(), CompressionAlgorithm::LZ4);

  std::cerr << "Block size: " << block_batch_size << std::endl;

  uint64_t bytes_decompressed = 0;
  {
    cudaStream_t stream = createNonBlockingStream();
    time_block t{[&](const auto& time) { decomp_time = time; }};
    for (size_t block = 0; block < compressed_blocks.size();
         block += block_batch_size) {
      if (block + block_batch_size >= compressed_blocks.size()) {
        block_end = compressed_blocks.size();
      } else {
        block_end = block + block_batch_size;
      }

      const size_t decomp_size = 2_M;

      std::vector<std::vector<uint32_t>> block_chunk_sizes;
      std::vector<std::span<char>> block_compressed_buffers;
      std::vector<std::span<char>> block_output_buffers;
      for (size_t i = block; i < block_end; i++) {
        block_chunk_sizes.push_back(compressed_blocks[i].chunk_sizes);
        std::span<char> compress_span_gpu;
        void* compressed_buf_gpu = nullptr;

        if (from_cpu_mem) {
          // Load compressed data to GPU.
          gpu_run(cudaMallocAsync(&compressed_buf_gpu,
                                  compressed_blocks[i].block.size(), stream));
          gpu_run(cudaMemcpyAsync(compressed_buf_gpu,
                                  compressed_blocks[i].block.data(),
                                  compressed_blocks[i].block.size(),
                                  cudaMemcpyHostToDevice, stream));
          compress_span_gpu =
              std::span<char>(static_cast<char*>(compressed_buf_gpu),
                              compressed_blocks[i].block.size());
        } else {
          compress_span_gpu = std::span<char>(
              const_cast<char*>(compressed_blocks[i].block.data()),
              compressed_blocks[i].block.size());
        }
        block_compressed_buffers.push_back(compress_span_gpu);

        void* decompressed_buf_gpu = nullptr;
        std::span<char> decomp_span_gpu;
        gpu_run(cudaMallocAsync(&decompressed_buf_gpu, decomp_size, stream));
        decomp_span_gpu = std::span<char>(
            static_cast<char*>(decompressed_buf_gpu), decomp_size);
        block_output_buffers.push_back(decomp_span_gpu);
      }

      // Run decompression in GPU.
      int result = gpu_decompressor.batch_decompress_block_gpu(
          block_chunk_sizes, block_compressed_buffers, block_output_buffers,
          stream);
      CHECK_GE(result, 0);
      cudaStreamSynchronize(stream);
      int batch_decomp_size_bytes =
          gpu_decompressor.get_last_batch_bytes_decompressed();
      CHECK_GT(batch_decomp_size_bytes, 0) << "Decompression failed!";
      bytes_decompressed += batch_decomp_size_bytes;

      // Free memory.
      if (from_cpu_mem) {
        for (const auto& block_compressed_buffer : block_compressed_buffers) {
          gpu_run(cudaFree(block_compressed_buffer.data()));
        }
      }
      for (const auto& block_output_buffer : block_output_buffers) {
        gpu_run(cudaFree(block_output_buffer.data()));
      }
    }
    syncAndDestroyStream(stream);
  }

  //  {
  //    time_block t{[&](const auto& time) { decomp_time = time; }};
  //    for (size_t block = 0; block < compressed_blocks.size(); block +=
  //    block_batch_size) {
  //      if (block + block_batch_size >= compressed_blocks.size()) {
  //        block_end = compressed_blocks.size();
  //        current_batch_size = compressed_blocks.size() - block;
  //      } else {
  //        block_end = block + block_batch_size;
  //      }
  //
  //      cudaStream_t stream = nullptr;
  //      gpu_run(cudaStreamCreate(&stream));
  //      void *compressed_buf_gpu = nullptr;
  //      void *decompressed_buf_gpu = nullptr;
  //      std::span<char> compress_span_gpu;
  //      std::span<char> decomp_span_gpu;
  //      size_t decomp_size = 2_M * current_batch_size;
  //      int decompressed_chunk_size =
  //      compressed_blocks[block].decompressed_chunk_size;
  //
  //      uint32_t compressed_size = 0;
  //      size_t total_chunks = 0;
  //      for (size_t i = block; i < block_end; i++) {
  //        compressed_size += compressed_blocks[i].block.size();
  //        total_chunks += compressed_blocks[i].chunk_sizes.size();
  //      }
  //
  //      vector<uint32_t> chunk_sizes(total_chunks);
  //      size_t offset = 0;
  //      for (size_t i = block; i < block_end; i++) {
  //        for (const auto& chunk_size : compressed_blocks[i].chunk_sizes) {
  //          chunk_sizes[offset++] = chunk_size;
  //        }
  //      }
  //
  //      if (from_cpu_mem) {
  //        // Load compressed data to GPU.
  //        gpu_run(cudaMallocAsync(&compressed_buf_gpu, compressed_size,
  //                                stream));
  //        offset = 0;
  //        for (size_t i = block; i < block_end; i++) {
  //          gpu_run(cudaMemcpyAsync(static_cast<char*>(compressed_buf_gpu) +
  //          offset,
  //                                  compressed_blocks[i].block.data(),
  //                                  compressed_blocks[i].block.size(),
  //                                  cudaMemcpyHostToDevice, stream));
  //          offset += compressed_blocks[i].block.size();
  //        }
  //        compress_span_gpu = std::span<char>(
  //            static_cast<char*>(compressed_buf_gpu), compressed_size);
  //      } else {
  //        compress_span_gpu = std::span<char>(
  //            const_cast<char*>(compressed_blocks[block].block.data()),
  //            compressed_size);
  //      }
  //
  //      gpu_run(cudaMallocAsync(&decompressed_buf_gpu, decomp_size, stream));
  //      decomp_span_gpu = std::span<char>(
  //          static_cast<char*>(decompressed_buf_gpu), decomp_size);
  //
  //      // Run decompression in GPU.
  //      bytes_decompressed += decompress_block_gpu(
  //          chunk_sizes, compress_span_gpu, decomp_span_gpu,
  //          decompressed_chunk_size, stream);
  //
  //      // Free memory.
  //      if (from_cpu_mem) {
  //        gpu_run(cudaFree(compressed_buf_gpu));
  //      }
  //      gpu_run(cudaFree(decompressed_buf_gpu));
  //
  //      gpu_run(cudaStreamDestroy(stream));
  //    }
  //  }

  //  std::chrono::milliseconds decomp_time;
  //  uint64_t bytes_decompressed = 0;
  //  {
  //    time_block t{[&](const auto& time) { decomp_time = time; }};
  //    for (const auto & block : compressed_blocks) {
  //
  //      cudaStream_t stream = nullptr;
  //      gpu_run(cudaStreamCreate(&stream));
  //
  //      void *compressed_buf_gpu = nullptr;
  //      void *decompressed_buf_gpu = nullptr;
  //      std::span<char> compress_span_gpu;
  //      std::span<char> decomp_span_gpu;
  //      size_t decomp_size = 2_M;
  //
  //      if (from_cpu_mem) {
  //        // Load compressed data to GPU.
  //        gpu_run(cudaMallocAsync(&compressed_buf_gpu, block.block.size(),
  //                                stream));
  //        gpu_run(cudaMemcpyAsync(compressed_buf_gpu, block.block.data(),
  //                                block.block.size(), cudaMemcpyHostToDevice,
  //                                stream));
  //        compress_span_gpu = std::span<char>(
  //            static_cast<char*>(compressed_buf_gpu), block.block.size());
  //      } else {
  //        compress_span_gpu = std::span<char>(
  //            const_cast<char*>(block.block.data()), block.block.size());
  //      }
  //
  //      gpu_run(cudaMallocAsync(&decompressed_buf_gpu, decomp_size, stream));
  //      decomp_span_gpu = std::span<char>(
  //          static_cast<char*>(decompressed_buf_gpu), decomp_size);
  //
  //      // Run decompression in GPU.
  //      bytes_decompressed += decompress_block_gpu(
  //          block.chunk_sizes, compress_span_gpu, decomp_span_gpu,
  //          block.decompressed_chunk_size, stream);
  //
  //      // Free memory.
  //      if (from_cpu_mem) {
  //        gpu_run(cudaFree(compressed_buf_gpu));
  //      }
  //      gpu_run(cudaFree(decompressed_buf_gpu));
  //
  //      gpu_run(cudaStreamDestroy(stream));
  //    }
  //  }

  std::cout << "Decompressed " << bytes_decompressed << " bytes " << std::endl;

  return {bytes_decompressed,
          ((double)bytes_decompressed / (double)decomp_time.count()) * 1e-6,
          decomp_time};
}

[[maybe_unused]] static std::string benchmark_lz4_gpu_decompression(
    const CompressedFile& input_file, bool from_cpu_mem) {
  std::vector<CompressedFile::CompressedBlock> compressed_blocks =
      input_file.getBlocks();

  std::stringstream results;
  results << "file,time_ms,decompressed_size,compressed_size" << std::endl;

  for (int i = 0; i < 5; i++) {
    LOG(INFO) << "Iteration " << i;
    auto bench_res =
        benchmark_lz4_gpu_decompression(compressed_blocks, from_cpu_mem);

    auto res = input_file.m_file_md.data_file_path.filename().string() + ", " +
               std::to_string(bench_res.decomp_time.count()) + ", " +
               std::to_string(bench_res.decompression_throughput) + ", " +
               std::to_string(bench_res.decompressed_size) + ", " +
               std::to_string(input_file.m_data.getFileSize());

    results << res << std::endl;
  }

  return results.str();
}

std::string benchmark_lz4_gpu_decompression_gpu_mem(
    const CompressedFile& input_file) {
  LOG(INFO) << "Running benchmark_lz4_gpu_decompression_gpu_mem on "
            << input_file.m_file_md.data_file_path;
  return benchmark_lz4_gpu_decompression(input_file, false);
}

std::string benchmark_lz4_gpu_decompression_from_cpu_mem(
    const CompressedFile& input_file) {
  LOG(INFO) << "Running benchmark_lz4_gpu_decompression_from_cpu_mem on "
            << input_file.m_file_md.data_file_path;
  return benchmark_lz4_gpu_decompression(input_file, true);
}
