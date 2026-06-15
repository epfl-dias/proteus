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

#include "cpu-benchmarks.hpp"

#include <latch>
#include <platform/threadpool/threadpool.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <sstream>

#include "compressed-file.hpp"
#include "lib/operators/mem-move/compression.hpp"


/**
 * @return count of bytes decompressed
 */
int64_t _thread_lz4_cpu_decompression(
    const std::vector<CompressedFile::CompressedBlock>& input_blocks,
    std::latch* latch) {
  auto& topo = topology::getInstance();
  CHECK_GT(input_blocks.size(), 0) << "need at least 1 block to decompress!";
  // we assume that all the input_blocks are on the same numa node
  auto& this_numa = topo.getNumaAddressed(input_blocks[0].block.data());

  // then we affinitize this thread to run on that numa node
  auto exec_scope = this_numa.set_on_scope();
  // not strictly necessary, but just being cautious here to ensure that the OS
  // is now scheduling this thread on the specified numa node.
  std::this_thread::yield();

  int64_t bytes_decompressed = 0;
  latch->arrive_and_wait(); /// wait for all threads to start
  for (const auto& block : input_blocks) {
    // TODO hardcoding block sizes
    // also allocating and deallocating in each loop to reflect "real" usage
    char* decompressed_buffer = new char[2_M];
    std::span<char> decompressed_span(decompressed_buffer, 2_M);
    bytes_decompressed +=
        decompress_block(block.chunk_sizes, block.block, decompressed_span,
                         block.decompressed_chunk_size);
    delete[] decompressed_buffer;
  }
  return bytes_decompressed;
}

/**
 * Benchmark decompressing a file with num_threads
 * @return string corresponding to a line of a csv file
 */
std::string _benchmark_lz4_cpu_decompression(const CompressedFile& input_file,
                                             int num_threads) {
  std::vector<CompressedFile::CompressedBlock> compressed_blocks =
      input_file.getBlocks();
  // Check the data was loaded to CPU memory and not GPU memory
  auto& topo = topology::getInstance();
  auto* cpu_numa_node =
      topo.getCpuNumaNodeAddressed(compressed_blocks[0].block.data());
  CHECK_NE(cpu_numa_node, nullptr) << "compressed block not on a CPU numa node";
  // create a vector of blocks for each thread
  std::vector<std::vector<CompressedFile::CompressedBlock>> thread_blocks(
      num_threads);
  for (int i = 0; i < compressed_blocks.size(); i++) {
    thread_blocks[i % num_threads].push_back(compressed_blocks[i]);
  }

  // using the global threadpool initialized by olap
  auto& tp = ThreadPool::getInstance();
  std::vector<std::future<int64_t>> futures;
  std::latch latch(num_threads + 1);  /// +1 for main thread
  for (uint32_t i = 0; i < num_threads; i++) {
    futures.emplace_back(
        tp.enqueue(_thread_lz4_cpu_decompression, thread_blocks[i], &latch));
  }

  std::chrono::milliseconds decomp_time;
  size_t total_decompressed_size = 0;
  latch.arrive_and_wait();
  {
    time_block t{[&](const auto& time) { decomp_time = time; }};
    for (auto& thread : futures) {
      total_decompressed_size += thread.get();
    }
  }
  LOG(INFO) << "Decompressed " << total_decompressed_size << " bytes in "
            << decomp_time.count() << "ms using " << num_threads << " threads";

  return input_file.m_file_md.data_file_path.filename().string() + "," +
         std::to_string(num_threads) + "," +
         std::to_string(decomp_time.count()) + "," +
         std::to_string(total_decompressed_size) + "," +
         std::to_string(input_file.m_data.getFileSize());
}

std::string benchmark_lz4_cpu_decompression(
    const CompressedFile& input_file, const std::vector<int>& decomp_threads) {
  LOG(INFO) << "Running benchmark_lz4_cpu_compression on "
            << input_file.m_file_md.data_file_path;
  std::stringstream results;
  results << "file,threads,time_ms,decompressed_size,compressed_size"
          << std::endl;
  // to a warmup run; we don't want to include this in the results
  _benchmark_lz4_cpu_decompression(input_file, decomp_threads.back());
  for (const auto& n_threads : decomp_threads) {
    LOG(INFO) << "Running benchmark_lz4_cpu_compression with " << n_threads
              << " threads";
    auto _ = _benchmark_lz4_cpu_decompression(input_file, n_threads);
    // run multiple iterations, take the average in Excel
    for (int i = 0; i < 5; i++) {
      LOG(INFO) << "Iteration " << i;
      auto res = _benchmark_lz4_cpu_decompression(input_file, n_threads);
      results << res << std::endl;
    }
  }

  return results.str();
}
