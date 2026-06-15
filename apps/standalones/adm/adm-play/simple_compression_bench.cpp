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
//
// Created by boffa on 20/09/24.
//
#include <lz4.h>
#include <lz4hc.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

size_t compressData(const std::vector<char>& data,
                    std::vector<char>& compressedData) {
  size_t compressedSize =
      LZ4_compress_fast(data.data(), compressedData.data(), data.size(),
                        compressedData.size(), 10);

  if (compressedSize <= 0) {
    throw std::runtime_error("Compression failed");
  }

  return compressedSize;
}

size_t compressData_HC(const std::vector<char>& data,
                       std::vector<char>& compressedData,
                       int compressionLevel) {
  size_t compressedSize =
      LZ4_compress_HC(data.data(), compressedData.data(), data.size(),
                      compressedData.size(), compressionLevel);

  if (compressedSize <= 0) {
    throw std::runtime_error("Compression failed");
  }

  return compressedSize;
}

// Just a simple benchmark to compare the throughput of LZ4 compression
// No blocks are used, just a single file (<2GB size) compressed with multiple
// threads The file is read from disk and compressed in memory

int main() {
  // generate data
  //  std::vector<char> data;
  //  size_t DATA_SIZE = 1 << 30;  // 1GB
  //  for (int i = 0; i < DATA_SIZE; i++) {
  //    data.push_back(std::rand() % 4 + 65);
  //  }

  fs::path input_file_path("/scratch/data/ssbm100/lineorder.csv.lo_linenumber");

  if (!fs::exists(input_file_path)) {
    std::cerr << "File does not exist: " << input_file_path << std::endl;
  }
  std::ifstream input_file(input_file_path, std::ios::binary);

  if (!input_file.is_open()) {
    std::cerr << "Failed to open file: " << input_file_path << std::endl;
  }

  // input_file.seekg(0, std::ios::end);
  // size_t file_size = input_file.tellg();
  // size_t file_size = (static_cast<size_t>(1) << 30);  //
  // file size = 1.5 GG
  size_t file_size = 1910612736;

  input_file.seekg(0, std::ios::beg);
  std::vector<char> data(file_size);
  input_file.read(data.data(), file_size);
  input_file.close();

  std::cout << "File size (GB): "
            << double(data.size()) / double(1024 * 1024 * 1024) << std::endl;

  // /scratch/data/ssbm1000/lineorder.csv.lo_linenumber

  // std::vector<char> data(DATA_SIZE, 'A');

  // size_t NUM_THREAD = 8;
  std::cout << "NUM_THREAD,CompressionLevel,CompressionRatio(%),"
               "UncompressedThroughput(MB/"
               "s),CompressedThroughput(MB/s)"
            << std::endl;
  std::vector<size_t> NUM_THREADS = {1, 2, 4, 8, 16, 32, 48, 64};
  auto compressionLevels = {0, 1, 3, 6, 9, 12, 15};

  for (auto NUM_THREAD : NUM_THREADS) {
    // std::vector<char> compressedData(data.size());

    // std::cout << "Original data size: " << data.size() << std::endl;

    for (auto compression_level : compressionLevels) {
      auto start_time = std::chrono::high_resolution_clock::now();
      size_t compressedSize = 0;

      if (compression_level == 0) {
        std::vector<std::thread> threads;
        size_t partSize = data.size() / NUM_THREAD;

        std::vector<std::vector<char>> compressedParts(
            NUM_THREAD, std::vector<char>(LZ4_compressBound(partSize)));

        start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_THREAD; ++i) {
          threads.emplace_back([&, i]() {
            size_t start = i * partSize;
            size_t end = (i == NUM_THREAD - 1) ? data.size() : start + partSize;
            std::vector<char> part(data.begin() + start, data.begin() + end);
            compressedParts[i].resize(LZ4_compressBound(part.size()));
            compressedSize += compressData(part, compressedParts[i]);
          });
        }

        for (auto& t : threads) {
          t.join();
        }
      } else {
        std::vector<std::thread> threads;
        size_t partSize = data.size() / NUM_THREAD;

        std::vector<std::vector<char>> compressedParts(
            NUM_THREAD, std::vector<char>(LZ4_compressBound(partSize)));

        start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_THREAD; ++i) {
          threads.emplace_back([&, i]() {
            size_t start = i * partSize;
            size_t end = (i == NUM_THREAD - 1) ? data.size() : start + partSize;
            std::vector<char> part(data.begin() + start, data.begin() + end);
            compressedParts[i].resize(LZ4_compressBound(part.size()));
            compressedSize +=
                compressData_HC(part, compressedParts[i], compression_level);
          });
        }

        for (auto& t : threads) {
          t.join();
        }
      }

      // size_t compressed_size = compressData(data, compressedData);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

      auto compression_ratio = compressedSize / (double)data.size() * 100;

      // compressedData.resize(compressedSize);

      // compressedData.resize(compressed_size);

      //    std::cout << "Compressed data size: " << compressedData.size() <<
      //    std::endl;
      //
      //    std::cout << "uncompressed bytes : " << std::endl;
      //    for (int i = 0; i < 100; i++) {
      //      std::cout << data[i] << " ";
      //    }
      //    std::cout << std::endl << "compressed bytes : " << std::endl;
      //    for (int i = 0; i < 100; i++) {
      //      std::cout << compressedData[i] << " ";
      //    }

      //    std::cout << std::endl
      //              << "Compression ratio : "
      //              << compressedSize / (double)data.size() * 100 << "%"
      //              << std::endl;

      std::cout << NUM_THREAD << "," << compression_level << ","
                << compression_ratio << ","
                << (data.size() * sizeof(char)) / (1024 * 1024) /
                       (elapsed / 1000.);

      std::cout << ","
                << (compressedSize * sizeof(char)) / (1024 * 1024) /
                       (elapsed / 1000.)
                << std::endl;
    }
  }
  return 0;
}