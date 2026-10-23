/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <gtest/gtest.h>
#include <fstream>

#include <filesystem>
#include <platform/memory/block-manager.hpp>
#include <storage/test/test-utils.hpp>
#include <vector>

void StorageTestEnvironment::SetUp() {
  assert(!has_already_been_setup);

  platform = std::make_unique<proteus::platform>(0.2, 0.1, 0);

  has_already_been_setup = true;
}

void StorageTestEnvironment::TearDown() { platform.reset(); }

bool StorageTestEnvironment::has_already_been_setup = false;

void validateInputFile(const std::filesystem::path& input_file) {
  EXPECT_TRUE(std::filesystem::exists(input_file))
      << "Make sure you have linked the data into $PROTEUS_DIR/tests/inputs "
         "and that the working directory is $CMAKE_BUILD_DIR/opt/pelago";
}

std::vector<std::byte> load_data(const std::filesystem::path& input_file) {
  validateInputFile(input_file);

  auto file_size = std::filesystem::file_size(input_file);
  std::vector<std::byte> data;
  std::ifstream file(input_file, std::ios::binary);

  data.resize(static_cast<std::size_t>(file_size));
  file.read(reinterpret_cast<char*>(data.data()), file_size);

  file.close();
  return data;
}

void validateLoadedBlocks(const std::vector<proteus::managed_ptr>& blocks,
                          const std::filesystem::path& input_file) {
  auto baseline_data = load_data(input_file);
  EXPECT_GT(blocks.size() * BlockManager::block_size, baseline_data.size())
      << "not enough 2MiB blocks to contain the contents of "
      << input_file.string();
  size_t byte_index = 0;
  for (const auto& block : blocks) {
    auto bytes_to_compare =
        byte_index + BlockManager::block_size < baseline_data.size()
            ? BlockManager::block_size
            : baseline_data.size() - byte_index;
    auto comparison =
        memcmp(static_cast<std::byte*>(block.get()),
               baseline_data.data() + byte_index, bytes_to_compare);
    EXPECT_EQ(comparison, 0)
        << "contents differ after: " << byte_index << "bytes";
    if (comparison != 0) {
      for (size_t i = 0; i < bytes_to_compare; i++) {
        ASSERT_EQ(
            (static_cast<char*>(block.get()) + byte_index)[i],
            (reinterpret_cast<char*>(baseline_data.data()) + byte_index)[i])
            << " bytes differ at " << byte_index + i;
      }
    }
    byte_index += BlockManager::block_size;
  }
}
