/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "utils.hpp"

#include <gtest/gtest.h>

#include <storage/mmap-file.hpp>

void validateLoadedBlocks(const std::vector<proteus::managed_ptr>& blocks,
                          const std::filesystem::path& input_file) {
  const auto baseline_data = mmap_file(input_file, PAGEABLE);
  const auto baseline_data_span = baseline_data.asSpan();
  EXPECT_GE(blocks.size() * BlockManager::block_size,
            baseline_data.getFileSize())
      << "not enough 2MiB blocks to contain the contents of "
      << input_file.string();
  size_t byte_index = 0;
  for (const auto& block : blocks) {
    auto bytes_to_compare =
        byte_index + BlockManager::block_size < baseline_data.getFileSize()
            ? BlockManager::block_size
            : baseline_data.getFileSize() - byte_index;
    auto comparison = memcmp(static_cast<std::byte*>(block.get()),
                             &baseline_data_span[byte_index], bytes_to_compare);
    EXPECT_EQ(comparison, 0)
        << "contents differ after: " << byte_index << " bytes";
    if (comparison != 0) {
      for (size_t i = 0; i < bytes_to_compare; i++) {
        ASSERT_EQ((static_cast<char*>(block.get()) + byte_index)[i],
                  (reinterpret_cast<const char*>(baseline_data.getData()) +
                   byte_index)[i])
            << " bytes differ at byte " << i << " of block "
            << byte_index / BlockManager::block_size;
      }
    }
    byte_index += BlockManager::block_size;
  }
}
