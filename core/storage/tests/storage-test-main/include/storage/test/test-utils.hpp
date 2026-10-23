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

#ifndef PROTEUS_TEST_UTILS_HPP
#define PROTEUS_TEST_UTILS_HPP

#include <filesystem>
#include <platform/common/common.hpp>
#include <platform/memory/managed-pointer.hpp>

class StorageTestEnvironment : public ::testing::Environment {
  static bool has_already_been_setup;
  std::unique_ptr<proteus::platform> platform;

 public:
  void SetUp() override;
  void TearDown() override;
};

/**
 * EXPECT_TRUE that @param input_file exists
 */
void validateInputFile(const std::filesystem::path& input_file);

/**
 * EXPECT_TRUE that the data in pointed to by blocks matches the contents of
 * input_file
 * @param blocks Assumed to be 2 MiB bytes
 * @param input_file path to file
 */
void validateLoadedBlocks(const std::vector<proteus::managed_ptr>& blocks,
                          const std::filesystem::path& input_file);

#endif  // PROTEUS_TEST_UTILS_HPP
