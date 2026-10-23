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

#include <filesystem>
#include <platform/memory/block-manager.hpp>
#include <storage/io_uring.hpp>
#include <storage/test/test-utils.hpp>

using namespace proteus::storage;

// printf("We are in test %s of test suite %s.\n",
//        test_info->name(),
//        test_info->test_suite_name());

class IoUringThreadUnsafeTest : public testing::Test {
 protected:
  void SetUp() override {
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    LOG(INFO) << "Running " << test_info->test_suite_name() << "."
              << test_info->name();
  }
};

TEST_F(IoUringThreadUnsafeTest,
       read_submit_and_flush_single_batch_smaller_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm100/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 64;
  EXPECT_LT(num_blocks_needed, max_inflight_requests)
      << "This test is intended to test the case where we submit a batch of "
         "requests that is smaller than the max number of inflight requests.";
  IoUringThreadUnsafe my_ring{max_inflight_requests};
  int num_completed = 0;  /// Doesn't need to be atomic because the callbacks
                          /// are called in this thread poll.
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
  }
  my_ring.submit();

  my_ring.flush();
  EXPECT_EQ(num_completed, num_blocks_needed);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);
  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}

TEST_F(IoUringThreadUnsafeTest,
       read_nosubmit_and_flush_single_batch_smaller_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm100/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 64;
  EXPECT_LT(num_blocks_needed, max_inflight_requests)
      << "This test is intended to test the case where we submit a batch of "
         "requests that is smaller than the max number of inflight requests.";
  IoUringThreadUnsafe my_ring{max_inflight_requests};
  int num_completed = 0;  /// Doesn't need to be atomic because the callbacks
                          /// are called in this thread poll.
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
  }

  my_ring.flush();
  EXPECT_EQ(num_completed, num_blocks_needed);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);
  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}

TEST_F(IoUringThreadUnsafeTest,
       read_submit_single_batch_smaller_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm100/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 64;
  EXPECT_LT(num_blocks_needed, max_inflight_requests)
      << "This test is intended to test the case where we submit a batch of "
         "requests that is smaller than the max number of inflight requests.";
  IoUringThreadUnsafe my_ring{max_inflight_requests};
  int num_completed = 0;  /// Doesn't need to be atomic because the callbacks
                          /// are called in this thread poll.
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
  }
  my_ring.submit();

  while (num_completed < num_blocks_needed) {
    my_ring.poll();
  }
  EXPECT_EQ(num_completed, num_blocks_needed);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);
  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}

TEST_F(IoUringThreadUnsafeTest,
       read_submit_single_batch_larger_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm100/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  int num_completed = 0;
  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 4;
  EXPECT_GT(num_blocks_needed, max_inflight_requests)
      << "This test is intended to test the case where we submit a batch of "
         "requests that is greater than the max number of inflight requests.";
  IoUringThreadUnsafe my_ring{max_inflight_requests};
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
  }
  my_ring.submit();

  while (num_completed < num_blocks_needed) {
    my_ring.poll();
  }
  EXPECT_EQ(num_completed, num_blocks_needed);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);

  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}

TEST_F(IoUringThreadUnsafeTest,
       read_submit_multiple_batches_smaller_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm1000/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  int num_completed = 0;
  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 16;
  const int submit_every_n = 4;
  EXPECT_GT(num_blocks_needed, max_inflight_requests)
      << "We want to test multiple batches";
  EXPECT_LT(submit_every_n, max_inflight_requests)
      << "We want our batches greater than max_inflight_requests";
  IoUringThreadUnsafe my_ring{max_inflight_requests};
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
    if (i % submit_every_n == 0) {
      my_ring.submit();
    }
  }
  my_ring.submit();

  while (num_completed < num_blocks_needed) {
    my_ring.poll();
  }
  EXPECT_EQ(num_completed, num_blocks_needed);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);

  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}

TEST_F(IoUringThreadUnsafeTest,
       read_submit_multiple_batches_greater_than_max_inflight) {
  const std::filesystem::path file_name =
      "inputs/ssbm1000/customer.csv.c_custkey";
  validateInputFile(file_name);
  auto fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
  const auto file_size = std::filesystem::file_size(file_name);
  PCHECK(fd > 0) "Failed to open file for reading: " << file_name;

  int num_completed = 0;
  // + 1 for non-round number of blocks;
  const int num_blocks_needed =
      file_size / BlockManager::block_size +
      (file_size % BlockManager::block_size == 0 ? 0 : 1);
  std::vector<proteus::managed_ptr> blocks{};
  for (int i = 0; i < num_blocks_needed; i++) {
    blocks.push_back(BlockManager::get_buffer());
  }

  LOG(INFO) << "num_blocks_needed " << num_blocks_needed;
  LOG(INFO) << "std::filesystem::file_size(file_name) "
            << std::filesystem::file_size(file_name);
  const int max_inflight_requests = 8;
  const int submit_every_n = 16;
  EXPECT_GT(num_blocks_needed, max_inflight_requests)
      << "We want to test multiple batches";
  EXPECT_GT(submit_every_n, max_inflight_requests)
      << "We want our batches smaller than max_inflight_requests";

  IoUringThreadUnsafe my_ring{max_inflight_requests};
  for (size_t i = 0; i < num_blocks_needed; i++) {
    EXPECT_LT(i * BlockManager::block_size,
              std::filesystem::file_size(file_name));
    EXPECT_EQ(BlockManager::block_size % 512, 0) << "we only support O_DIRECT";
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blocks.at(i).get()) % 512, 0);
    my_ring.read(fd, blocks.at(i).get(), BlockManager::block_size,
                 i * BlockManager::block_size,
                 [&num_completed]() { num_completed += 1; });
    if (i % submit_every_n == 0) {
      my_ring.submit();
    }
  }
  my_ring.submit();

  while (num_completed < num_blocks_needed) {
    my_ring.poll();
  }
  EXPECT_EQ(num_completed, num_blocks_needed);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  SCOPED_TRACE("");
#pragma clang diagnostic pop
  validateLoadedBlocks(blocks, file_name);

  for (int i = 0; i < num_blocks_needed; i++) {
    BlockManager::release_buffer(blocks.at(i).release());
  }
}
