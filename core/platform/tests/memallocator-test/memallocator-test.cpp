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

#include <iostream>
#include <platform/common/common.hpp>
#include <platform/memory/allocator.hpp>

class MemAllocatorTest : public ::testing::Test {};

using namespace std::chrono_literals;

template <size_t alignment>
void verifyAlignmnet() {
  struct alignas(alignment) X {};

  /*
   * For some reason the condition below breaks for alignment >= 512M
   * making this test invalid. (alignof(X) returns 1)
   */
  static_assert(alignment == alignof(X));

  auto allocator = ::proteus::memory::PinnedMemoryAllocator<X>{};
  auto alloced = allocator.allocate(3);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(alloced) % alignment, 0);
  allocator.deallocate(alloced, 3);
}

#define ALIGN_TEST(n) \
  TEST_F(MemAllocatorTest, align_##n) { verifyAlignmnet<n>(); }

ALIGN_TEST(1)
ALIGN_TEST(2)
ALIGN_TEST(4)
ALIGN_TEST(8)
ALIGN_TEST(16)
ALIGN_TEST(32)
ALIGN_TEST(64)
ALIGN_TEST(128)
ALIGN_TEST(256)
ALIGN_TEST(512)
ALIGN_TEST(1_K)
ALIGN_TEST(2_K)
ALIGN_TEST(4_K)
ALIGN_TEST(8_K)
ALIGN_TEST(16_K)
ALIGN_TEST(32_K)
ALIGN_TEST(64_K)
ALIGN_TEST(128_K)
ALIGN_TEST(256_K)
ALIGN_TEST(512_K)
ALIGN_TEST(1_M)
ALIGN_TEST(2_M)
ALIGN_TEST(4_M)
ALIGN_TEST(8_M)
ALIGN_TEST(16_M)
ALIGN_TEST(32_M)
ALIGN_TEST(64_M)
ALIGN_TEST(128_M)
ALIGN_TEST(256_M)
// See comment in function above for the following tests
// ALIGN_TEST(512_M)
// ALIGN_TEST(1_G)
// ALIGN_TEST(2_G)
// ALIGN_TEST(4_G)
