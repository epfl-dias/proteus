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
#include <numaif.h>

#include <iostream>
#include <platform/common/common.hpp>
#include <platform/memory/allocator.hpp>

class MemAllocatorTest : public ::testing::Test {};

using namespace std::chrono_literals;

template <size_t alignment>
void verifyAlignment() {
  struct alignas(alignment) X {};

  /*
   * For some reason the condition below breaks for alignment >= 512M
   * making this test invalid. (alignof(X) returns 1)
   */
  static_assert(alignment == alignof(X));

  auto allocator = ::proteus::memory::PinnedMemoryAllocator<X>{};
  auto allocated = allocator.allocate(3);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(allocated) % alignment, 0);
  EXPECT_TRUE(MemoryManager::is_aligned(allocated, alignment));
  allocator.deallocate(allocated, 3);
}

#define ALIGN_TEST(n) \
  TEST_F(MemAllocatorTest, align_##n) { verifyAlignment<n>(); }

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

void verifyCpuNumaAlloc(int cpuNumaNodeIndex) {
  auto allocator =
      ::proteus::memory::ExplicitSocketPinnedMemoryAllocator<int32_t>{
          cpuNumaNodeIndex};
  auto allocated = allocator.allocate(3);
  auto& topo = topology::getInstance();
  auto cpu_node_addressed = topo.getCpuNumaNodeAddressed(allocated);
  ASSERT_EQ(cpu_node_addressed->index_in_topo, cpuNumaNodeIndex)
      << " Allocated on unexpected node";

  int actual_numa_id = -1;
  auto x =
      get_mempolicy(&actual_numa_id, nullptr, 0, static_cast<void*>(allocated),
                    MPOL_F_NODE | MPOL_F_ADDR);
  ASSERT_EQ(x, 0) << "Failed to get NUMA policy";
  ASSERT_EQ(cpu_node_addressed->id, actual_numa_id);
  allocator.deallocate(allocated, 3);
}

TEST_F(MemAllocatorTest, verifyCpuNumaAlloc) {
  auto& topo = topology::getInstance();
  LOG(INFO) << "Found " << topo.getCpuNumaNodeCount() << " cpuNumaNodes";

  for (auto i = 0; i < topo.getCpuNumaNodeCount(); i++) {
    LOG(INFO) << "Testing for cpuNumaNodes: " << i;
    verifyCpuNumaAlloc(i);
  }
}

template <size_t alignment>
void verifyMallocPinnedAligned() {
  auto allocated = MemoryManager::mallocPinnedAligned(1024, alignment);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(allocated) % alignment, 0);
  EXPECT_TRUE(MemoryManager::is_aligned(allocated, alignment));
  MemoryManager::freePinnedAligned(allocated);
}

#define ALIGNED_ALLOC_TEST(n)                                     \
  TEST_F(MemAllocatorTest, verifyMallocPinnedAligned_align_##n) { \
    verifyMallocPinnedAligned<n>();                               \
  }

ALIGNED_ALLOC_TEST(1)
ALIGNED_ALLOC_TEST(2)
ALIGNED_ALLOC_TEST(4)
ALIGNED_ALLOC_TEST(8)
ALIGNED_ALLOC_TEST(16)
ALIGNED_ALLOC_TEST(32)
ALIGNED_ALLOC_TEST(64)
ALIGNED_ALLOC_TEST(128)
ALIGNED_ALLOC_TEST(256)
ALIGNED_ALLOC_TEST(512)
ALIGNED_ALLOC_TEST(1_K)
ALIGNED_ALLOC_TEST(2_K)
ALIGNED_ALLOC_TEST(4_K)
ALIGNED_ALLOC_TEST(8_K)
ALIGNED_ALLOC_TEST(16_K)
ALIGNED_ALLOC_TEST(32_K)
ALIGNED_ALLOC_TEST(64_K)
ALIGNED_ALLOC_TEST(128_K)
ALIGNED_ALLOC_TEST(256_K)
ALIGNED_ALLOC_TEST(512_K)
ALIGNED_ALLOC_TEST(1_M)
ALIGNED_ALLOC_TEST(2_M)
ALIGNED_ALLOC_TEST(4_M)
ALIGNED_ALLOC_TEST(8_M)
ALIGNED_ALLOC_TEST(16_M)
ALIGNED_ALLOC_TEST(32_M)
ALIGNED_ALLOC_TEST(64_M)
ALIGNED_ALLOC_TEST(128_M)
ALIGNED_ALLOC_TEST(256_M)
ALIGNED_ALLOC_TEST(512_M)
ALIGNED_ALLOC_TEST(1_G)
ALIGNED_ALLOC_TEST(2_G)
ALIGNED_ALLOC_TEST(4_G)
