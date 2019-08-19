/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include "gtest/gtest.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"

// #include "nvToolsExt.h"

// #include "llvm/DerivedTypes.h"
// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include "memory/memory-manager.hpp"
#include "plan/plan-parser.hpp"
#include "storage/storage-manager.hpp"

#include <numeric>
#include <thread>
#include <vector>

#include "test-utils.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

using namespace llvm;

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

static int page_size;

class SnapshottingTest : public ::testing::Test {
 protected:
  size_t N;

  virtual void SetUp();

  const char *testPath = TEST_OUTPUTS "/tests-snapshotting/";

  const char *catalogJSON = "inputs";

 public:
};

void SnapshottingTest::SetUp() {
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  N = size_bytes / sizeof(int);
}

template <typename Ca, typename Da>
void benchmark_writeall_sequencial(int *oltp_arena, size_t N,
                                   Ca create_snapshot, Da destroy_snapshot) {
  for (size_t j = 0; j < 10; ++j) {
    time_block t{"T: "};
    // Put some data
    for (size_t k = 0; k < N; ++k) {
      oltp_arena[k] = 0;
    }

    EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(0 * N));

    for (size_t i = 0; i < 10; ++i) {
      int *olap_arena = create_snapshot();
      // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

      // Try to write something from the OLTP side:
      for (size_t k = 0; k < N; ++k) {
        oltp_arena[k] = i + 1;
      }

      EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(i * N));
      EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
                (int)((i + 1) * N));
      // gpu_run(cudaHostUnregister(olap_arena));
      destroy_snapshot(olap_arena);
    }
  }
}

class cow_arenas_t {
  static int shm_fd;
  static size_t size_bytes;
  static int8_t *save_to;
  static int8_t *start;

 public:
  static void handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    //   printf("\nAttempt to access memory at address %p\n", siginfo->si_addr);
    // #ifdef LINUX_64BIT
    //   printf("Instruction pointer: %p\n",
    //          (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
    // #elif defined(LINUX_32BIT)
    //   printf("Instruction pointer: %p\n",
    //          (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
    // #endif
    void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));
    auto offset = ((int8_t *)addr) - start;

    // auto f = mremap(src2 + offset, page_size, page_size,
    //                 MREMAP_FIXED | MREMAP_MAYMOVE, addr);
    // assert(f != MAP_FAILED);
    // assert(f == addr);
    std::memcpy(save_to + offset, addr, page_size);

    mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    // signal(sig, SIG_DFL);
    // raise(sig);
#pragma clang diagnostic pop
  }

  static int *oltp_arena;
  int *olap_arena;

  static void init(size_t size_bytes) {
    cow_arenas_t::size_bytes = size_bytes;
    shm_fd = memfd_create("cow", 0);
    ASSERT_GE(shm_fd, 0);
    ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
    size_t N = size_bytes / sizeof(int);

    // Mark MAP_SHARED so that we can map the same memory multiple times
    oltp_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                             MAP_SHARED, shm_fd, 0);
    ASSERT_NE(oltp_arena, MAP_FAILED);
    start = (int8_t *)oltp_arena;

    // Put some data
    for (size_t i = 0; i < N; ++i) {
      oltp_arena[i] = i;
    }

    EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
              (int)(N * (N - 1) / 2));

    // Create snapshot

    // 0. Install handler to lazily steal and save data to-be-overwritten by
    // OLTP
    struct sigaction act {};
    sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    act.sa_sigaction = cow_arenas_t::handler;
#pragma clang diagnostic pop
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &act, nullptr);
  }

  int *oltp() { return oltp_arena; }
  int *olap() { return olap_arena; }

  void create_snapshot(void *place_at = nullptr) {
    // Should we mprotect before or after?
    // ANSWER:
    //    If we mprotect after, we may loose some updates!
    //    If we mprotect before, our handler may not know where to copy some
    //    updates!

    // 1. Mark OLTP side as RO
    assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

    // 2. Create a second mapping of the OLTP data for the OLAP side
    // 2.i   The mapping can not be MAP_SHARED as we do not want to create
    // changes private to the OLTP side.
    // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
    // side, we have to catch OLTP writes, still the underlying data and
    // replicate the, to the OLAP side.
    olap_arena =
        (int *)mmap(place_at, size_bytes, PROT_WRITE | PROT_READ,
                    MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
    assert(olap_arena != MAP_FAILED);
    save_to = (int8_t *)olap_arena;
    // return olap_arena;
  }

  void destroy_snapshot() {
    // time_block t{"T2: "};
    // Throw away snapshot
    munmap(olap_arena, size_bytes);
  }
};

TEST_F(SnapshottingTest, cow2) {
  std::string testLabel = "cow2";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  cow_arenas_t::init(size_bytes);

  size_t N = size_bytes / sizeof(int);

  cow_arenas_t arenas{};
  arenas.create_snapshot();
  // // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

  int *olap_arena = arenas.olap();
  int *oltp_arena = arenas.oltp();
  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 1;
  }

  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
            (int)(N * (N - 1) / 2));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)N);

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    ++(oltp_arena[i]);
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(2 * N));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 5;
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(2 * N));
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(5 * N));

  // benchmark_writeall_sequencial(oltp_arena, N, create_snapshot,
  //                               destroy_snapshot);
}

// TEST_F(SnapshottingTest, cow) {
//   std::string testLabel = "cow";
//   size_t num_of_pages = 1024;
//   page_size = sysconf(_SC_PAGE_SIZE);
//   size_t size_bytes = num_of_pages * page_size;
//   int shm_fd = memfd_create(testLabel.c_str(), 0);
//   ASSERT_GE(shm_fd, 0);
//   ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
//   size_t N = size_bytes / sizeof(int);

//   // Mark MAP_SHARED so that we can map the same memory multiple times
//   int *oltp_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
//                                 MAP_SHARED, shm_fd, 0);
//   ASSERT_NE(oltp_arena, MAP_FAILED);
//   start = (int8_t *)oltp_arena;

//   // Put some data
//   for (size_t i = 0; i < N; ++i) {
//     oltp_arena[i] = i;
//   }

//   EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
//             (int)(N * (N - 1) / 2));

//   // Create snapshot

//   // 0. Install handler to lazily steal and save data to-be-overwritten by
//   OLTP struct sigaction act {}; sigemptyset(&act.sa_mask);
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
//   act.sa_sigaction = handler;
// #pragma clang diagnostic pop
//   act.sa_flags = SA_SIGINFO | SA_ONSTACK;
//   sigaction(SIGSEGV, &act, nullptr);

//   auto create_snapshot = [&](void *place_at = nullptr) {
//     // Should we mprotect before or after?
//     // ANSWER:
//     //    If we mprotect after, we may loose some updates!
//     //    If we mprotect before, our handler may not know where to copy some
//     //    updates!

//     // 1. Mark OLTP side as RO
//     assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

//     // 2. Create a second mapping of the OLTP data for the OLAP side
//     // 2.i   The mapping can not be MAP_SHARED as we do not want to create
//     // changes private to the OLTP side.
//     // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
//     // side, we have to catch OLTP writes, still the underlying data and
//     // replicate the, to the OLAP side.
//     auto olap_arena =
//         (int *)mmap(place_at, size_bytes, PROT_WRITE | PROT_READ,
//                     MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
//     assert(olap_arena != MAP_FAILED);
//     save_to = (int8_t *)olap_arena;
//     return olap_arena;
//   };

//   auto destroy_snapshot = [&](void *olap_arena) {
//     // Throw away snapshot
//     munmap(olap_arena, size_bytes);
//   };

//   int *olap_arena = create_snapshot();
//   // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

//   // Try to write something from the OLTP side:
//   for (size_t i = 0; i < N; ++i) {
//     oltp_arena[i] = 1;
//   }

//   EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)N);
//   EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
//             (int)(N * (N - 1) / 2));

//   // To get the next snapshot, we have to throw away the current one,
//   otherwise
//   // we may have to update multiple ones upon a OLTP write.
//   destroy_snapshot(olap_arena);

//   olap_arena = create_snapshot();

//   // Try to write something from the OLTP side:
//   for (size_t i = 0; i < N; ++i) {
//     oltp_arena[i] = 2;
//   }

//   EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)N);
//   EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(2 * N));

//   // benchmark_writeall_sequencial(oltp_arena, N, create_snapshot,
//   //                               destroy_snapshot);
// }

// uint64_t *dirty;
// int8_t **find_at;
// int8_t *olap_start;
// size_t size_bytes;

// inline static void cor_fix_olap(int8_t *addr) {
//   auto offset = (addr - olap_start) / page_size;
//   mprotect(addr, page_size, PROT_READ | PROT_WRITE);
//   std::memcpy(addr, find_at[offset], page_size);

//   uint64_t mask = ((uint64_t)1) << (offset % 64);
//   dirty[offset / 64] &= ~mask;
//   find_at[offset] = nullptr;
// }

// static void handler_cor(int sig, siginfo_t *siginfo, void *uap) {
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
//   //   printf("\nAttempt to access memory at address %p\n",
//   siginfo->si_addr);
//   // #ifdef LINUX_64BIT
//   //   printf("Instruction pointer: %p\n",
//   //          (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
//   // #elif defined(LINUX_32BIT)
//   //   printf("Instruction pointer: %p\n",
//   //          (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
//   // #endif

//   void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

//   if (addr >= olap_start && addr < olap_start + size_bytes) {
//     cor_fix_olap((int8_t *)addr);
//   } else {
//     auto offset = (((int8_t *)addr) - start) / page_size;
//     uint64_t mask = ((uint64_t)1) << (offset % 64);
//     if (dirty[offset / 64] & mask) {
//       cor_fix_olap(((int8_t *)olap_start) + offset * page_size);
//       mprotect(addr, page_size, PROT_READ);
//     } else {
//       dirty[offset / 64] |= mask;
//       mprotect(addr, page_size, PROT_READ | PROT_WRITE);
//     }
//   }
// #pragma clang diagnostic pop
// }

// TEST_F(SnapshottingTest, cor) {
//   std::string testLabel = "cor";
//   size_t num_of_pages = 1024;
//   page_size = sysconf(_SC_PAGE_SIZE);
//   size_t size_bytes = num_of_pages * page_size;
//   ::size_bytes = size_bytes;
//   int shm_fd = memfd_create(testLabel.c_str(), 0);
//   ASSERT_GE(shm_fd, 0);
//   ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
//   size_t N = size_bytes / sizeof(int);

//   size_t dirty_segs = (num_of_pages + 63) / 64;
//   dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
//   find_at = (int8_t **)calloc(num_of_pages, sizeof(void *));

//   // Mark MAP_SHARED so that we can map the same memory multiple times
//   int *olap_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
//                                 MAP_SHARED, shm_fd, 0);
//   olap_start = (int8_t *)olap_arena;
//   ASSERT_NE(olap_arena, MAP_FAILED);

//   // Put some data
//   for (size_t i = 0; i < N; ++i) {
//     olap_arena[i] = i;
//   }

//   EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
//             (int)(N * (N - 1) / 2));

//   // Create snapshot

//   // 0. Install handler to lazily steal and save data to-be-overwritten by
//   OLTP struct sigaction act {}; sigemptyset(&act.sa_mask);
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
//   act.sa_sigaction = handler_cor;
// #pragma clang diagnostic pop
//   act.sa_flags = SA_SIGINFO | SA_ONSTACK;
//   sigaction(SIGSEGV, &act, nullptr);

//   class arenas_t {
//    public:
//     const int *olap;
//     int *oltp;
//   };

//   auto create_snapshot = [&](void *place_at = nullptr) {
//     // Should we mprotect before or after?
//     // ANSWER:
//     //    If we mprotect after, we may loose some updates!
//     //    If we mprotect before, our handler may not know where to copy some
//     //    updates!

//     // 1. Mark OLTP side as ROdirty
//     // assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

//     // TODO: fix dirty pages

//     // 2. Create a second mapping of the OLTP data for the OLAP side
//     // 2.i   The mapping can not be MAP_SHARED as we do not want to create
//     // changes private to the OLTP side.
//     // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
//     // side, we have to catch OLTP writes, still the underlying data and
//     // replicate the, to the OLAP side.
//     auto oltp_arena =
//         (int *)mmap(place_at, size_bytes, PROT_READ,
//                     MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
//     assert(oltp_arena != MAP_FAILED);
//     start = (int8_t *)oltp_arena;
//     for (size_t i = 0; i < dirty_segs; ++i) {
//       uint64_t d = dirty[i];
//       while (d) {
//         uint64_t bit = d & -d;
//         d ^= bit;
//         size_t bit_index = __builtin_popcountll(bit - 1);
//         size_t page_offset = i * 64 + bit_index;
//         size_t offset = page_offset * page_size;
//         assert(mprotect(((int8_t *)oltp_arena) + offset, page_size,
//                         // PROT_WRITE would be sufficient and prefered, but
//                         on
//                         // some architectures in also implies PROT_READ!
//                         PROT_NONE) >= 0);
//       }
//     }
//     return arenas_t{olap_arena, oltp_arena};
//   };

//   auto destroy_snapshot = [&](arenas_t arenas) {
//     const void *olap_arena = arenas.olap;
//     void *oltp_arena = arenas.oltp;
//     // TODO the eager version: instead of the code below, apply the changes
//     assert(mprotect(oltp_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
//     for (size_t i = 0; i < dirty_segs; ++i) {
//       uint64_t d = dirty[i];
//       while (d) {
//         uint64_t bit = d & -d;
//         d ^= bit;
//         size_t bit_index = __builtin_popcountll(bit - 1);
//         size_t page_offset = i * 64 + bit_index;
//         size_t offset = page_offset * page_size;
//         find_at[page_offset] = ((int8_t *)oltp_arena) + offset;
//         assert(mprotect(((int8_t *)olap_arena) + offset, page_size,
//                         // PROT_WRITE would be sufficient and prefered, but
//                         on
//                         // some architectures in also implies PROT_READ!
//                         PROT_NONE) >= 0);
//       }
//     }
//     // If we do not mremap oltp_arena, then the next oltp_arena will be in a
//     // different virtual address

//     // Throw away snapshot munmap(oltp_arena, size_bytes);
//   };

//   auto arenas = create_snapshot();
//   // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

//   // Try to write something from the OLTP side:
//   for (size_t i = 0; i < N; ++i) {
//     arenas.oltp[i] = 1;
//   }

//   EXPECT_EQ(std::accumulate(arenas.oltp, arenas.oltp + N, 0), (int)N);
//   EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0),
//             (int)(N * (N - 1) / 2));

//   // To get the next snapshot, we have to throw away the current one,
//   otherwise
//   // we may have to update multiple ones upon a OLTP write.
//   destroy_snapshot(arenas);

//   arenas = create_snapshot();

//   // EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)N);

//   // Try to write something from the OLTP side:
//   for (size_t i = 0; i < N; ++i) {
//     ++(arenas.oltp[i]);
//   }

//   EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)N);
//   EXPECT_EQ(std::accumulate(arenas.oltp, arenas.oltp + N, 0), (int)(2 * N));

//   // To get the next snapshot, we have to throw away the current one,
//   otherwise
//   // we may have to update multiple ones upon a OLTP write.
//   destroy_snapshot(arenas);

//   arenas = create_snapshot();

//   // Try to write something from the OLTP side:
//   for (size_t i = 0; i < N; ++i) {
//     arenas.oltp[i] = 5;
//   }

//   EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)(2 * N));
//   EXPECT_EQ(std::accumulate(arenas.oltp, arenas.oltp + N, 0), (int)(5 * N));

//   // benchmark_writeall_sequencial(oltp_arena, N, create_snapshot,
//   //                               destroy_snapshot);
// }

template <typename arenas_t>
void benchmark_writeall_sequencial(size_t N) {
  arenas_t::init(N * sizeof(int));

  arenas_t arenas{};
  for (size_t j = 0; j < 10; ++j) {
    time_block t{"T: "};
    arenas.create_snapshot();
    // Put some data
    int *oltp_arena = arenas.oltp();
    for (size_t k = 0; k < N; ++k) {
      oltp_arena[k] = 0;
    }

    EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(0 * N));

    arenas.destroy_snapshot();
    for (size_t i = 0; i < 10; ++i) {
      arenas.create_snapshot();
      int *olap_arena = arenas.olap();
      int *oltp_arena = arenas.oltp();
      // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

      // Try to write something from the OLTP side:
      for (size_t k = 0; k < N; ++k) {
        oltp_arena[k] = i + 1;
      }

      EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(i * N));
      EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
                (int)((i + 1) * N));
      // gpu_run(cudaHostUnregister(olap_arena));
      arenas.destroy_snapshot();
    }
  }
}

class cor_arenas_t {
 private:
  static uint64_t *dirty;
  static uint64_t *new_dirty;
  static int8_t **find_at;
  static int8_t *olap_start;
  static size_t size_bytes;

  int *oltp_arena;
  static int *olap_arena;
  static int shm_fd;
  static size_t dirty_segs;
  static int8_t *start;

 public:
  inline static void fix_olap(int8_t *addr) {
    auto offset = (addr - olap_start) / page_size;
    mprotect(addr, page_size, PROT_READ | PROT_WRITE);
    std::memcpy(addr, find_at[offset], page_size);

    uint64_t mask = ((uint64_t)1) << (offset % 64);
    dirty[offset / 64] &= ~mask;
    find_at[offset] = nullptr;
  }

  static void handler2(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    //     printf("\nAttempt to access memory at address %p\n",
    //     siginfo->si_addr);
    // #ifdef LINUX_64BIT
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
    // #elif defined(LINUX_32BIT)
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
    // #endif

    void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

    if (addr >= olap_start && addr < olap_start + size_bytes) {
      fix_olap((int8_t *)addr);
    } else {
      auto offset = (((int8_t *)addr) - start) / page_size;
      uint64_t mask = ((uint64_t)1) << (offset % 64);
      assert(offset / 64 <= dirty_segs);
      if (dirty[offset / 64] & mask) {
        fix_olap(((int8_t *)olap_start) + offset * page_size);
        mprotect(addr, page_size, PROT_READ);
      } else {
        // dirty[offset / 64] |= mask;
        mprotect(addr, page_size, PROT_READ | PROT_WRITE);
      }
    }
#pragma clang diagnostic pop
  }

  static void handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    //     printf("\nAttempt to access memory at address %p\n",
    //     siginfo->si_addr);
    // #ifdef LINUX_64BIT
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
    // #elif defined(LINUX_32BIT)
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
    // #endif

    void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

    if (addr >= olap_start && addr < olap_start + size_bytes) {
      fix_olap((int8_t *)addr);
    } else {
      auto offset = (((int8_t *)addr) - start) / page_size;
      uint64_t mask = ((uint64_t)1) << (offset % 64);
      assert(offset / 64 <= dirty_segs);
      new_dirty[offset / 64] |= mask;
      if (dirty[offset / 64] & mask) {
        {
          // auto offset = (addr - olap_start) / page_size;
          mprotect(addr, page_size, PROT_READ | PROT_WRITE);
          std::memcpy(addr, find_at[offset], page_size);

          // uint64_t mask = ((uint64_t)1) << (offset % 64);
          // dirty[offset / 64] &= ~mask;
        }
        // mprotect(addr, page_size, PROT_READ | PROT_WRITE);
      } else {
        // dirty[offset / 64] |= mask;
        mprotect(addr, page_size, PROT_READ | PROT_WRITE);
      }
    }
#pragma clang diagnostic pop
  }

 public:
  static void init(size_t size_bytes) {
    cor_arenas_t::size_bytes = size_bytes;
    shm_fd = memfd_create("cor_arenas_t", 0);
    ASSERT_GE(shm_fd, 0);
    ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
    size_t N = size_bytes / sizeof(int);
    size_t num_of_pages = size_bytes / page_size;

    dirty_segs = (num_of_pages + 63) / 64;
    dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
    new_dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
    find_at = (int8_t **)calloc(num_of_pages, sizeof(void *));

    // Mark MAP_SHARED so that we can map the same memory multiple times
    olap_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                             MAP_SHARED, shm_fd, 0);
    olap_start = (int8_t *)olap_arena;
    ASSERT_NE(olap_arena, MAP_FAILED);

    // Put some data
    for (size_t i = 0; i < N; ++i) {
      olap_arena[i] = i;
    }

    EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
              (int)(N * (N - 1) / 2));

    // Create snapshot

    // 0. Install handler to lazily steal and save data to-be-overwritten by
    // OLTP
    struct sigaction act {};
    sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    act.sa_sigaction = cor_arenas_t::handler;
#pragma clang diagnostic pop
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &act, nullptr);
  }

  int *oltp() { return oltp_arena; }
  int *olap() { return olap_arena; }

  void create_snapshot(void *place_at = nullptr) {
    // Should we mprotect before or after?
    // ANSWER:
    //    If we mprotect after, we may loose some updates!
    //    If we mprotect before, our handler may not know where to copy some
    //    updates!

    // 1. Mark OLTP side as ROdirty
    // assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

    // TODO: fix dirty pages

    // 2. Create a second mapping of the OLTP data for the OLAP side
    // 2.i   The mapping can not be MAP_SHARED as we do not want to create
    // changes private to the OLTP side.
    // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
    // side, we have to catch OLTP writes, still the underlying data and
    // replicate the, to the OLAP side.
    oltp_arena =
        (int *)mmap(place_at, size_bytes, PROT_READ,
                    MAP_PRIVATE | (place_at ? MAP_FIXED : 0), shm_fd, 0);
    assert(oltp_arena != MAP_FAILED);
    start = (int8_t *)oltp_arena;
    for (size_t i = 0; i < dirty_segs; ++i) {
      uint64_t d = dirty[i];
      while (d) {
        uint64_t bit = d & -d;
        d ^= bit;
        size_t bit_index = __builtin_popcountll(bit - 1);
        size_t page_offset = i * 64 + bit_index;
        size_t offset = page_offset * page_size;
        assert(mprotect(((int8_t *)oltp_arena) + offset, page_size,
                        // PROT_WRITE would be sufficient and prefered, but on
                        // some architectures in also implies PROT_READ!
                        PROT_NONE) >= 0);
      }
    }

    // return {olap_arena, oltp_arena};
  }

  void destroy_snapshot() {
    // time_block t{"T2: "};
    // TODO the eager version: instead of the code below, apply the changes
    assert(mprotect(oltp_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
    for (size_t i = 0; i < dirty_segs; ++i) {
      dirty[i] |= new_dirty[i];
      uint64_t d = dirty[i];
      while (d) {
        uint64_t bit = d & -d;
        d ^= bit;
        size_t bit_index = __builtin_popcountll(bit - 1);
        size_t page_offset = i * 64 + bit_index;
        size_t offset = page_offset * page_size;
        if (new_dirty[i] & bit) {
          find_at[page_offset] = ((int8_t *)oltp_arena) + offset;
        }
        assert(mprotect(((int8_t *)olap_arena) + offset, page_size,
                        // PROT_WRITE would be sufficient and prefered, but on
                        // some architectures in also implies PROT_READ!
                        PROT_NONE) >= 0);
      }
      new_dirty[i] = 0;
    }
    // If we do not mremap oltp_arena, then the next oltp_arena will be in a
    // different virtual address

    // Throw away snapshot munmap(oltp_arena, size_bytes);
  }
};

uint64_t *cor_arenas_t::dirty;
uint64_t *cor_arenas_t::new_dirty;
int8_t **cor_arenas_t::find_at;
int8_t *cor_arenas_t::olap_start;
size_t cor_arenas_t::size_bytes;

int *cor_arenas_t::olap_arena;
int cor_arenas_t::shm_fd;
size_t cor_arenas_t::dirty_segs;
int8_t *cor_arenas_t::start;

int cow_arenas_t::shm_fd;
size_t cow_arenas_t::size_bytes;
int8_t *cow_arenas_t::save_to;
int8_t *cow_arenas_t::start;
int *cow_arenas_t::oltp_arena;

class cor_const_arenas_t {
 protected:
  static uint64_t *dirty;
  static uint64_t *new_dirty;
  static int8_t **find_at;
  static int8_t *olap_start;
  static size_t size_bytes;

  static int *oltp_arena;
  static int *olap_arena;
  static int shm_fd;
  static size_t dirty_segs;
  static int8_t *start;

 public:
  static void handler(int sig, siginfo_t *siginfo, void *uap) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    // printf("\nAttempt to access memory at address %p\n", siginfo->si_addr);
    // #ifdef LINUX_64BIT
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[16]));
    // #elif defined(LINUX_32BIT)
    //     printf("Instruction pointer: %p\n",
    //            (((ucontext_t *)uap)->uc_mcontext.gregs[14]));
    // #endif

    void *addr = (void *)(((uintptr_t)siginfo->si_addr) & ~(page_size - 1));

    if (addr >= olap_start && addr < olap_start + size_bytes) {
      auto offset = (((int8_t *)addr) - olap_start) / page_size;

      mprotect(addr, page_size, PROT_READ | PROT_WRITE);
      mprotect(start + offset * page_size, page_size, PROT_READ);

      std::memcpy(addr, start + offset * page_size, page_size);
      uint64_t mask = ((uint64_t)1) << (offset % 64);
      assert(offset / 64 <= dirty_segs);
      dirty[offset / 64] &= ~mask;
      new_dirty[offset / 64] |= mask;

      // fix_olap((int8_t *)addr);
    } else {
      auto offset = (((int8_t *)addr) - start) / page_size;

      mprotect(olap_start + offset * page_size, page_size,
               PROT_READ | PROT_WRITE);
      mprotect(addr, page_size, PROT_READ | PROT_WRITE);

      uint64_t mask = ((uint64_t)1) << (offset % 64);
      assert(offset / 64 <= dirty_segs);
      if (!(new_dirty[offset / 64] & mask)) {
        std::memcpy(olap_start + offset * page_size, addr, page_size);
      }
      dirty[offset / 64] |= mask;
    }
#pragma clang diagnostic pop
  }

 public:
  static void init(size_t size_bytes) {
    cor_const_arenas_t::size_bytes = size_bytes;
    shm_fd = memfd_create("cor_const_arenas_t", 0);
    ASSERT_GE(shm_fd, 0);
    ASSERT_GE(ftruncate(shm_fd, size_bytes), 0);
    size_t N = size_bytes / sizeof(int);
    size_t num_of_pages = size_bytes / page_size;

    dirty_segs = (num_of_pages + 63) / 64;
    dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
    new_dirty = (uint64_t *)calloc(dirty_segs, sizeof(int64_t));
    find_at = (int8_t **)calloc(num_of_pages, sizeof(void *));

    // Mark MAP_SHARED so that we can map the same memory multiple times
    olap_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                             MAP_SHARED, shm_fd, 0);
    olap_start = (int8_t *)olap_arena;
    ASSERT_NE(olap_arena, MAP_FAILED);

    oltp_arena = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    start = (int8_t *)oltp_arena;
    assert(oltp_arena != MAP_FAILED);

    // Put some data
    for (size_t i = 0; i < N; ++i) {
      olap_arena[i] = i;
      oltp_arena[i] = i;
    }

    EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
              (int)(N * (N - 1) / 2));

    // Create snapshot

    // 0. Install handler to lazily steal and save data to-be-overwritten by
    // OLTP
    struct sigaction act {};
    sigemptyset(&act.sa_mask);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    act.sa_sigaction = cor_const_arenas_t::handler;
#pragma clang diagnostic pop
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &act, nullptr);
  }

  int *oltp() { return oltp_arena; }
  int *olap() { return olap_arena; }

  void create_snapshot(void *place_at = nullptr) {
    // Should we mprotect before or after?
    // ANSWER:
    //    If we mprotect after, we may loose some updates!
    //    If we mprotect before, our handler may not know where to copy some
    //    updates!

    // 1. Mark OLTP side as ROdirty
    // assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);

    // TODO: fix dirty pages

    // 2. Create a second mapping of the OLTP data for the OLAP side
    // 2.i   The mapping can not be MAP_SHARED as we do not want to create
    // changes private to the OLTP side.
    // 2.ii  In addition, as we want to hide the OLTP changes from the OLAP
    // side, we have to catch OLTP writes, still the underlying data and
    // replicate the, to the OLAP side.
    // assert(mprotect(olap_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
    assert(mprotect(oltp_arena, size_bytes, PROT_READ) >= 0);
    for (size_t i = 0; i < dirty_segs; ++i) {
      new_dirty[i] = 0;
      uint64_t d = dirty[i];
      while (d) {
        uint64_t bit = d & -d;
        d ^= bit;
        size_t bit_index = __builtin_popcountll(bit - 1);
        size_t page_offset = i * 64 + bit_index;
        size_t offset = page_offset * page_size;
        assert(mprotect(((int8_t *)olap_arena) + offset, page_size,
                        // PROT_WRITE would be sufficient and prefered, but on
                        // some architectures in also implies PROT_READ!
                        PROT_NONE) >= 0);
        // assert(mprotect(((int8_t *)oltp_arena) + offset, page_size,
        //                 // PROT_WRITE would be sufficient and prefered, but
        //                 on
        //                 // some architectures in also implies PROT_READ!
        //                 PROT_READ) >= 0);
      }
    }

    // return {olap_arena, oltp_arena};
  }

  void destroy_snapshot() {
    // // time_block t{"T2: "};
    // // TODO the eager version: instead of the code below, apply the changes
    // assert(mprotect(oltp_arena, size_bytes, PROT_WRITE | PROT_READ) >= 0);
    // for (size_t i = 0; i < dirty_segs; ++i) {
    //   dirty[i] |= new_dirty[i];
    //   uint64_t d = dirty[i];
    //   while (d) {
    //     uint64_t bit = d & -d;
    //     d ^= bit;
    //     size_t bit_index = __builtin_popcountll(bit - 1);
    //     size_t page_offset = i * 64 + bit_index;
    //     size_t offset = page_offset * page_size;
    //     if (new_dirty[i] & bit) {
    //       find_at[page_offset] = ((int8_t *)oltp_arena) + offset;
    //     }
    //     assert(mprotect(((int8_t *)olap_arena) + offset, page_size,
    //                     // PROT_WRITE would be sufficient and prefered, but
    //                     on
    //                     // some architectures in also implies PROT_READ!
    //                     PROT_NONE) >= 0);
    //   }
    //   new_dirty[i] = 0;
  }
  // If we do not mremap oltp_arena, then the next oltp_arena will be in a
  // different virtual address

  // Throw away snapshot munmap(oltp_arena, size_bytes);
};

uint64_t *cor_const_arenas_t::dirty;
uint64_t *cor_const_arenas_t::new_dirty;
int8_t **cor_const_arenas_t::find_at;
int8_t *cor_const_arenas_t::olap_start;
size_t cor_const_arenas_t::size_bytes;

int *cor_const_arenas_t::olap_arena;
int *cor_const_arenas_t::oltp_arena;
int cor_const_arenas_t::shm_fd;
size_t cor_const_arenas_t::dirty_segs;
int8_t *cor_const_arenas_t::start;

TEST_F(SnapshottingTest, cor2) {
  std::string testLabel = "cor";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  cor_arenas_t::init(size_bytes);

  size_t N = size_bytes / sizeof(int);

  cor_arenas_t arenas{};
  arenas.create_snapshot();
  // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

  int *olap_arena = arenas.olap();
  int *oltp_arena = arenas.oltp();
  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 1;
  }

  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
            (int)(N * (N - 1) / 2));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // EXPECT_EQ(std::accumulate(arenas.olap, arenas.olap + N, 0), (int)N);

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    ++(oltp_arena[i]);
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)N);
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(2 * N));

  // To get the next snapshot, we have to throw away the current one, otherwise
  // we may have to update multiple ones upon a OLTP write.
  arenas.destroy_snapshot();

  arenas.create_snapshot();
  olap_arena = arenas.olap();
  oltp_arena = arenas.oltp();

  // Try to write something from the OLTP side:
  for (size_t i = 0; i < N; ++i) {
    oltp_arena[i] = 5;
  }

  EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(2 * N));
  EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(5 * N));

  // benchmark_writeall_sequencial(oltp_arena, N, create_snapshot,
  //                               destroy_snapshot);

  benchmark_writeall_sequencial<cor_arenas_t>(N);
}

TEST_F(SnapshottingTest, cor) {
  std::string testLabel = "cor";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  size_t N = size_bytes / sizeof(int);
  benchmark_writeall_sequencial<cor_arenas_t>(N);
}

TEST_F(SnapshottingTest, cow) {
  std::string testLabel = "cow";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  size_t N = size_bytes / sizeof(int);
  benchmark_writeall_sequencial<cow_arenas_t>(N);
}

TEST_F(SnapshottingTest, cor_const) {
  std::string testLabel = "cor_const";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  size_t N = size_bytes / sizeof(int);
  benchmark_writeall_sequencial<cor_const_arenas_t>(N);
}

template <typename arenas_t>
void benchmark_writeone_random(size_t N) {
  arenas_t::init(N * sizeof(int));

  size_t m = 4;
  arenas_t arenas{};
  for (size_t j = 0; j < 10; ++j) {
    time_block t{"T: "};
    arenas.create_snapshot();
    // Put some data
    int *oltp_arena = arenas.oltp();
    for (size_t k = 0; k < N; ++k) {
      oltp_arena[k] = 0;
    }

    EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(0 * 4));

    arenas.destroy_snapshot();
    {
      arenas.create_snapshot();
      int *olap_arena = arenas.olap();
      int *oltp_arena = arenas.oltp();
      EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)(0 * 4));
      EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(0 * 4));
      arenas.destroy_snapshot();
    }
    for (size_t i = 0; i < 10; ++i) {
      arenas.create_snapshot();
      int *olap_arena = arenas.olap();
      int *oltp_arena = arenas.oltp();
      // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

      // Try to write something from the OLTP side:
      for (size_t k = 0; k < m; ++k) {
        ++(oltp_arena[(size_t)((rand() * 1.0 / RAND_MAX) * N)]);
      }

      EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(i * m));
      // EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
      //           (int)((i + 1) * 10));
      // gpu_run(cudaHostUnregister(olap_arena));
      arenas.destroy_snapshot();
    }
  }
}

TEST_F(SnapshottingTest, cor_random) {
  std::string testLabel = "cor_random";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  size_t N = size_bytes / sizeof(int);
  benchmark_writeone_random<cor_arenas_t>(N);
}

TEST_F(SnapshottingTest, cow_random) {
  std::string testLabel = "cow_random";
  size_t num_of_pages = 1024;
  page_size = sysconf(_SC_PAGE_SIZE);
  size_t size_bytes = num_of_pages * page_size;
  size_t N = size_bytes / sizeof(int);
  benchmark_writeone_random<cow_arenas_t>(N);
}

TEST_F(SnapshottingTest, cor_const_random) {
  std::string testLabel = "cor_const_random";
  benchmark_writeone_random<cor_const_arenas_t>(N);
}

template <typename arenas_t, bool cuda = false>
void benchmark_writeone_random_readsome(size_t N) {
  arenas_t::init(N * sizeof(int));

  int *g = nullptr;
  int *h = nullptr;
  if (cuda) {
    g = (int *)MemoryManager::mallocGpu(N * sizeof(int));
    h = (int *)MemoryManager::mallocPinned(N * sizeof(int));
  }

  // size_t m = 4;
  arenas_t arenas{};
  size_t sum = 0;
  for (size_t j = 0; j < 10; ++j) {
    time_block t{"T: "};
    if (cuda) {
      arenas.create_snapshot();
      // Put some data
      int *oltp_arena = arenas.oltp();
      for (size_t k = 0; k < N; ++k) {
        oltp_arena[k] = k;
      }

      EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
                (int)(N * (N - 1) / 2));

      arenas.destroy_snapshot();
      {
        arenas.create_snapshot();
        int *olap_arena = arenas.olap();
        int *oltp_arena = arenas.oltp();

        gpu_run(cudaMemcpy(g, olap_arena, N * sizeof(int), cudaMemcpyDefault));
        gpu_run(cudaMemcpy(h, g, N * sizeof(int), cudaMemcpyDefault));
        std::memcpy(oltp_arena, h, N * sizeof(int));

        EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
                  (int)(N * (N - 1) / 2));
        EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)(0 * 4));
        arenas.destroy_snapshot();
      }
      {
        arenas.create_snapshot();
        int *olap_arena = arenas.olap();
        int *oltp_arena = arenas.oltp();
        EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
                  (int)(N * (N - 1) / 2));
        EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0),
                  (int)(N * (N - 1) / 2));
        arenas.destroy_snapshot();
      }
    }
    //   {
    //     arenas.create_snapshot();

    //     int *oltp_arena = arenas.oltp();
    //     for (size_t k = 0; k < N; ++k) {
    //       oltp_arena[k] = 0;
    //     }

    //     arenas.destroy_snapshot();
    //   }
    //   {
    //     arenas.create_snapshot();
    //     int *olap_arena = arenas.olap();
    //     int *oltp_arena = arenas.oltp();
    //     EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0), (int)0);
    //     EXPECT_EQ(std::accumulate(olap_arena, olap_arena + N, 0), (int)0);
    //     arenas.destroy_snapshot();
    //   }
    //   for (size_t i = 0; i < 10; ++i) {
    //     arenas.create_snapshot();
    //     int *olap_arena = arenas.olap();
    //     int *oltp_arena = arenas.oltp();
    //     // gpu_run(cudaHostRegister(olap_arena, size_bytes, 0));

    //     // Try to write something from the OLTP side:
    //     for (size_t k = 0; k < m; ++k) {
    //       ++(oltp_arena[(size_t)((rand() * 1.0 / RAND_MAX) * N)]);
    //     }

    //     for (size_t k = 0; k < m; ++k) {
    //       sum += olap_arena[(size_t)((rand() * 1.0 / RAND_MAX) * N)];
    //     }

    //     // EXPECT_EQ(std::accumulate(olap_arena + j, olap_arena + N, 0),
    //     //           (int)(i * m));
    //     // EXPECT_EQ(std::accumulate(oltp_arena, oltp_arena + N, 0),
    //     //           (int)((i + 1) * 10));
    //     // gpu_run(cudaHostUnregister(olap_arena));
    //     arenas.destroy_snapshot();
    //   }
  }
  std::cout << sum << std::endl;

  if (cuda) MemoryManager::freeGpu(g);
}

class cuda_cor_const_arenas_t : public cor_const_arenas_t {
 public:
  static void init(size_t size_bytes) {
    cor_const_arenas_t::init(size_bytes);
    // gpu_run(cudaHostRegister(cor_const_arenas_t::olap_arena, size_bytes, 0));
    // gpu_run(cudaHostRegister(cor_const_arenas_t::oltp_arena, size_bytes, 0));
  }
};

TEST_F(SnapshottingTest, cor_random_readsome) {
  std::string testLabel = "cor_random_readsome";
  benchmark_writeone_random_readsome<cor_arenas_t>(N);
}

TEST_F(SnapshottingTest, cow_random_readsome) {
  std::string testLabel = "cow_random_readsome";
  benchmark_writeone_random_readsome<cow_arenas_t>(N);
}

TEST_F(SnapshottingTest, cor_const_random_readsome) {
  std::string testLabel = "cor_const_random_readsome";
  benchmark_writeone_random_readsome<cor_const_arenas_t>(N);
}

TEST_F(SnapshottingTest, cuda_cor_const_random_readsome) {
  std::string testLabel = "cuda_cor_const_random_readsome";
  benchmark_writeone_random_readsome<cuda_cor_const_arenas_t, true>(N);
}
