/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "codegen/util/parallel-context.hpp"
#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include "memory/memory-manager.hpp"
#include "plan/plan-parser.hpp"
#include "storage/storage-manager.hpp"
// #include <cuda_profiler_api.h>
#include "test-utils.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

#include <thread>
#include <vector>

using namespace llvm;

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class ThreadTest : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  void runAndVerify(const char *testLabel, const char *planPath,
                    bool unordered = false);

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-threads/";

  const char *catalogJSON = "inputs";

 public:
};

void ThreadTest::SetUp() { gpu_run(cudaSetDevice(0)); }

void ThreadTest::TearDown() { StorageManager::unloadAll(); }

void ThreadTest::runAndVerify(const char *testLabel, const char *planPath,
                              bool unordered) {
  ::runAndVerify(testLabel, planPath, testPath, catalogJSON, unordered);
}

TEST_F(ThreadTest, power9_getcpu) {
  const auto &topo = topology::getInstance();

  uint32_t target = std::min(topo.getCoreCount() / 2 + 1, topo.getCoreCount());

  cpu_set_t c;

  CPU_ZERO(&c);
  CPU_SET(target, &c);

  int s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &c);
  EXPECT_EQ(s, 0);

  int s2 = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &c);
  EXPECT_EQ(s2, 0);

  for (size_t i = 0; i < topo.getCoreCount(); ++i) {
    if (CPU_ISSET(i, &c)) std::cout << i << std::endl;
  }

  std::this_thread::yield();  // btw, is this necessary?

  // size_t s3 = 0;
  // for (int i = 0 ; i < 12310391023910 ; ++i) s3 += std::pow(5, i);
  // std::cout << s3 << std::endl;

  std::cout << "sched:" << sched_getcpu() << std::endl;
  EXPECT_NE(sched_getcpu(), 0);
  EXPECT_EQ(sched_getcpu(), (long)target);
}

TEST_F(ThreadTest, affinity) {
  const auto &topo = topology::getInstance();

  constexpr size_t spawned_threads = 1024;

  int tests[spawned_threads];

  for (size_t i = 0; i < spawned_threads; ++i) tests[i] = -1;

  std::vector<std::thread> threads;
  const auto &cpus = topo.getCpuNumaNodes();

  for (size_t i = 0; i < spawned_threads; ++i) {
    threads.emplace_back([&cpus, &tests, i] {
      const auto &cpu = cpus[i % cpus.size()];
      set_exec_location_on_scope el{cpu};

      tests[i] = sched_getcpu();
    });
  }
  for (auto &t : threads) t.join();

  for (size_t i = 0; i < spawned_threads; ++i) {
    const auto &cpu = cpus[i % cpus.size()];
    EXPECT_TRUE(CPU_ISSET(tests[i], &(cpu.local_cpu_set)));
  }
}

TEST_F(ThreadTest, local_memory_allocation) {
  const auto &topo = topology::getInstance();

  constexpr size_t spawned_threads = 1024;

  uint32_t tests[spawned_threads];

  for (size_t i = 0; i < spawned_threads; ++i) tests[i] = ~0;

  std::vector<std::thread> threads;
  const auto &cpus = topo.getCpuNumaNodes();

  for (size_t i = 0; i < spawned_threads; ++i) {
    threads.emplace_back([&cpus, &tests, &topo, i] {
      const auto &cpu = cpus[i % cpus.size()];
      set_exec_location_on_scope el{cpu};

      void *mem = MemoryManager::mallocPinned(1024);

      tests[i] = (topo.getCpuNumaNodeAddressed(mem)->id);

      MemoryManager::freePinned(mem);
    });
  }
  for (auto &t : threads) t.join();

  for (size_t i = 0; i < spawned_threads; ++i) {
    const auto &cpu = cpus[i % cpus.size()];
    EXPECT_EQ(tests[i], cpu.id);
  }
}
