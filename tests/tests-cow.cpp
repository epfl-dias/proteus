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

#include "codegen/util/parallel-context.hpp"
#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include "memory/memory-manager.hpp"
#include "plan/plan-parser.hpp"
#include "storage/storage-manager.hpp"
// #include <cuda_profiler_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <thread>
#include <vector>

#include "test-utils.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

using namespace llvm;

// ::testing::Environment *const pools_env =
//     ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class ThreadCOW : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  void runAndVerify(const char *testLabel, const char *planPath,
                    bool unordered = false);

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-cow/";

  const char *catalogJSON = "inputs";

 public:
};

void ThreadCOW::SetUp() {
  setbuf(stdout, nullptr);

  google::InstallFailureSignalHandler();

  set_trace_allocations(true, true);
}

void ThreadCOW::TearDown() { StorageManager::unloadAll(); }

void ThreadCOW::runAndVerify(const char *testLabel, const char *planPath,
                             bool unordered) {
  ::runAndVerify(testLabel, planPath, testPath, catalogJSON, unordered);
}

TEST_F(ThreadCOW, fork_cow_test) {
  std::string key = "/test_fork";
  std::string flag_key = "/test_fork_flag";
  size_t size_bytes = sizeof(int) * 100;

  int shm_fd_flag =
      shm_open(flag_key.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (shm_fd_flag == -1) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    EXPECT_TRUE(false);
  }

  if (ftruncate(shm_fd_flag, sizeof(int) * 3) < 0) {  //== -1){
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    EXPECT_TRUE(false);
  }

  int *flag = (int *)mmap(nullptr, sizeof(int) * 3, PROT_WRITE | PROT_READ,
                          MAP_SHARED, shm_fd_flag, 0);
  if (!flag) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    EXPECT_TRUE(false);
  }

  close(shm_fd_flag);

  // ---------

  int shm_fd = shm_open(key.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    EXPECT_TRUE(false);
  }

  if (ftruncate(shm_fd, size_bytes) < 0) {  //== -1){
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, strerror(errno));
    EXPECT_TRUE(false);
  }

  int *arr = (int *)mmap(nullptr, size_bytes, PROT_WRITE | PROT_READ,
                         MAP_PRIVATE, shm_fd, 0);

  // initializations
  for (int i = 0; i < 100; i++) arr[i] = 1;
  *flag = 0;
  *(flag + 1) = 0;
  *(flag + 2) = 0;

  // begin
  pid_t child_one = fork();

  if (child_one == 0) {
    // We are only allowed to init topology and all managers AFTER a fork!
    topology::init();

    // child_one
    int sum_child_one = 0;
    for (int i = 0; i < 100; i++) sum_child_one += arr[i];

    EXPECT_EQ(sum_child_one, 100);

    while (*flag == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(250));

    sum_child_one = 0;
    for (int i = 0; i < 100; i++) sum_child_one += arr[i];

    EXPECT_EQ(sum_child_one, 100);

    while (*flag == 1)
      std::this_thread::sleep_for(std::chrono::milliseconds(250));

    sum_child_one = 0;
    for (int i = 0; i < 100; i++) sum_child_one += arr[i];

    EXPECT_EQ(sum_child_one, 100);
    *(flag + 1) = 1;
    munmap(arr, size_bytes);
    munmap(flag, sizeof(int) * 3);

    exit(testing::Test::HasFailure());
  } else {
    // parent
    int sum_par_one = 0;
    for (int i = 0; i < 100; i++) sum_par_one += arr[i];
    EXPECT_EQ(sum_par_one, 100);

    for (int i = 0; i < 100; i++) arr[i] += 1;
    *flag = 1;

    sum_par_one = 0;
    for (int i = 0; i < 100; i++) sum_par_one += arr[i];
    EXPECT_EQ(sum_par_one, 200);

    pid_t child_two = fork();
    if (child_two == 0) {
      // We are only allowed to init topology and all managers AFTER a fork!
      topology::init();
      // child_two
      int sum_child_two = 0;
      for (int i = 0; i < 100; i++) sum_child_two += arr[i];

      EXPECT_EQ(sum_child_two, 200);

      while (*flag == 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

      sum_child_two = 0;
      for (int i = 0; i < 100; i++) sum_child_two += arr[i];
      EXPECT_EQ(sum_child_two, 200);
      *(flag + 2) = 1;
      munmap(arr, size_bytes);
      munmap(flag, sizeof(int) * 3);
      exit(testing::Test::HasFailure());
    } else {
      // We are only allowed to init topology and all managers AFTER a fork!
      topology::init();
      // parent

      int sum_par_two = 0;
      for (int i = 0; i < 100; i++) sum_par_two += arr[i];
      EXPECT_EQ(sum_par_two, 200);

      for (int i = 0; i < 100; i++) arr[i] += 1;
      *flag = 2;

      sum_par_two = 0;
      for (int i = 0; i < 100; i++) sum_par_two += arr[i];
      EXPECT_EQ(sum_par_two, 300);

      // finally kill all childs..
      while (*(flag + 1) != 1 && *(flag + 2) != 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

      munmap(arr, size_bytes);
      munmap(flag, sizeof(int) * 3);
    }
  }
}
