/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#include <glog/logging.h>

#include <filesystem>

#include "gtest/gtest.h"
#include "platform/common/common.hpp"
#include "platform/memory/memory-manager.hpp"
#include "storage-test.hpp"

/*
 * We have our own test main because there is some set up that should be done
 * for all storage tests.
 *
 * If you need setup that is specific to your test suite, write a test harness.
 * Do not add it here.
 */

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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging((argv)[0]);
  FLAGS_logtostderr = true;
  LOG(INFO) << "Running in: " << std::filesystem::current_path() << " \n";

  setbuf(stdout, nullptr);

  // for reproducibility
  srand(time(nullptr));
  google::InstallFailureSignalHandler();
  // for debugging:
  set_trace_allocations(true);

  ::testing::AddGlobalTestEnvironment(new StorageTestEnvironment);

  return RUN_ALL_TESTS();
}
