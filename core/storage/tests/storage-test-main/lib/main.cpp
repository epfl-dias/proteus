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

#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/util/glog.hpp>
#include <storage/test/test-utils.hpp>

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
  set_trace_allocations(false);

  ::testing::AddGlobalTestEnvironment(new StorageTestEnvironment);

  auto return_code = RUN_ALL_TESTS();
  google::ShutdownGoogleLogging();

  return return_code;
}
