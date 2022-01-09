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

#include <gtest/gtest.h>

#include <platform/util/glog.hpp>
#include <storage/mmap-file.hpp>

#include "storage-test.hpp"

void segfaultBlockbacked(const void* data) {
  const int* int_data = reinterpret_cast<const int*>(data);
  LOG(INFO) << int_data[0];
}

TEST(mmapFile, testVirtual) {
  // Here nothing should have actually been loaded and we should segfault if we
  // try to access the data
  std::string input_file = "inputs/ssbm100/customer.csv.c_custkey";
  validateInputFile(input_file);
  auto anon_mmap_file = mmap_file(input_file, VIRTUAL);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
  EXPECT_DEATH(segfaultBlockbacked(anon_mmap_file.getData()), ".*");
#pragma clang diagnostic pop
}
