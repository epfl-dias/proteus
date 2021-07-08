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

#ifndef PROTEUS_OLAP_TEST_ENVIRONMENT_HPP
#define PROTEUS_OLAP_TEST_ENVIRONMENT_HPP

#include <gtest/gtest.h>

#include <olap/common/olap-common.hpp>

class OLAPTestEnvironment : public ::testing::Environment {
  bool is_noop = false;
  static bool has_already_been_setup;
  std::unique_ptr<proteus::olap> olap;

 public:
  void SetUp() override;
  void TearDown() override;
};

#endif  // PROTEUS_OLAP_TEST_ENVIRONMENT_HPP
