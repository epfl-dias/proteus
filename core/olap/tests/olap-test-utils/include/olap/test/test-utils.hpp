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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_
#include <olap/common/olap-common.hpp>

#include "gtest/gtest.h"

bool verifyTestResult(const char *testsPath, const char *testLabel,
                      bool unordered = false);
void runAndVerify(const char *testLabel, const char *planPath,
                  const char *testPath, const char *catalogJSON,
                  bool unordered = false);

#endif /* TEST_UTILS_HPP_ */
