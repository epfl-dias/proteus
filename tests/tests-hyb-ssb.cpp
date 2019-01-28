/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include "plan/plan-parser.hpp"
#include "storage/raw-storage-manager.hpp"
#include "test-utils.hpp"
#include "topology/topology.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include "util/raw-functions.hpp"
#include "util/raw-memory-manager.hpp"
#include "util/raw-pipeline.hpp"

#include <thread>
#include <vector>

using namespace llvm;

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new RawTestEnvironment);

class HYBSSBTest : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  void runAndVerify(const char *testLabel, const char *planPath,
                    bool unordered = false);

  bool flushResults = true;
  const char *testPath = TEST_OUTPUTS "/tests-hyb-ssb/";

  const char *catalogJSON = "inputs";

 public:
};

void HYBSSBTest::SetUp() { gpu_run(cudaSetDevice(0)); }

void HYBSSBTest::TearDown() { StorageManager::unloadAll(); }

void HYBSSBTest::runAndVerify(const char *testLabel, const char *planPath,
                              bool unordered) {
  ::runAndVerify(testLabel, planPath, testPath, catalogJSON, unordered);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_extendedprice*lo_discount) as revenue
//     from ssbm_lineorder, ssbm_date
//     where lo_orderdate = d_datekey
//      and d_year = 1993
//      and lo_discount between 1 and 3
//      and lo_quantity < 25;
TEST_F(HYBSSBTest, ssb_q1_1_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_quantity");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/lineorder.csv.lo_discount");

  const char *testLabel = "ssb_q1_1_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q1_1_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_extendedprice*lo_discount) as revenue
//     from ssbm_lineorder, ssbm_date
//     where lo_orderdate = d_datekey
//      and d_yearmonthnum = 199401
//      and lo_discount between 4 and 6
//      and lo_quantity between 26 and 35;
TEST_F(HYBSSBTest, ssb_q1_2_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_quantity");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
  load("inputs/ssbm100/lineorder.csv.lo_discount");
  load("inputs/ssbm100/date.csv.d_yearmonthnum");

  const char *testLabel = "ssb_q1_2_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q1_2_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_extendedprice*lo_discount) as revenue
//     from ssbm_lineorder, ssbm_date
//     where lo_orderdate = d_datekey
//      and d_weeknuminyear = 6
//      and d_year = 1994
//      and lo_discount between 5 and 7
//      and lo_quantity between 26 and 35;
TEST_F(HYBSSBTest, ssb_q1_3_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_quantity");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/lineorder.csv.lo_extendedprice");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/date.csv.d_weeknuminyear");
  load("inputs/ssbm100/lineorder.csv.lo_discount");

  const char *testLabel = "ssb_q1_3_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q1_3_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_revenue) as lo_revenue, d_year, p_brand1
//     from ssbm_lineorder, ssbm_date, ssbm_part, ssbm_supplier
//     where lo_orderdate = d_datekey
//      and lo_partkey = p_partkey
//      and lo_suppkey = s_suppkey
//      and p_category = 'MFGR#12'
//      and s_region = 'AMERICA'
//     group by d_year, p_brand1;
TEST_F(HYBSSBTest, ssb_q2_1_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/part.csv.p_category");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/part.csv.p_brand1");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q2_1_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q2_1_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_revenue) as lo_revenue, d_year, p_brand1
//     from ssbm_lineorder, ssbm_date, ssbm_part, ssbm_supplier
//     where lo_orderdate = d_datekey
//      and lo_partkey = p_partkey
//      and lo_suppkey = s_suppkey
//      and p_brand1 between 'MFGR#2221'
//      and 'MFGR#2228'
//      and s_region = 'ASIA'
//     group by d_year, p_brand1;
TEST_F(HYBSSBTest, ssb_q2_2_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/part.csv.p_brand1");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q2_2_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q2_2_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select sum(lo_revenue) as lo_revenue, d_year, p_brand1
//     from ssbm_lineorder, ssbm_date, ssbm_part, ssbm_supplier
//     where lo_orderdate = d_datekey
//      and lo_partkey = p_partkey
//      and lo_suppkey = s_suppkey
//      and p_brand1 = 'MFGR#2239'
//      and s_region = 'EUROPE'
//     group by d_year, p_brand1;
TEST_F(HYBSSBTest, ssb_q2_3_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/part.csv.p_brand1");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q2_3_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q2_3_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select c_nation, s_nation, d_year, sum(lo_revenue) as lo_revenue
//     from ssbm_customer, ssbm_lineorder, ssbm_supplier, ssbm_date
//     where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_orderdate = d_datekey
//      and c_region = 'ASIA'
//      and s_region = 'ASIA'
//      and d_year >= 1992
//      and d_year <= 1997
//     group by c_nation, s_nation, d_year;
TEST_F(HYBSSBTest, ssb_q3_1_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/supplier.csv.s_nation");
  load("inputs/ssbm100/customer.csv.c_region");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/customer.csv.c_nation");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q3_1_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q3_1_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
//     from ssbm_customer, ssbm_lineorder, ssbm_supplier, ssbm_date
//     where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_orderdate = d_datekey
//      and c_nation = 'UNITED STATES'
//      and s_nation = 'UNITED STATES'
//      and d_year >= 1992
//      and d_year <= 1997
//     group by c_city, s_city, d_year;
TEST_F(HYBSSBTest, ssb_q3_2_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/customer.csv.c_city");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/supplier.csv.s_nation");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/supplier.csv.s_city");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/customer.csv.c_nation");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q3_2_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q3_2_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
//     from ssbm_customer, ssbm_lineorder, ssbm_supplier, ssbm_date
//     where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_orderdate = d_datekey
//      and (c_city='UNITED KI1' or c_city='UNITED KI5')
//      and (s_city='UNITED KI1' or s_city='UNITED KI5')
//      and d_year >= 1992
//      and d_year <= 1997
//     group by c_city, s_city, d_year;
TEST_F(HYBSSBTest, ssb_q3_3_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/customer.csv.c_city");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/supplier.csv.s_city");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q3_3_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q3_3_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select c_city, s_city, d_year, sum(lo_revenue) as lo_revenue
//     from ssbm_customer, ssbm_lineorder, ssbm_supplier, ssbm_date
//     where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_orderdate = d_datekey
//      and (c_city='UNITED KI1' or c_city='UNITED KI5')
//      and (s_city='UNITED KI1' or s_city='UNITED KI5')
//      and d_yearmonth = 'Dec1997'
//     group by c_city, s_city, d_year;
TEST_F(HYBSSBTest, ssb_q3_4_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/customer.csv.c_city");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/date.csv.d_yearmonth");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/supplier.csv.s_city");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q3_4_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q3_4_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit
//     from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part, ssbm_lineorder
//     where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_partkey = p_partkey
//      and lo_orderdate = d_datekey
//      and c_region = 'AMERICA'
//      and s_region = 'AMERICA'
//      and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
//     group by d_year, c_nation;
TEST_F(HYBSSBTest, ssb_q4_1_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/lineorder.csv.lo_supplycost");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/customer.csv.c_region");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/customer.csv.c_nation");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");
  load("inputs/ssbm100/part.csv.p_mfgr");

  const char *testLabel = "ssb_q4_1_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q4_1_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as
//     profit from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part,
//     ssbm_lineorder where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_partkey = p_partkey
//      and lo_orderdate = d_datekey
//      and c_region = 'AMERICA'
//      and s_region = 'AMERICA'
//      and (d_year = 1997 or d_year = 1998)
//      and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
//     group by d_year, s_nation, p_category;
TEST_F(HYBSSBTest, ssb_q4_2_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/lineorder.csv.lo_supplycost");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/supplier.csv.s_nation");
  load("inputs/ssbm100/customer.csv.c_region");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/part.csv.p_category");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/supplier.csv.s_region");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/supplier.csv.s_suppkey");
  load("inputs/ssbm100/part.csv.p_mfgr");

  const char *testLabel = "ssb_q4_2_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q4_2_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}

// Options during test generation:
//     parallel    : True
//     memcpy      : True
//     cpu_only    : False
// Query:
//     select d_year, s_city, p_brand1, sum(lo_revenue - lo_supplycost) as
//     profit from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part,
//     ssbm_lineorder where lo_custkey = c_custkey
//      and lo_suppkey = s_suppkey
//      and lo_partkey = p_partkey
//      and lo_orderdate = d_datekey
//      and c_region = 'AMERICA'
//      and s_nation = 'UNITED STATES'
//      and (d_year = 1997 or d_year = 1998)
//      and p_category = 'MFGR#14'
//     group by d_year, s_city, p_brand1;
TEST_F(HYBSSBTest, ssb_q4_3_par_cpy) {
  if (topology::getInstance().getGpuCount() < 2) GTEST_SKIP();

  auto load = [](string filename) {
    StorageManager::loadToCpus(filename, sizeof(int32_t));
  };
  load("inputs/ssbm100/lineorder.csv.lo_supplycost");
  load("inputs/ssbm100/customer.csv.c_custkey");
  load("inputs/ssbm100/date.csv.d_datekey");
  load("inputs/ssbm100/supplier.csv.s_nation");
  load("inputs/ssbm100/customer.csv.c_region");
  load("inputs/ssbm100/lineorder.csv.lo_revenue");
  load("inputs/ssbm100/lineorder.csv.lo_partkey");
  load("inputs/ssbm100/part.csv.p_partkey");
  load("inputs/ssbm100/part.csv.p_category");
  load("inputs/ssbm100/lineorder.csv.lo_orderdate");
  load("inputs/ssbm100/supplier.csv.s_city");
  load("inputs/ssbm100/date.csv.d_year");
  load("inputs/ssbm100/lineorder.csv.lo_custkey");
  load("inputs/ssbm100/lineorder.csv.lo_suppkey");
  load("inputs/ssbm100/part.csv.p_brand1");
  load("inputs/ssbm100/supplier.csv.s_suppkey");

  const char *testLabel = "ssb_q4_3_par_cpy_hyb_plan";
  const char *planPath = "inputs/plans/hyb-ssb/ssb_q4_3_par_cpy_plan.json";

  runAndVerify(testLabel, planPath);
}
