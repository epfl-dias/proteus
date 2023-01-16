/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include <lib/plugins/csv-plugin-pm.hpp>
#include <lib/util/catalog.hpp>
#include <olap/expressions/expressions.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/test/test-utils.hpp>
#include <olap/values/expressionTypes.hpp>
#include <platform/common/common.hpp>
#include <storage/storage-manager.hpp>

#include "gtest/gtest.h"

class SailorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    catalog = &Catalog::getInstance();
    caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  void TearDown() override { StorageManager::getInstance().unloadAll(); }

  const char *testPath = TEST_OUTPUTS "/tests-sailors/";

 private:
  Catalog *catalog;
  CachingService *caches;
};

using namespace dangling_attr;

// Schema in cpp instead of read from a catalog
static auto sailors_csv = rel("inputs/sailors.csv")(
    Int("sid"), String("sname"), Int("rating"), Float("age"));

static auto boats_csv =
    rel("inputs/boats.csv")(Int("bid"), String("bname"), String("color"));

static auto reserves_csv =
    rel("inputs/reserves.csv")(Int("sid"), Int("bid"), String("day"));

TEST_F(SailorsTest, Scan) {
  const char *testLabel = "sailorsScan.csv";
  RelBuilderFactory factory{testLabel};

  expression_t constant_one = 1;
  constant_one.registerAs("tmp", "one");
  auto query = factory.getBuilder()
                   .scan(sailors_csv, {"sid", "age"}, "pm-csv")
                   .reduce(
                       [&](const auto &arg) -> std::vector<expression_t> {
                         return {arg["sid"].as("tmp1", "max_sid"),
                                 arg["sid"].as("tmp1", "sum_sid"),
                                 (expression_t{1}).as("tmp1", "count_sid")};
                       },
                       {MAX, SUM, SUM})
                   .print(pg{"pm-csv"});
  //  query result persists in shared memory with lifetime of result
  auto result = query.prepare().execute();
  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, Select) {
  const char *testLabel = "sailorsSel.csv";
  RelBuilderFactory factory{testLabel};

  auto query = factory.getBuilder()
                   .scan(sailors_csv, {"sid", "age"}, "pm-csv")
                   .filter([&](const auto &arg) -> expression_t {
                     return gt(arg["age"], 40.0);
                   })
                   .reduce(
                       [&](const auto &arg) -> std::vector<expression_t> {
                         return {arg["sid"].as("tmp2", "max_sid"),
                                 arg["sid"].as("tmp2", "sum_sid"),
                                 (expression_t{1}).as("tmp2", "count_sid")};
                       },
                       {MAX, SUM, SUM})
                   .print(pg{"pm-csv"});
  auto result = query.prepare().execute();
  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, ScanBoats) {
  const char *testLabel = "sailorsScanBoats.csv";
  RelBuilderFactory factory{testLabel};

  auto query = factory.getBuilder()
                   .scan(boats_csv, {"bid"}, "pm-csv")
                   .reduce(
                       [&](const auto &arg) -> std::vector<expression_t> {
                         return {arg["bid"].as("tmp3", "max_bid"),
                                 arg["bid"].as("tmp3", "sum_bid"),
                                 (expression_t{1}).as("tmp3", "count_bid")};
                       },
                       {MAX, SUM, SUM})
                   .print(pg{"pm-csv"});
  auto result = query.prepare().execute();
  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}

TEST_F(SailorsTest, JoinLeft3) {
  const char *testLabel = "sailorsJoinLeft3.csv";
  RelBuilderFactory factory{testLabel};

  auto sailors_scan =
      factory.getBuilder().scan(sailors_csv, {"sid", "age"}, "pm-csv");

  auto reserves_scan =
      factory.getBuilder().scan(reserves_csv, {"sid", "bid"}, "pm-csv");

  auto boats_scan = factory.getBuilder().scan(boats_csv, {"bid"}, "pm-csv");

  auto probe = sailors_scan
                   .join(
                       reserves_scan,
                       [&](const auto &build_arg) -> expression_t {
                         return build_arg["sid"];
                       },
                       [&](const auto &probe_arg) -> expression_t {
                         return probe_arg["sid"];
                       },
                       16, 128)
                   .join(
                       boats_scan,
                       [&](const auto &build_arg) -> expression_t {
                         return build_arg["bid"];
                       },
                       [&](const auto &probe_arg) -> expression_t {
                         return probe_arg["bid"];
                       },
                       16, 128)
                   .reduce(
                       [&](const auto &arg) -> std::vector<expression_t> {
                         return {arg["sid"].as("tmp4", "max_sid"),
                                 arg["sid"].as("tmp4", "sum_sid"),
                                 (expression_t{1}).as("tmp4", "count_sid")};
                       },
                       {MAX, SUM, SUM})
                   .print(pg{"pm-csv"});
  auto probe_result = probe.prepare().execute();
  EXPECT_TRUE(verifyTestResult(testPath, testLabel));
}
