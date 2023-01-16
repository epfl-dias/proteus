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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/test/environment.hpp>
#include <olap/test/test-utils.hpp>
#include <storage/storage-manager.hpp>

#include "gtest/gtest.h"

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

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new OLAPTestEnvironment);

class OutputTest : public ::testing::Test {
 protected:
  const char *testPath = TEST_OUTPUTS "/tests-output/";
  const char *catalogJSON = "inputs";

  void runAndVerify(PreparedStatement &statement, const char *testLabel,
                    bool unordered = true) {
    auto qr = statement.execute();

    EXPECT_TRUE(verifyTestResult(testPath, testLabel, true));
  }
};

using namespace dangling_attr;

static auto sailors = rel("inputs/sailors.csv")(Int("sid"), String("sname"),
                                                Int("rating"), Float("age"));

// linehint = 10;
// policy = 2;
// pm::CSVPlugin *pgReserves =
//    openCSV(&ctx, reservesPath, reserveRec, reserveAttsToProject, ';',
//            linehint, policy, false);
static auto reserves =
    rel("inputs/reserves.csv")(Int("sid"), Int("bid"), String("day"));

//  linehint = 4;
//  policy = 2;
//  pm::CSVPlugin *pgBoats =
//      openCSV(&ctx, filenameBoats, recBoats, whichFieldsBoats, ';',
//      linehint,
//              policy, false);
static auto boats =
    rel("inputs/boats.csv")(Int("bid"), String("bname"), String("color"));

//  int linehint = 10;
//  int policy = 5;
//  char delimInner = '|';
//  pm::CSVPlugin *pg = openCSV(&ctx, lineitemPath, rec, projections,
//  delimInner,
//                              linehint, policy, false);

static auto lineitem = rel("inputs/tpch/lineitem10.csv")(
    Int("l_orderkey"), Int("l_partkey"), Int("l_suppkey"), Int("l_linenumber"),
    Float("l_quantity"), Float("l_extendedprice"), Float("l_discount"),
    Float("l_tax"), String("l_returnflag"), String("l_linestatus"),
    String("l_shipdate"), String("l_receiptdate"), String("l_shipinstruct"),
    String("l_comment"));

// works on new planner
// select max(sid) from sailors where age > 40 ;
TEST_F(OutputTest, ReduceNumeric) {
  const char *testLabel = "reduceNumeric.json";
  RelBuilderFactory factory{testLabel};

  auto statement = factory.getBuilder()
                       .scan(sailors, {"age", "sid"}, "pm-csv")
                       .filter([&](const auto &arg) -> expression_t {
                         return gt(arg["age"], 40.0);
                       })
                       .reduce(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sid"]};
                           },
                           {MAX})
                       .print(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sid"]};
                           },
                           pg{"pm-csv"})
                       .prepare();

  runAndVerify(statement, testLabel);
}

// works on new planner BUT planner does not request the output as json
// select sum(sid), max(sid) from sailors where age > 40 ;
TEST_F(OutputTest, MultiReduceNumeric) {
  const char *testLabel = "multiReduceNumeric.json";

  RelBuilderFactory factory{testLabel};

  auto statement = factory.getBuilder()
                       .scan(sailors, {"age", "sid"}, "pm-csv")
                       .filter([&](const auto &arg) -> expression_t {
                         return gt(arg["age"], 40.0);
                       })
                       .reduce(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sid"].as("tmp", "sum_sid"),
                                     arg["sid"].as("tmp", "max_sid")};
                           },
                           {SUM, MAX})
                       .print(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sum_sid"], arg["max_sid"]};
                           },
                           pg{"pm-csv"})
                       .prepare();

  runAndVerify(statement, testLabel);
}

// works on new planner BUT planner does not request the output as json
// select sid from sailors where age > 40 ;
TEST_F(OutputTest, ReduceBag) {
  const char *testLabel = "reduceBag.json";
  RelBuilderFactory factory{testLabel};

  auto statement = factory.getBuilder()
                       .scan(sailors, {"age", "sid"}, "pm-csv")
                       .filter([&](const auto &arg) -> expression_t {
                         return gt(arg["age"], 40.0);
                       })
                       .print(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sid"]};
                           },
                           pg("json"))
                       .prepare();  // FIMXE: ask for json

  runAndVerify(statement, testLabel);
}

// works on new planner BUT planner does not request the output as json
// select sid as id, age as age from sailors where age > 40 ;
TEST_F(OutputTest, ReduceBagRecord) {
  const char *testLabel = "reduceBagRecord.json";
  RelBuilderFactory factory{testLabel};

  auto statement = factory.getBuilder()
                       .scan(sailors, {"age", "sid"}, "pm-csv")
                       .filter([&](const auto &arg) -> expression_t {
                         return gt(arg["age"], 40.0);
                       })
                       .print(
                           [&](const auto &arg) -> std::vector<expression_t> {
                             return {arg["sid"].as("tmp", "id"), arg["age"]};
                           },
                           pg("json"))
                       .prepare();  // FIMXE: ask for json

  runAndVerify(statement, testLabel);
}

// table not in catalog/repo
TEST_F(OutputTest, NestBagTPCH) {
  const char *testLabel = "nestBagTPCH.json";
  RelBuilderFactory factory{testLabel};

  auto statement =
      factory.getBuilder()
          .scan(lineitem, {"l_orderkey", "l_linenumber", "l_quantity"},
                "pm-tsv")
          .filter([&](const auto &arg) -> expression_t {
            return lt(arg["l_orderkey"], 4);
          })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["l_linenumber"].as("tmp", "l_linenumber")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{expression_t{1}.as("tmp", "cnt"), 1, 0, SUM},
                    GpuAggrMatExpr{arg["l_quantity"].as("tmp", "max_qty"), 1,
                                   32, MAX}};
              },
              4, 16)
          .print(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["max_qty"]};
              },
              pg("json"))
          .prepare();

  runAndVerify(statement, testLabel);
}

TEST_F(OutputTest, JoinLeft3) {
  const char *testLabel = "3wayJoin.json";
  RelBuilderFactory factory{testLabel};

  auto statement =
      factory.getBuilder()
          .scan(reserves, {"sid", "bid"}, "pm-csv")
          .join(
              factory.getBuilder().scan(sailors, {"age", "sid"}, "pm-csv"),
              [&](const auto &build_arg) -> expression_t {
                return build_arg["sid"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["sid"];
              },
              4, 16)
          .join(
              factory.getBuilder().scan(boats, {"bid"}, "pm-csv"),
              [&](const auto &build_arg) -> expression_t {
                return build_arg["bid"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["bid"];
              },
              4, 16)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["sid"].as("tmp", "max_sid"),
                        arg["sid"].as("tmp", "sum_sid"),
                        expression_t{1}.as("tmp", "cnt")};
              },
              {MAX, SUM, SUM})
          .print(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["max_sid"], arg["sum_sid"], arg["cnt"]};
              },
              pg("json"))
          .prepare();

  runAndVerify(statement, testLabel);
}

/* Corresponds to plan parser tests */
TEST_F(OutputTest, NestReserves) {
  const char *testLabel = "nestReserves.json";
  RelBuilderFactory factory{testLabel};

  auto statement =
      factory.getBuilder()
          .scan(reserves, {"sid"}, "pm-csv")
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["sid"].as("tmp", "sid")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{expression_t{1}.as("tmp", "_groupCount"),
                                       1, 0, SUM}};
              },
              4, 16)
          .print(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["_groupCount"]};
              },
              pg("json"))
          .prepare();

  runAndVerify(statement, testLabel);
}

TEST_F(OutputTest, MultiNestReservesStaticAlloc) {
  const char *testLabel = "multinestReserves.json";
  RelBuilderFactory factory{testLabel};

  auto statement =
      factory.getBuilder()
          .scan(reserves, {"sid", "bid"}, "pm-csv")
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["sid"].as("tmp", "sid")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{expression_t{1}.as("tmp", "_outCount"),
                                       1, 0, SUM},
                        GpuAggrMatExpr{arg["bid"].as("tmp", "_outMax"), 1, 32,
                                       MAX}};
              },
              4, 16)
          .print(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["_outCount"], arg["_outMax"]};
              },
              pg("json"))
          .prepare();

  runAndVerify(statement, testLabel);
}
