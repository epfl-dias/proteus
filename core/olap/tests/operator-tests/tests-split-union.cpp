/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/operators/relbuilder.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <storage/storage-manager.hpp>

class UnionTest : public testing::Test {
 public:
  static int pip_number;  // used to avoid duplicate pipeline names and the
                          // resulting issues of query results being saved to
                          // the same shared memory file
  static void TearDownTestSuite() { StorageManager::getInstance().unloadAll(); }

 protected:
  void SetUp() override {}

  void TearDown() override {}

  static RelBuilderFactory getRelBuilderFactory() {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(test_info->test_suite_name()) + "_" +
                       test_info->name() + "_pip" + std::to_string(pip_number);
    std::replace(name.begin(), name.end(), '/', '-');
    pip_number += 1;
    return RelBuilderFactory{name};
  }

  static std::pair<size_t, int32_t> parse_count_and_sum(
      const QueryResult& res) {
    std::stringstream res_str;
    res_str << res;
    std::string out_tuple;
    int tuple_count = 0;
    size_t count = 0;
    // this could wrap around in query processing, but that should not matter
    // for the test
    int32_t sum = 0;

    while (std::getline(res_str, out_tuple)) {
      std::istringstream iss(out_tuple);
      std::string attr_str;
      int attr_idx = 0;
      while (std::getline(iss, attr_str, ',')) {
        if (attr_idx == 0) {
          count = std::stoull(attr_str);
        }
        if (attr_idx == 1) {
          sum = std::stoi(attr_str);
        }
        attr_idx += 1;
        EXPECT_LE(attr_idx, 2)
            << "Expected only two attribute (count,sum) in the result";
      }
      tuple_count += 1;
      EXPECT_EQ(tuple_count, 1) << "Expected only one tuple in the result";
    }
    EXPECT_EQ(tuple_count, 1) << "Expected only one tuple in the result";
    return {count, sum};
  }
};

int UnionTest::pip_number = 0;

TEST_F(UnionTest, dop1_split) {
  auto rbf_baseline = getRelBuilderFactory();
  auto baseline_statement =
      rbf_baseline.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                        arg["lo_suppkey"].as("tmp", "sum")};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto baseline_res = baseline_statement.execute();
  auto [baseline_count, baseline_sum] = parse_count_and_sum(baseline_res);

  auto rbf_split = getRelBuilderFactory();
  auto split_statement =
      rbf_split.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .split(2, 2, RoutingPolicy::RANDOM);
  auto split_one = split_statement.unpack().reduce(
      [&](const auto& arg) -> std::vector<expression_t> {
        return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                arg["lo_suppkey"].as("tmp", "sum")};
      },
      {SUM, SUM});
  auto split_two = split_statement.unpack().reduce(
      [&](const auto& arg) -> std::vector<expression_t> {
        return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                arg["lo_suppkey"].as("tmp", "sum")};
      },
      {SUM, SUM});
  auto union_statement =
      split_one.unionAll({split_two})
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["sum"]};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto [union_count, union_sum] =
      parse_count_and_sum(union_statement.execute());
  EXPECT_EQ(baseline_count, union_count);
  EXPECT_EQ(baseline_sum, union_sum);
}

TEST_F(UnionTest, dop4_split_dop1union) {
  auto rbf_baseline = getRelBuilderFactory();
  auto baseline_statement =
      rbf_baseline.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                        arg["lo_suppkey"].as("tmp", "sum")};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto baseline_res = baseline_statement.execute();
  auto [baseline_count, baseline_sum] = parse_count_and_sum(baseline_res);

  constexpr int slack = 4;
  constexpr int dop_each_split_side = 4;
  auto rbf_split = getRelBuilderFactory();
  auto split_statement =
      rbf_split.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .split(2, 2, RoutingPolicy::RANDOM);
  auto split_one = split_statement
                       .router(DegreeOfParallelism(dop_each_split_side), slack,
                               RoutingPolicy::LOCAL, DeviceType::CPU,
                               std::make_unique<CpuNumaNodeAffinitizer>())
                       .unpack()
                       .reduce(
                           [&](const auto& arg) -> std::vector<expression_t> {
                             return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                                     arg["lo_suppkey"].as("tmp", "sum")};
                           },
                           {SUM, SUM});
  auto split_two = split_statement
                       .router(DegreeOfParallelism(dop_each_split_side), slack,
                               RoutingPolicy::LOCAL, DeviceType::CPU,
                               std::make_unique<CpuNumaNodeAffinitizer>())
                       .unpack()
                       .reduce(
                           [&](const auto& arg) -> std::vector<expression_t> {
                             return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                                     arg["lo_suppkey"].as("tmp", "sum")};
                           },
                           {SUM, SUM});
  auto union_statement =
      split_one.unionAll({split_two})
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["sum"]};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto [union_count, union_sum] =
      parse_count_and_sum(union_statement.execute());
  EXPECT_EQ(baseline_count, union_count);
  EXPECT_EQ(baseline_sum, union_sum);
}

TEST_F(UnionTest, dop4_split_dop2union) {
  auto rbf_baseline = getRelBuilderFactory();
  auto baseline_statement =
      rbf_baseline.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                        arg["lo_suppkey"].as("tmp", "sum")};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto baseline_res = baseline_statement.execute();
  auto [baseline_count, baseline_sum] = parse_count_and_sum(baseline_res);

  constexpr int slack = 4;
  constexpr int dop_each_split_side = 4;
  auto rbf_split = getRelBuilderFactory();
  auto split_statement =
      rbf_split.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .split(2, 2, RoutingPolicy::RANDOM);
  auto split_one = split_statement
                       .router(DegreeOfParallelism(dop_each_split_side), slack,
                               RoutingPolicy::LOCAL, DeviceType::CPU,
                               std::make_unique<CpuNumaNodeAffinitizer>())
                       .unpack()
                       .reduce(
                           [&](const auto& arg) -> std::vector<expression_t> {
                             return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                                     arg["lo_suppkey"].as("tmp", "sum")};
                           },
                           {SUM, SUM});
  auto split_two = split_statement
                       .router(DegreeOfParallelism(dop_each_split_side), slack,
                               RoutingPolicy::LOCAL, DeviceType::CPU,
                               std::make_unique<CpuNumaNodeAffinitizer>())
                       .unpack()
                       .reduce(
                           [&](const auto& arg) -> std::vector<expression_t> {
                             return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                                     arg["lo_suppkey"].as("tmp", "sum")};
                           },
                           {SUM, SUM});
  auto union_statement =
      split_one.unionAll({split_two}, DegreeOfParallelism(2))
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["sum"]};
              },
              {SUM, SUM})
          .router(DegreeOfParallelism(1), slack, RoutingPolicy::LOCAL,
                  DeviceType::CPU, std::make_unique<CpuNumaNodeAffinitizer>())
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["sum"]};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto [union_count, union_sum] =
      parse_count_and_sum(union_statement.execute());
  EXPECT_EQ(baseline_count, union_count);
  EXPECT_EQ(baseline_sum, union_sum);
}

TEST_F(UnionTest, packed_dop4_split_dop2union) {
  // in this test the Union routes packed blocks instead of tuples like above
  auto rbf_baseline = getRelBuilderFactory();
  auto baseline_statement =
      rbf_baseline.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return ge(arg["lo_suppkey"], 10);
          })
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                        arg["lo_suppkey"].as("tmp", "sum")};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto baseline_res = baseline_statement.execute();
  auto [baseline_count, baseline_sum] = parse_count_and_sum(baseline_res);

  constexpr int slack = 4;
  constexpr int dop_each_split_side = 4;
  auto rbf_split = getRelBuilderFactory();
  auto split_statement =
      rbf_split.getBuilder()
          .scan("inputs/ssbm100/lineorder.csv", {"lo_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .split(2, 2, RoutingPolicy::RANDOM);
  auto split_one =
      split_statement
          .router(DegreeOfParallelism(dop_each_split_side), slack,
                  RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          // arbitrary filter just to have some operator in the split pipelines
          .filter([&](const auto& arg) -> expression_t {
            return ge(arg["lo_suppkey"], 10);
          })
          .pack();

  auto split_two = split_statement
                       .router(DegreeOfParallelism(dop_each_split_side), slack,
                               RoutingPolicy::LOCAL, DeviceType::CPU,
                               std::make_unique<CpuNumaNodeAffinitizer>())
                       .unpack()
                       .filter([&](const auto& arg) -> expression_t {
                         return ge(arg["lo_suppkey"], 10);
                       })
                       .pack();

  auto union_statement =
      split_one.unionAll({split_two}, DegreeOfParallelism(2))
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt"),
                        arg["lo_suppkey"].as("tmp", "sum")};
              },
              {SUM, SUM})
          .router(DegreeOfParallelism(1), slack, RoutingPolicy::LOCAL,
                  DeviceType::CPU, std::make_unique<CpuNumaNodeAffinitizer>())
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"], arg["sum"]};
              },
              {SUM, SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto [union_count, union_sum] =
      parse_count_and_sum(union_statement.execute());
  EXPECT_EQ(baseline_count, union_count);
  EXPECT_EQ(baseline_sum, union_sum);
}
