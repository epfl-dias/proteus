/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "lib/operators/bloom-filter/bloom-filter-build.hpp"

class BloomFilterTest : public testing::TestWithParam<size_t> {
 public:
  BloomFilterTest() : bloom_filter_size(GetParam()) {}

  static void TearDownTestSuite() { StorageManager::getInstance().unloadAll(); }

 protected:
  const size_t bloom_filter_size;
  void SetUp() override {}

  void TearDown() override { cleanBloomFilterRegistry(); }

  static RelBuilderFactory getRelBuilderFactoryBuild() {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(test_info->test_suite_name()) + "_" +
                       test_info->name() + "_build";
    std::replace(name.begin(), name.end(), '/', '-');
    return RelBuilderFactory{name};
  }

  static RelBuilderFactory getRelBuilderFactoryProbe() {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(test_info->test_suite_name()) + "_" +
                       test_info->name() + "_probe";
    std::replace(name.begin(), name.end(), '/', '-');
    return RelBuilderFactory{name};
  }

  static size_t parse_count(const QueryResult& res) {
    std::stringstream res_str;
    res_str << res;
    std::string out_tuple;
    int tuple_count = 0;
    size_t probe_count = 0;

    while (std::getline(res_str, out_tuple)) {
      std::istringstream iss(out_tuple);
      std::string count;
      int attr = 0;
      while (std::getline(iss, count, ',')) {
        attr += 1;
        EXPECT_EQ(attr, 1)
            << "Expected only one attribute (the count) in the result";
        probe_count = std::stoull(count);
      }
      tuple_count += 1;
      EXPECT_EQ(tuple_count, 1)
          << "Expected only one tuple (the count) in the result";
    }
    EXPECT_EQ(tuple_count, 1)
        << "Expected only one tuple (the count) in the result";
    return probe_count;
  }
};

TEST_P(BloomFilterTest, SingleThreadEmptyFilter) {
  constexpr int filter_id = 1;

  auto rbf_build = getRelBuilderFactoryBuild();
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey", "d_year"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return (le(arg["d_year"], 1900));
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .bloomfilter_probe(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  EXPECT_EQ(probe_count, 0);

  size_t build_count = parse_count(build_res);
  EXPECT_EQ(build_count, 0);
}

TEST_P(BloomFilterTest, SingleThreadFullFilter) {
  constexpr int filter_id = 2;
  auto rbf_build = getRelBuilderFactoryBuild();
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey", "d_year"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return (ge(arg["d_year"], 1900) & le(arg["d_year"], 2100));
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .bloomfilter_probe(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  size_t build_count = parse_count(build_res);

  EXPECT_GT(probe_count, 0);
  EXPECT_GT(build_count, 0);
  ASSERT_EQ(probe_count, build_count);
  LOG(INFO) << "False positive rate: "
            << (static_cast<double>(probe_count - build_count) /
                static_cast<double>(probe_count));
}

TEST_P(BloomFilterTest, SingleThreadPartialFilter) {
  constexpr int filter_id = 3;
  auto rbf_build = getRelBuilderFactoryBuild();
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey", "d_year"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return (ge(arg["d_year"], 1994) & le(arg["d_year"], 1997));
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/date.csv", {"d_datekey"},
                CatalogParser::getInstance(), pg{"block"})
          .unpack()
          .bloomfilter_probe(
              [&](const auto& arg) -> expression_t { return arg["d_datekey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  size_t build_count = parse_count(build_res);

  EXPECT_GT(probe_count, 0);
  EXPECT_GT(build_count, 0);
  ASSERT_GE(probe_count, build_count);
  LOG(INFO) << "False positive rate: "
            << (static_cast<double>(probe_count - build_count) /
                static_cast<double>(probe_count))
            << " with bloom filter size " << bloom_filter_size;
}

TEST_P(BloomFilterTest, PartialFilter) {
  constexpr int filter_id = 4;
  auto dop =
      DegreeOfParallelism(topology::getInstance().getCpuNumaNodes().size());

  auto rbf_build = getRelBuilderFactoryBuild();
  // Broadcast the build side and build the bloom filter on each NUMA node
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey", "s_region"},
                CatalogParser::getInstance(), pg{"block"})
          .membrdcst(dop, true, true)
          .router(
              [&](const auto& arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              dop, 1, DeviceType::CPU,
              std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return eq(arg["s_region"], "ASIA");
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .router(dop, 2, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .bloomfilter_probe(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  // this has been multiplied by the build fanout (number of numa nodes) because
  // we did a broadcast.
  size_t build_count_broad_cast = parse_count(build_res);
  EXPECT_EQ(build_count_broad_cast % dop, 0);
  size_t build_count = build_count_broad_cast / dop;

  EXPECT_GT(probe_count, 0);
  EXPECT_GT(build_count, 0);
  ASSERT_GE(probe_count, build_count);
  LOG(INFO) << "False positive rate: "
            << (static_cast<double>(probe_count - build_count) /
                static_cast<double>(probe_count))
            << " with bloom filter size " << bloom_filter_size;
}

TEST_P(BloomFilterTest, PartialFilterPackRepack) {
  /// bloomfilter_repack collapses unpack and probe and pack into one operator
  constexpr int filter_id = 5;
  auto dop =
      DegreeOfParallelism(topology::getInstance().getCpuNumaNodes().size());

  auto rbf_build = getRelBuilderFactoryBuild();
  // Broadcast the build side and build the bloom filter on each NUMA node
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey", "s_region"},
                CatalogParser::getInstance(), pg{"block"})
          .membrdcst(dop, true, true)
          .router(
              [&](const auto& arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              dop, 1, DeviceType::CPU,
              std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return eq(arg["s_region"], "ASIA");
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .router(dop, 2, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .bloomfilter_repack(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .router(dop, 2, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  // this has been multiplied by the build fanout (number of numa nodes) because
  // we did a broadcast.
  size_t build_count_broad_cast = parse_count(build_res);
  EXPECT_EQ(build_count_broad_cast % dop, 0);
  size_t build_count = build_count_broad_cast / dop;

  EXPECT_GT(probe_count, 0);
  EXPECT_GT(build_count, 0);
  ASSERT_GE(probe_count, build_count);
  LOG(INFO) << "False positive rate: "
            << (static_cast<double>(probe_count - build_count) /
                static_cast<double>(probe_count))
            << " with bloom filter size " << bloom_filter_size;
}

TEST_P(BloomFilterTest, EmptyFilterPackRepack) {
  /// bloomfilter_repack collapses unpack and probe and pack into one operator
  constexpr int filter_id = 6;
  auto dop =
      DegreeOfParallelism(topology::getInstance().getCpuNumaNodes().size());

  auto rbf_build = getRelBuilderFactoryBuild();
  // Broadcast the build side and build the bloom filter on each NUMA node
  auto build_statement =
      rbf_build.getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .membrdcst(dop, true, true)
          .router(
              [&](const auto& arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              dop, 1, DeviceType::CPU,
              std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .filter([&](const auto& arg) -> expression_t {
            return eq(arg["s_suppkey"], 10000000);  // non-existent key
          })
          .bloomfilter_build(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  auto probe_statement =
      getRelBuilderFactoryProbe()
          .getBuilder()
          .scan("inputs/ssbm100/supplier.csv", {"s_suppkey"},
                CatalogParser::getInstance(), pg{"block"})
          .router(dop, 2, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .bloomfilter_repack(
              [&](const auto& arg) -> expression_t { return arg["s_suppkey"]; },
              bloom_filter_size, filter_id)
          .router(dop, 2, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<CpuNumaNodeAffinitizer>())
          .unpack()
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {expression_t{int64_t{1}}.as("tmp", "cnt")};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();
  auto build_res = build_statement.execute();
  auto probe_res = probe_statement.execute();
  size_t probe_count = parse_count(probe_res);
  // this has been multiplied by the build fanout (number of numa nodes) because
  // we did a broadcast.
  size_t build_count_broad_cast = parse_count(build_res);
  EXPECT_EQ(build_count_broad_cast % dop, 0);
  size_t build_count = build_count_broad_cast / dop;

  EXPECT_EQ(probe_count, 0);
  EXPECT_EQ(build_count, 0);
}

INSTANTIATE_TEST_SUITE_P(BloomFilterTest, BloomFilterTest,
                         testing::Values(4_K, 4_M, 512_M));
