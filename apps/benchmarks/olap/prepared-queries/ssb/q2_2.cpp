/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include <ssb/query.hpp>

constexpr auto query = "ssb100_Q2_2";

PreparedStatement ssb::Query::prepare22(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel17693 = morph.scan("date", {"d_datekey", "d_year"});
  auto rel17698 = morph.scan("supplier", {"s_suppkey", "s_region"});
  auto rel17702 = morph.scan("part", {"p_partkey", "p_brand1"});

  auto rel = morph.scan(
      "lineorder", {"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"});

  return morph
      .parallel(
          rel, {rel17693, rel17698, rel17702},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel17693_d = build.at(0).unpack();

            auto rel17698_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["s_region"], "ASIA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"]};
                    });

            auto rel17702_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ge(arg["p_brand1"], "MFGR#2221") &
                              le(arg["p_brand1"], "MFGR#2228"),
                          expressions::Selectivity{1.0 / 125});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"], arg["p_brand1"]};
                    });

            return probe.unpack()
                .join(
                    rel17702_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .join(
                    rel17698_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_suppkey"];
                    })
                .join(
                    rel17693_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["d_datekey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_orderdate"];
                    })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {
                          arg["d_year"].as("PelagoAggregate#11428", "d_year"),
                          arg["p_brand1"].as("PelagoAggregate#11428",
                                             "p_brand1")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          (arg["lo_revenue"])
                              .as("PelagoAggregate#11428", "EXPR$0"),
                          1, 0, SUM}};
                    },
                    10, 128 * 1024)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["p_brand1"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["EXPR$0"], 1, 0, SUM}};
          },
          10, 128 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["EXPR$0"], arg["d_year"], arg["p_brand1"]};
          },
          {direction::NONE, direction::ASC, direction::ASC})
      .print(pg{"pm-csv"})
      .prepare();
}
