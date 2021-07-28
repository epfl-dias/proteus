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

constexpr auto query = "ssb100_Q2_1";

PreparedStatement ssb::Query::prepare21(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel11407 = morph.scan("date", {"d_datekey", "d_year"});

  auto rel11412 = morph.scan("supplier", {"s_suppkey", "s_region"});

  auto rel11417 = morph.scan("part", {"p_partkey", "p_category", "p_brand1"});

  auto rel = morph.scan(
      "lineorder", {"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"});

  return morph
      .parallel(
          rel, {rel11407, rel11412, rel11417},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel11407_d = build.at(0).unpack();

            auto rel11412_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["s_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"]};
                    });

            auto rel11417_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["p_category"], "MFGR#12"),
                          expressions::Selectivity{1.0 / 25});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"], arg["p_brand1"]};
                    });

            return probe.unpack()
                .join(
                    rel11417_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .join(
                    rel11412_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_suppkey"];
                    })
                .join(
                    rel11407_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["d_datekey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_orderdate"];
                    })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["d_year"].as("PelagoProject#11438", "d_year"),
                              arg["p_brand1"].as("PelagoProject#11438",
                                                 "p_brand1")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          arg["lo_revenue"].as("PelagoProject#11438", "EXPR$0"),
                          1, 0, SUM}};
                    },
                    10, 1024 * 1024)
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
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {arg["EXPR$0"], arg["d_year"], arg["p_brand1"]};
      })
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["EXPR$0"], arg["d_year"], arg["p_brand1"]};
          },
          {direction::NONE, direction::ASC, direction::ASC})
      .print(pg{"pm-csv"})
      .prepare();
}
