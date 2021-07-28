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

constexpr auto query = "ssb100_Q4_3";

PreparedStatement ssb::Query::prepare43(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel64283 = morph.scan("date", {"d_datekey", "d_year"});
  auto rel64288 = morph.scan("customer", {"c_custkey", "c_region"});
  auto rel64293 = morph.scan("part", {"p_partkey", "p_category", "p_brand1"});
  auto rel64298 = morph.scan("supplier", {"s_suppkey", "s_city", "s_nation"});
  auto rel =
      morph.scan("lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                               "lo_orderdate", "lo_revenue", "lo_supplycost"});

  return morph
      .parallel(
          rel, {rel64283, rel64288, rel64293, rel64298},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel64283_d = build.at(0).unpack().filter(
                [&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      eq(arg["d_year"], 1997) | eq(arg["d_year"], 1998),
                      expressions::Selectivity{2.0 / 7});
                });

            auto rel64288_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["c_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"]};
                    });

            auto rel64293_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["p_category"], "MFGR#14"),
                          expressions::Selectivity{1.0 / 25});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"], arg["p_brand1"]};
                    });

            auto rel64298_d =
                build.at(3)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["s_nation"], "UNITED STATES"),
                          expressions::Selectivity{1.0 / 25});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"], arg["s_city"]};
                    });

            return probe.unpack()
                .join(
                    rel64298_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_suppkey"];
                    })
                .join(
                    rel64293_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .join(
                    rel64288_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_custkey"];
                    })
                .join(
                    rel64283_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["d_datekey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_orderdate"];
                    })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["d_year"].as("tmp", "d_year"),
                              arg["s_city"].as("tmp", "s_city"),
                              arg["p_brand1"].as("tmp", "p_brand1")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          (arg["lo_revenue"] - arg["lo_supplycost"])
                              .as("tmp", "profit"),
                          1, 0, SUM}};
                    },
                    10, 128 * 1024)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["s_city"], arg["p_brand1"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
          },
          10, 16 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["s_city"], arg["p_brand1"],
                    arg["profit"]};
          },
          {direction::ASC, direction::ASC, direction::ASC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
