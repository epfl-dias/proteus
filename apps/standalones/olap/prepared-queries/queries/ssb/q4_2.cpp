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

constexpr auto query = "ssb100_Q4_2";

PreparedStatement ssb::Query::prepare42(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel57931 = morph.scan("part", {"p_partkey", "p_mfgr", "p_category"});
  auto rel57935 = morph.scan("date", {"d_datekey", "d_year"});
  auto rel57940 = morph.scan("customer", {"c_custkey", "c_region"});
  auto rel57945 = morph.scan("supplier", {"s_suppkey", "s_nation", "s_region"});
  auto rel =
      morph.scan("lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                               "lo_orderdate", "lo_revenue", "lo_supplycost"});

  return morph
      .parallel(
          rel, {rel57931, rel57935, rel57940, rel57945},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel57931_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["p_mfgr"], "MFGR#1") |
                              eq(arg["p_mfgr"], "MFGR#2"),
                          expressions::Selectivity{2.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"], arg["p_category"]};
                    });

            auto rel57935_d = build.at(1).unpack().filter(
                [&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      eq(arg["d_year"], 1997) | eq(arg["d_year"], 1998),
                      expressions::Selectivity{2.0 / 7});
                });

            auto rel57940_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["c_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"]};
                    });

            auto rel57945_d =
                build.at(3)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["s_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"], arg["s_nation"]};
                    });

            return probe.unpack()
                .join(
                    rel57945_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_suppkey"];
                    })
                .join(
                    rel57935_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["d_datekey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_orderdate"];
                    })
                .join(
                    rel57940_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_custkey"];
                    })
                .join(
                    rel57931_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["d_year"].as("tmp", "d_year"),
                              arg["s_nation"].as("tmp", "s_nation"),
                              arg["p_category"].as("tmp", "p_category")};
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
            return {arg["d_year"], arg["s_nation"], arg["p_category"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
          },
          10, 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["s_nation"], arg["p_category"],
                    arg["profit"]};
          },
          {direction::ASC, direction::ASC, direction::ASC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
