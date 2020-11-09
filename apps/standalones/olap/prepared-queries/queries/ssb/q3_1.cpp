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

constexpr auto query = "ssb100_Q3_1";

PreparedStatement ssb::Query::prepare31(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel29287 =
      morph.distribute_build(morph.scan("date", {"d_datekey", "d_year"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(
                ge(arg["d_year"], 1992) & le(arg["d_year"], 1997),
                expressions::Selectivity{6.0 / 7});
          });

  auto rel29292 =
      morph
          .distribute_build(
              morph.scan("customer", {"c_custkey", "c_nation", "c_region"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["c_region"], "ASIA"),
                                     expressions::Selectivity{1.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["c_custkey"], arg["c_nation"]};
          });

  auto rel29297 =
      morph
          .distribute_build(
              morph.scan("supplier", {"s_suppkey", "s_nation", "s_region"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["s_region"], "ASIA"),
                                     expressions::Selectivity{1.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["s_suppkey"], arg["s_nation"]};
          });

  auto rel =
      morph
          .distribute_probe(morph.scan(
              "lineorder",
              {"lo_custkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}))
          .unpack()
          .join(
              rel29292,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["c_custkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_custkey"];
              })
          .join(
              rel29297,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["s_suppkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              })
          .join(
              rel29287,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["d_datekey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_orderdate"];
              })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["c_nation"].as("tmp", "c_nation"),
                        arg["s_nation"].as("tmp", "s_nation"),
                        arg["d_year"].as("tmp", "d_year")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    arg["lo_revenue"].as("tmp", "lo_revenue"), 1, 0, SUM}};
              },
              10, 1024 * 1024)
          .pack();
  rel = morph.collect(rel)
            .unpack()
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["c_nation"], arg["s_nation"], arg["d_year"]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg["lo_revenue"], 1, 0, SUM}};
                },
                10, 131072)
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["c_nation"], arg["s_nation"], arg["d_year"],
                          arg["lo_revenue"]};
                },
                {direction::NONE, direction::NONE, direction::ASC,
                 direction::DESC})
            .print(pg{"pm-csv"});
  return rel.prepare();
}
