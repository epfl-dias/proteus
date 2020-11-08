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

constexpr auto query = "ssb100_Q3_2";

PreparedStatement ssb::Query::prepare32(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel34584 =
      morph.distribute_build(morph.scan("date", {"d_datekey", "d_year"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(
                ge(arg["d_year"], 1992) & le(arg["d_year"], 1997),
                expressions::Selectivity{6.0 / 7});
          });

  auto rel34589 =
      morph
          .distribute_build(
              morph.scan("customer", {"c_custkey", "c_city", "c_nation"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["c_nation"], "UNITED STATES"),
                                     expressions::Selectivity{1.0 / 25});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["c_custkey"], arg["c_city"]};
          });

  auto rel34594 =
      morph
          .distribute_build(
              morph.scan("supplier", {"s_suppkey", "s_city", "s_nation"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["s_nation"], "UNITED STATES"),
                                     expressions::Selectivity{1.0 / 25});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["s_suppkey"], arg["s_city"]};
          });

  auto rel =
      morph
          .distribute_probe(morph.scan(
              "lineorder",
              {"lo_custkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}))
          .unpack()
          .join(
              rel34594,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["s_suppkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              })
          .join(
              rel34589,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["c_custkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_custkey"];
              })
          .join(
              rel34584,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["d_datekey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_orderdate"];
              })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["c_city"].as("tmp", "c_city"),
                        arg["s_city"].as("tmp", "s_city"),
                        arg["d_year"].as("tmp", "d_year")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    arg["lo_revenue"].as("tmp", "lo_revenue"), 1, 0, SUM}};
              },
              10, 128 * 1024)
          .pack();

  rel = morph.collect(rel)
            .unpack()
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["c_city"], arg["s_city"], arg["d_year"]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg["lo_revenue"], 1, 0, SUM}};
                },
                10, 128 * 1024)
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["c_city"], arg["s_city"], arg["d_year"],
                          arg["lo_revenue"]};
                },
                {direction::NONE, direction::NONE, direction::ASC,
                 direction::DESC})
            .print(pg{"pm-csv"});
  return rel.prepare();
}
