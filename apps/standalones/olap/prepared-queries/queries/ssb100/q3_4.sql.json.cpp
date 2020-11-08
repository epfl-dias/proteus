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

#include <ssb100/query.hpp>

constexpr auto query = "ssb100_Q3_4";

PreparedStatement ssb100::Query::prepare34(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel44853 =
      morph
          .distribute_build(
              morph.scan("date", {"d_datekey", "d_year", "d_yearmonth"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["d_yearmonth"], "Dec1997"),
                                     expressions::Selectivity{1.0 / 84});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_datekey"], arg["d_year"]};
          });

  auto rel44857 =
      morph.distribute_build(morph.scan("customer", {"c_custkey", "c_city"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["c_city"], "UNITED KI1") |
                                         eq(arg["c_city"], "UNITED KI5"),
                                     expressions::Selectivity{1.0 / 125});
          });

  auto rel44861 =
      morph.distribute_build(morph.scan("supplier", {"s_suppkey", "s_city"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["s_city"], "UNITED KI1") |
                                         eq(arg["s_city"], "UNITED KI5"),
                                     expressions::Selectivity{1.0 / 125});
          });

  auto rel =
      morph
          .distribute_probe(morph.scan(
              "lineorder",
              {"lo_custkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}))
          .unpack()
          .join(
              rel44861,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["s_suppkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              })
          .join(
              rel44857,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["c_custkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_custkey"];
              })
          .join(
              rel44853,
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
              10, 131072)
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
                10, 16)
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
