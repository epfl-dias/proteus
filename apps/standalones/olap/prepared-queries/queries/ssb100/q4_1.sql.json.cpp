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

constexpr auto query = "ssb100_Q4_1";

PreparedStatement ssb100::Query::prepare41(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel51013 =
      morph.distribute_build(morph.scan("date", {"d_datekey", "d_year"}))
          .unpack();

  auto rel51018 =
      morph.distribute_build(morph.scan("part", {"p_partkey", "p_mfgr"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(
                eq(arg["p_mfgr"], "MFGR#1") | eq(arg["p_mfgr"], "MFGR#2"),
                expressions::Selectivity{2.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["p_partkey"]};
          });

  auto rel51023 =
      morph
          .distribute_build(
              morph.scan("customer", {"c_custkey", "c_nation", "c_region"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["c_region"], "AMERICA"),
                                     expressions::Selectivity{1.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["c_custkey"], arg["c_nation"]};
          });

  auto rel51028 =
      morph.distribute_build(morph.scan("supplier", {"s_suppkey", "s_region"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["s_region"], "AMERICA"),
                                     expressions::Selectivity{1.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["s_suppkey"]};
          });

  auto rel =
      morph
          .distribute_probe(morph.scan(
              "lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                            "lo_orderdate", "lo_revenue", "lo_supplycost"}))
          .unpack()
          .join(
              rel51028,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["s_suppkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              })
          .join(
              rel51023,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["c_custkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_custkey"];
              })
          .join(
              rel51018,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["p_partkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_partkey"];
              })
          .join(
              rel51013,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["d_datekey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_orderdate"];
              })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["d_year"].as("tmp", "d_year"),
                        arg["c_nation"].as("tmp", "c_nation")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["lo_revenue"] - arg["lo_supplycost"])
                                       .as("tmp", "profit"),
                                   1, 0, SUM}};
              },
              10, 128 * 1024)
          .pack();

  rel = morph.collect(rel)
            .unpack()
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["d_year"], arg["c_nation"]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
                },
                10, 64)
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["d_year"], arg["c_nation"], arg["profit"]};
                },
                {direction::ASC, direction::ASC, direction::NONE})
            .print(pg{"pm-csv"});
  return rel.prepare();
}
