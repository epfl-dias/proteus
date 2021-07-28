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

constexpr auto query = "ssb100_Q4_1";

PreparedStatement ssb::Query::prepare41(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel51013 = morph.scan("date", {"d_datekey", "d_year"});
  auto rel51018 = morph.scan("part", {"p_partkey", "p_mfgr"});
  auto rel51023 = morph.scan("customer", {"c_custkey", "c_nation", "c_region"});
  auto rel51028 = morph.scan("supplier", {"s_suppkey", "s_region"});
  auto rel =
      morph.scan("lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                               "lo_orderdate", "lo_revenue", "lo_supplycost"});

  return morph
      .parallel(
          rel, {rel51013, rel51018, rel51023, rel51028},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel51013_d = build.at(0).unpack();

            auto rel51018_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["p_mfgr"], "MFGR#1") |
                              eq(arg["p_mfgr"], "MFGR#2"),
                          expressions::Selectivity{2.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"]};
                    });

            auto rel51023_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["c_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"], arg["c_nation"]};
                    });

            auto rel51028_d =
                build.at(3)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["s_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"]};
                    });

            return probe.unpack()
                .join(
                    rel51028_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_suppkey"];
                    })
                .join(
                    rel51023_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_custkey"];
                    })
                .join(
                    rel51018_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .join(
                    rel51013_d,
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
                      return {GpuAggrMatExpr{
                          (arg["lo_revenue"] - arg["lo_supplycost"])
                              .as("tmp", "profit"),
                          1, 0, SUM}};
                    },
                    10, 1024 * 1024)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["c_nation"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
          },
          10, 128 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["c_nation"], arg["profit"]};
          },
          {direction::ASC, direction::ASC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}

PreparedStatement ssb::Query::prepare41_pushdown(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel51013 = morph.scan("date", {"d_datekey", "d_year"});
  auto rel51018 = morph.scan("part", {"p_partkey", "p_mfgr"});
  auto rel51023 = morph.scan("customer", {"c_custkey", "c_nation", "c_region"});
  auto rel51028 = morph.scan("supplier", {"s_suppkey", "s_region"});
  auto rel =
      morph.scan("lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                               "lo_orderdate", "lo_revenue", "lo_supplycost"});

  return morph
      .parallel(
          rel.router(64, RoutingPolicy::LOCAL, DeviceType::CPU)
              .unpack()
              .join(
                  rel51028.router(64, RoutingPolicy::LOCAL, DeviceType::CPU)
                      .unpack()
                      .filter([&](const auto &arg) -> expression_t {
                        return expressions::hint(
                            eq(arg["s_region"], "AMERICA"),
                            expressions::Selectivity{1.0 / 5});
                      })
                      .project(
                          [&](const auto &arg) -> std::vector<expression_t> {
                            return {arg["s_suppkey"]};
                          }),
                  [&](const auto &build_arg) -> expression_t {
                    return build_arg["s_suppkey"];
                  },
                  [&](const auto &probe_arg) -> expression_t {
                    return probe_arg["lo_suppkey"];
                  })
              .project([&](const auto &arg) -> std::vector<expression_t> {
                return {arg["lo_custkey"].as("tmp", "lo_custkey"),
                        arg["lo_partkey"].as("tmp", "lo_partkey"),
                        arg["lo_orderdate"].as("tmp", "lo_orderdate"),
                        (arg["lo_revenue"] - arg["lo_supplycost"])
                            .as("tmp", "profit")};
              })
              .pack()
              .router(DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM,
                      DeviceType::CPU),
          {rel51013, rel51018, rel51023},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel51013_d = build.at(0).unpack();

            auto rel51018_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["p_mfgr"], "MFGR#1") |
                              eq(arg["p_mfgr"], "MFGR#2"),
                          expressions::Selectivity{2.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["p_partkey"]};
                    });

            auto rel51023_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["c_region"], "AMERICA"),
                          expressions::Selectivity{1.0 / 5});
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"], arg["c_nation"]};
                    });
            return probe.unpack()
                .join(
                    rel51023_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_custkey"];
                    })
                .join(
                    rel51018_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_partkey"];
                    })
                .join(
                    rel51013_d,
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
                      return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
                    },
                    10, 1024 * 1024)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["c_nation"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["profit"], 1, 0, SUM}};
          },
          10, 128 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_year"], arg["c_nation"], arg["profit"]};
          },
          {direction::ASC, direction::ASC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
