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

constexpr auto query = "ssb100_Q1_1";

PreparedStatement ssb::Query::prepare11(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel2337 = morph.scan("date", {"d_datekey", "d_year"});

  auto rel = morph.scan("lineorder", {"lo_orderdate", "lo_quantity",
                                      "lo_extendedprice", "lo_discount"});

  return morph
      .parallel(
          rel, {rel2337},
          [](RelBuilder probe, std::vector<RelBuilder> build) {
            auto rel2337_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["d_year"], 1993),
                          expressions::Selectivity(1.0 / 7));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {(arg["d_datekey"])};
                    });

            return probe.unpack()
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      ge(arg["lo_discount"], 1) & le(arg["lo_discount"], 3) &
                          lt(arg["lo_quantity"], 25),
                      expressions::Selectivity(0.5 * 3.0 / 11));
                })
                .join(
                    rel2337_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["d_datekey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["lo_orderdate"];
                    })
                .reduce(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {(arg["lo_extendedprice"] * arg["lo_discount"])
                                  .as("tmp", "revenue")};
                    },
                    {SUM});
            //                .pack();
          })
      //      .unpack()
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["revenue"]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
