/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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

  auto rel = morph.scan("lineorder", {"lo_orderdate", "lo_quantity",
                                      "lo_extendedprice", "lo_discount"});

  return morph
      .parallel(
          rel, {},
          [](RelBuilder probe, const std::vector<RelBuilder> &) {
            return probe.unpack()
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      gt(arg["lo_orderdate"], 19930000) &
                          lt(arg["lo_orderdate"], 19940000) &
                          ge(arg["lo_discount"], 1) &
                          le(arg["lo_discount"], 3) &
                          lt(arg["lo_quantity"], 25),
                      expressions::Selectivity((1.0 / 7) *
                                               (0.5 * 3.0 / 11)));
                })
                .reduce(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {(arg["lo_extendedprice"] * arg["lo_discount"])
                                  .as("tmp", "revenue")};
                    },
                    {SUM});
          })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["revenue"]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
