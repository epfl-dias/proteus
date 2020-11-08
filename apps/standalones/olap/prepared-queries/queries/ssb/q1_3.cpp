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

constexpr auto query = "ssb100_Q1_3";

PreparedStatement ssb::Query::prepare13(proteus::QueryShaper &morph) {
  auto rel6631 =
      morph
          .distribute_build(
              morph.scan("date", {"d_datekey", "d_year", "d_weeknuminyear"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(
                eq(arg["d_weeknuminyear"], 6) & eq(arg["d_year"], 1994),
                expressions::Selectivity{1.0 / 364});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_datekey"])};
          });

  auto rel = morph.distribute_probe(morph.scan(
      "lineorder",
      {"lo_orderdate", "lo_quantity", "lo_extendedprice", "lo_discount"}));

  rel = rel.unpack()
            .join(
                rel6631,
                [&](const auto &build_arg) -> expression_t {
                  return build_arg["d_datekey"];
                },
                [&](const auto &probe_arg) -> expression_t {
                  return probe_arg["lo_orderdate"];
                },
                5, 8)
            .filter([&](const auto &arg) -> expression_t {
              return expressions::hint(
                  ge(arg["lo_discount"], 5) & le(arg["lo_discount"], 7) &
                      ge(arg["lo_quantity"], 26) & le(arg["lo_quantity"], 35),
                  expressions::Selectivity{0.1 * 3 / 11});
            })
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {(arg["lo_extendedprice"] * arg["lo_discount"])
                              .as("tmp", "revenue")};
                },
                {SUM});
  rel = morph.collect_unpacked(rel)
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["revenue"]};
                },
                {SUM})
            .print(pg{"pm-csv"});
  return rel.prepare();
}
