/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <tpch/query.hpp>

// TPC-H Q6 -- Forecasting Revenue Change.
//   select sum(l_extendedprice * l_discount) as revenue
//   from   lineitem
//   where  l_shipdate >= date '1994-01-01'
//     and  l_shipdate <  date '1994-01-01' + interval '1' year
//     and  l_discount between 0.06 - 0.01 and 0.06 + 0.01
//     and  l_quantity < 24;
//
// Pure scan + filter + scalar aggregate over the lineitem fact table; no join.
// This is the cleanest query for isolating decompression placement effects
// (NCU bandwidth/footprint), analogous to why SSB measurements use Q1.*.
// Scans 4 RFOR-compressible integer columns.
//
// SIMPLIFICATIONS vs. canonical TPC-H Q6:
//   * NONE on the expression side -- revenue = sum(l_extendedprice*l_discount)
//     is exactly the TPC-H Q6 measure. No clause/predicate dropped.
//   * Only deviation is the data encoding (see query.hpp): l_shipdate is an
//     int YYYYMMDD so the date range becomes [19940101, 19950101); l_discount
//     is integer percentage points so "between 0.05 and 0.07" becomes [5, 7].
//     l_quantity / l_extendedprice are scaled ints. Result fits int32.

constexpr auto query = "tpch100_Q6";

PreparedStatement tpch::Query::prepare06(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel = morph.scan(
      "lineitem", {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"});

  return morph
      .parallel(
          rel, {},
          [](RelBuilder probe, const std::vector<RelBuilder> &) {
            return probe.unpack()
                .filter([&](const auto &arg) -> expression_t {
                  // dates as YYYYMMDD int; discount as integer percentage points
                  return expressions::hint(
                      ge(arg["l_shipdate"], 19940101) &
                          lt(arg["l_shipdate"], 19950101) &
                          ge(arg["l_discount"], 5) & le(arg["l_discount"], 7) &
                          lt(arg["l_quantity"], 24),
                      // ~ (1/7 years) * (3/11 discount values) * (~0.46 qty)
                      expressions::Selectivity(0.02));
                })
                .reduce(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {(arg["l_extendedprice"] * arg["l_discount"])
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
