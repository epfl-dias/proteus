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

#include <cstdint>
#include <tpch/query.hpp>
#include <vector>

// TPC-H Q12 -- Shipping Modes and Order Priority.
//
//   select l_shipmode,
//     sum(case when o_orderpriority='1-URGENT' or o_orderpriority='2-HIGH'
//              then 1 else 0 end) as high_line_count,
//     sum(case when o_orderpriority<>'1-URGENT' and o_orderpriority<>'2-HIGH'
//              then 1 else 0 end) as low_line_count
//   from orders, lineitem
//   where o_orderkey = l_orderkey
//     and l_shipmode in ('MAIL','SHIP')
//     and l_commitdate < l_receiptdate
//     and l_shipdate   < l_commitdate
//     and l_receiptdate >= date '1994-01-01'
//     and l_receiptdate <  date '1994-01-01' + interval '1' year
//   group by l_shipmode
//   order by l_shipmode;
//
// Flat, non-nested: join + group-by + CASE + order-by (no subquery). Strings
// (l_shipmode, o_orderpriority) are dstring: eq(col, "literal") resolves to the
// dict code, IN(...) is an OR of dict equalities, CASE is `cond(pred, 1, 0)`.
// Dates are int YYYYMMDD, so the interval bound becomes 19950101.

constexpr auto query = "tpch100_Q12_simplified";

PreparedStatement tpch::Query::prepare12(proteus::QueryShaper& morph) {
  morph.setQueryName(query);

  auto rel_orders = morph.scan("orders", {"o_orderkey", "o_orderpriority"});
  auto rel_lineitem =
      morph.scan("lineitem", {"l_orderkey", "l_shipmode", "l_commitdate",
                              "l_receiptdate", "l_shipdate"});

  return morph
      .parallel(
          rel_lineitem, {rel_orders},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto orders_d = build.at(0).unpack().project(
                [&](const auto& arg) -> std::vector<expression_t> {
                  return {arg["o_orderkey"], arg["o_orderpriority"]};
                });

            return probe.unpack()
                .filter([&](const auto& arg) -> expression_t {
                  auto mode = eq(arg["l_shipmode"], "MAIL") |
                              eq(arg["l_shipmode"], "SHIP");
                  return mode & lt(arg["l_commitdate"], arg["l_receiptdate"]) &
                         lt(arg["l_shipdate"], arg["l_commitdate"]) &
                         ge(arg["l_receiptdate"], 19940101) &
                         lt(arg["l_receiptdate"], 19950101);
                })
                .join(
                    orders_d,
                    [&](const auto& build_arg) -> expression_t {
                      return build_arg["o_orderkey"];
                    },
                    [&](const auto& probe_arg) -> expression_t {
                      return probe_arg["l_orderkey"];
                    })
                // partial (per-partition) group-by on the GPU
                .groupby(
                    [&](const auto& arg) -> std::vector<expression_t> {
                      return {arg["l_shipmode"].as("tmp", "l_shipmode")};
                    },
                    [&](const auto& arg) -> std::vector<GpuAggrMatExpr> {
                      auto is_high = eq(arg["o_orderpriority"], "1-URGENT") |
                                     eq(arg["o_orderpriority"], "2-HIGH");
                      return {GpuAggrMatExpr{cond(is_high, expression_t{1},
                                                  expression_t{0})
                                                .as("tmp", "high_line_count"),
                                             1, 0, SUM},
                              GpuAggrMatExpr{cond(is_high, expression_t{0},
                                                  expression_t{1})
                                                .as("tmp", "low_line_count"),
                                             2, 0, SUM}};
                    },
                    10, 16)
                .pack();
          })
      .unpack()
      // final merge of the per-partition partial aggregates
      .groupby(
          [&](const auto& arg) -> std::vector<expression_t> {
            return {arg["l_shipmode"]};
          },
          [&](const auto& arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["high_line_count"], 1, 0, SUM},
                    GpuAggrMatExpr{arg["low_line_count"], 2, 0, SUM}};
          },
          10, 16)
      .sort(
          [&](const auto& arg) -> std::vector<expression_t> {
            return {arg["l_shipmode"], arg["high_line_count"],
                    arg["low_line_count"]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
