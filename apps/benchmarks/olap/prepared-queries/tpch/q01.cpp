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

// TPC-H Q1 -- Pricing Summary Report.
//   select l_returnflag, l_linestatus,
//          sum(l_quantity)                               as sum_qty,
//          sum(l_extendedprice)                          as sum_base_price,
//          sum(l_extendedprice*(1-l_discount))           as sum_disc_price,
//          sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
//          avg(l_quantity)                               as avg_qty,
//          avg(l_extendedprice)                          as avg_price,
//          avg(l_discount)                               as avg_disc,
//          count(*)                                      as count_order
//   from   lineitem
//   where  l_shipdate <= date '1998-12-01' - interval '90' day
//   group by l_returnflag, l_linestatus
//   order by l_returnflag, l_linestatus;
//
// Group-by-bound query, all-numeric scan of the lineitem fact table, no join.
// Scans 7 RFOR-compressible columns (l_shipdate, l_returnflag, l_linestatus,
// l_quantity, l_extendedprice, l_discount, l_tax) -- which sits right at the
// ~8-column shared-memory decode budget, so this is the query that exercises
// the tile-size vs. occupancy trade-off.
//
// This is the FULL Q1: all eight aggregates (incl. the three avg() outputs)
// and the complete (1-discount)*(1+tax) charge formula are kept. Nothing in
// the query shape is dropped.
//
// SIMPLIFICATIONS vs. canonical TPC-H Q1: only the data encoding differs (see
// query.hpp). All columns are integers:
//   * l_shipdate is YYYYMMDD, so the predicate becomes <= 19980902.
//   * l_discount / l_tax are integer percentage points, so the rate factors
//     (1 - l_discount) and (1 + l_tax) are expressed as (100 - l_discount) and
//     (100 + l_tax). sum_disc_price is therefore scaled by 100 and sum_charge
//     by 100*100 relative to the spec; the per-row charge product can wrap in
//     int32 -- accepted, since we measure GPU execution cost, not numeric
//     result fidelity.
//   * avg() outputs are integer divisions sum/count_order (computed in a final
//     projection), so they truncate -- again acceptable for an int32 result.

constexpr auto query = "tpch100_Q1";

PreparedStatement tpch::Query::prepare01(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel = morph.scan("lineitem", {"l_shipdate", "l_returnflag",
                                     "l_linestatus", "l_quantity",
                                     "l_extendedprice", "l_discount", "l_tax"});

  return morph
      .parallel(
          rel, {},
          [](RelBuilder probe, const std::vector<RelBuilder> &) {
            return probe.unpack()
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      le(arg["l_shipdate"], 19980902),
                      expressions::Selectivity(0.98));
                })
                // partial (per-partition) aggregation on the GPU
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["l_returnflag"].as("tmp", "l_returnflag"),
                              arg["l_linestatus"].as("tmp", "l_linestatus")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      // (100 - l_discount) == (1 - discount) in percent points
                      auto disc = expression_t{100} - arg["l_discount"];
                      // (100 + l_tax) == (1 + tax) in percent points
                      auto tax = expression_t{100} + arg["l_tax"];
                      return {
                          GpuAggrMatExpr{arg["l_quantity"].as("tmp", "sum_qty"),
                                         1, 0, SUM},
                          GpuAggrMatExpr{
                              arg["l_extendedprice"].as("tmp", "sum_base_price"),
                              2, 0, SUM},
                          GpuAggrMatExpr{(arg["l_extendedprice"] * disc)
                                             .as("tmp", "sum_disc_price"),
                                         3, 0, SUM},
                          GpuAggrMatExpr{(arg["l_extendedprice"] * disc * tax)
                                             .as("tmp", "sum_charge"),
                                         4, 0, SUM},
                          GpuAggrMatExpr{arg["l_discount"].as("tmp", "sum_disc"),
                                         5, 0, SUM},
                          GpuAggrMatExpr{
                              expression_t{1}.as("tmp", "count_order"), 6, 0,
                              SUM}};
                    },
                    // only ~4 distinct (returnflag,linestatus) groups
                    10, 16)
                .pack();
          })
      .unpack()
      // final merge of the per-partition partial aggregates
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["l_returnflag"], arg["l_linestatus"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["sum_qty"], 1, 0, SUM},
                    GpuAggrMatExpr{arg["sum_base_price"], 2, 0, SUM},
                    GpuAggrMatExpr{arg["sum_disc_price"], 3, 0, SUM},
                    GpuAggrMatExpr{arg["sum_charge"], 4, 0, SUM},
                    GpuAggrMatExpr{arg["sum_disc"], 5, 0, SUM},
                    GpuAggrMatExpr{arg["count_order"], 6, 0, SUM}};
          },
          10, 16)
      // derive the avg() outputs as sum / count_order
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {arg["l_returnflag"], arg["l_linestatus"],
                arg["sum_qty"], arg["sum_base_price"],
                arg["sum_disc_price"], arg["sum_charge"],
                (arg["sum_qty"] / arg["count_order"]).as("tmp", "avg_qty"),
                (arg["sum_base_price"] / arg["count_order"])
                    .as("tmp", "avg_price"),
                (arg["sum_disc"] / arg["count_order"]).as("tmp", "avg_disc"),
                arg["count_order"]};
      })
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["l_returnflag"], arg["l_linestatus"],
                    arg["sum_qty"],      arg["sum_base_price"],
                    arg["sum_disc_price"], arg["sum_charge"],
                    arg["avg_qty"],      arg["avg_price"],
                    arg["avg_disc"],     arg["count_order"]};
          },
          {direction::ASC, direction::ASC, direction::NONE, direction::NONE,
           direction::NONE, direction::NONE, direction::NONE, direction::NONE,
           direction::NONE, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
