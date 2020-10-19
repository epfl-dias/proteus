/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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
#include "q01.hpp"

#include "ch-queries.hpp"

static int q_instance = 0;

PreparedStatement Q_1_cpar(DegreeOfParallelism dop, const aff_t &aff_parallel,
                           const aff_t &aff_reduce, DeviceType dev,
                           const scan_t &scan) {
  std::string count_order = "count_order";

  return scan(
             tpcc_orderline, {"ol_number",
                              "ol_delivery_d", "ol_quantity", "ol_amount"})  // (table=[[SSB, ch100w_orderline]], fields=[[3, 6, 7, 8]], traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
      .router(
          dop, 32, RoutingPolicy::LOCAL, dev,
          aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homRandom.hetSingle.none])
      .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg["$1"], expressions::DateConstant("2007-01-02 00:00:00"));
      })  // (condition=[>($1, 2007-01-02 00:00:00:TIMESTAMP(3))],
          // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
          // isS=[false])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#1052", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#1052", "$1"),
                                   1, 0, SUM},
                    GpuAggrMatExpr{
                        (expression_t{1}).as("PelagoAggregate#1052", "$2"), 2,
                        0, SUM},
                    GpuAggrMatExpr{(arg["$3"]).as("PelagoAggregate#1052", "$3"),
                                   3, 0, SUM},
                    GpuAggrMatExpr{
                        (expression_t{1}).as("PelagoAggregate#1052", "$4"), 4,
                        0, SUM},
                    GpuAggrMatExpr{
                        (expression_t{1}).as("PelagoAggregate#1052", "$5"), 5,
                        0, SUM}};
          },
          5,
          128)  // (group=[{0}], sum_qty=[$SUM0($2)], agg#1=[COUNT($2)],
                // sum_amount=[$SUM0($3)], agg#3=[COUNT($3)],
                // count_order=[COUNT()],
                // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
                // global=[false])
      .router(
          DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM, DeviceType::CPU,
          aff_reduce())  // (trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#1054", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#1054", "$1"),
                                   1, 0, SUM},
                    GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#1054", "$2"),
                                   2, 0, SUM},
                    GpuAggrMatExpr{(arg["$3"]).as("PelagoAggregate#1054", "$3"),
                                   3, 0, SUM},
                    GpuAggrMatExpr{(arg["$4"]).as("PelagoAggregate#1054", "$4"),
                                   4, 0, SUM},
                    GpuAggrMatExpr{(arg["$5"]).as("PelagoAggregate#1054", "$5"),
                                   5, 0, SUM}};
          },
          5,
          128)  // (group=[{0}], sum_qty=[$SUM0($1)], agg#1=[$SUM0($2)],
                // sum_amount=[$SUM0($3)], agg#3=[$SUM0($4)],
                // count_order=[$SUM0($5)],
                // trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64],
                // global=[true])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$0"]).as("PelagoProject#1055", "ol_number"),
                (cond(eq(arg["$2"], 0), 0, arg["$1"]))
                    .as("PelagoProject#1055", "sum_qty"),
                (cond(eq(arg["$4"], 0), 0.0, arg["$3"]))
                    .as("PelagoProject#1055", "sum_amount"),
                (cond(eq(arg["$2"], 0), 0, arg["$1"]).template as<FloatType>() /
                 arg["$2"].template as<FloatType>())
                    .as("PelagoProject#1055", "avg_qty"),
                (cond(eq(arg["$4"], 0), 0.0, arg["$3"]) /
                 arg["$4"].template as<FloatType>())
                    .as("PelagoProject#1055", "avg_amount"),
                (arg["$5"]).as("PelagoProject#1055", "count_order")};
      })  // (ol_number=[$0], sum_qty=[CASE(=($2, 0), null:INTEGER, $1)],
          // sum_amount=[CASE(=($4, 0), null:DOUBLE, $3)],
          // avg_qty=[CAST(/(CASE(=($2, 0), null:INTEGER, $1), $2)):INTEGER],
          // avg_amount=[/(CASE(=($4, 0), null:DOUBLE, $3), $4)],
          // count_order=[$5],
          // trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"], arg["$2"],
                    arg["$3"], arg["$4"], arg["$5"]};
          },
          {
              direction::ASC,
              direction::NONE,
              direction::NONE,
              direction::NONE,
              direction::NONE,
              direction::NONE,
          })  // (sort0=[$0], dir0=[ASC],
              // trait=[Pelago.[0].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        auto outrel = "tmp";
        return {arg["$0"].as(outrel, "ol_number"),
                arg["$1"].as(outrel, "sum_qty"),
                arg["$2"].as(outrel, "sum_amount"),
                arg["$3"].as(outrel, "avg_qty"),
                arg["$4"].as(outrel, "avg_amount"),
                arg["$5"].as(outrel, "count_order")};
      })  // (trait=[ENUMERABLE.[0].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .print(pg{"pm-csv"})
      .prepare();
}
