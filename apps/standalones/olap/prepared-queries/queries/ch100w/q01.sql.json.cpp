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

// AUTOGENERATED FILE. DO NOT EDIT.

constexpr auto query = "ch100w_Q01";

#include "query.cpp.inc"

PreparedStatement Query::prepare01(bool memmv) {
  auto rel =
      getBuilder<Tplugin>()
          .scan<Tplugin>(
              "tpcc_orderline",
              {"ol_number", "ol_delivery_d", "ol_quantity", "ol_amount"},
              getCatalog())  // (table=[[SSB, ch100w_orderline]], fields=[[3, 6,
                             // 7, 8]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .router(
              dop, 16, RoutingPolicy::LOCAL, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
      ;

  if (memmv) rel = rel.memmove(8, dev == DeviceType::CPU);

  rel =
      rel.to_gpu()   // (trait=[Pelago.[].packed.NVPTX.homRandom.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return gt(arg["ol_delivery_d"],
                      expressions::DateConstant(1167696000000));
          })  // (condition=[>($1, 2007-01-02 00:00:00:TIMESTAMP(3))],
              // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle],
              // isS=[false])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["ol_number"].as("PelagoAggregate#1404", "$0")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{
                        (arg["ol_quantity"]).as("PelagoAggregate#1404", "$1"),
                        1, 0, SUM},
                    GpuAggrMatExpr{
                        (expression_t{1}).as("PelagoAggregate#1404", "$2"), 2,
                        0, SUM},
                    GpuAggrMatExpr{
                        (arg["ol_amount"]).as("PelagoAggregate#1404", "$3"), 3,
                        0, SUM}};
              },
              4,
              131072 * 4)  // (group=[{0}], sum_qty=[$SUM0($1)], agg#1=[COUNT
                           // ($1)],
                           // sum_amount=[$SUM0($2)], agg#3=[COUNT($2)],
                           // count_order=[COUNT()],
          // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .to_cpu()  // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
          .router(
              DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU,
              aff_reduce())  // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"].as("PelagoAggregate#1410", "$0")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#1410", "$1"),
                                   1, 0, SUM},
                    GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#1410", "$2"),
                                   2, 0, SUM},
                    GpuAggrMatExpr{(arg["$3"]).as("PelagoAggregate#1410", "$3"),
                                   3, 0, SUM}};
              },
              4,
              32)  // (group=[{0}], sum_qty=[$SUM0($1)], agg#1=[$SUM0($2)],
                   // sum_amount=[$SUM0($3)], agg#3=[$SUM0($4)],
                   // count_order=[$SUM0($5)],
                   // trait=[Pelago.[].unpckd.NVPTX.homSingle.hetSingle])
                   //          .sort(
          //              [&](const auto &arg) -> std::vector<expression_t> {
          //                return {arg["$0"], arg["$1"], arg["$2"],
          //                        arg["$3"]};
          //              },
          //              {direction::ASC, direction::NONE, direction::NONE,
          //               direction::NONE})
          .print(
              [&](const auto &arg,
                  std::string outrel) -> std::vector<expression_t> {
                return {arg["$0"].as(outrel, "ol_number"),
                        arg["$1"].as(outrel, "sum_qty"),
                        arg["$3"].as(outrel, "sum_amount"),
                        (arg["$1"].template as<FloatType>() /
                         arg["$2"].template as<FloatType>())
                            .as(outrel, "avg_qty"),
                        (arg["$3"].template as<FloatType>() /
                         arg["$2"].template as<FloatType>())
                            .as(outrel, "avg_amount"),
                        arg["$2"].as(outrel, "count_order")};
              },
              std::string{query} +
                  (memmv
                       ? "mv"
                       : "nmv"))  // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle])
      ;

  return rel.prepare();
}
