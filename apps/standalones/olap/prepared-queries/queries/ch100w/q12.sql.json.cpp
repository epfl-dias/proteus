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

constexpr auto query = "ch100w_Q12";

#include "query.cpp.inc"

PreparedStatement Query::prepare12(bool memmv) {
  auto rel2508 =
      RelBuilder{getContext<Tplugin>()}
          .scan<Tplugin>(
              "inputs/ch100w/order.csv",
              {"o_id", "o_d_id", "o_w_id", "o_entry_d", "o_carrier_id",
               "o_ol_cnt"},
              getCatalog())  // (table=[[SSB, ch100w_order]], fields=[[0, 1, 2,
                             // 4, 5, 6]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle])
      ;

  if (memmv) rel2508 = rel2508.memmove(8, dev == DeviceType::CPU);

  rel2508 =
      rel2508
          .to_gpu()  // (trait=[Pelago.[].packed.NVPTX.homBrdcst.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#2508", "o_id"),
                    (arg["$1"]).as("PelagoProject#2508", "o_d_id"),
                    (arg["$2"]).as("PelagoProject#2508", "o_w_id"),
                    (arg["$3"]).as("PelagoProject#2508", "o_entry_d"),
                    (arg["$5"]).as("PelagoProject#2508", "o_carrier_id"),
                    (cond((eq(arg["$4"], 1) | eq(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#2508", "or"),
                    (cond((ne(arg["$4"], 1) & ne(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#2508", "nor")};
          })  // (o_id=[$0], o_d_id=[$1], o_w_id=[$2], o_entry_d=[$3],
              // o_ol_cnt=[$5], CASE=[CASE(OR(=($4, 1), =($4, 2)), 1, 0)],
              // CASE6=[CASE(AND(<>($4, 1), <>($4, 2)), 1, 0)],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
      ;
  auto rel =
      RelBuilder{getContext<Tplugin>()}
          .scan<Tplugin>(
              "inputs/ch100w/orderline.csv",
              {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"},
              getCatalog())  // (table=[[SSB, ch100w_orderline]], fields=[[0, 1,
                             // 2, 6]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .router(
              dop, 8, RoutingPolicy::LOCAL, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
      ;

  if (memmv) rel = rel.memmove(8, dev == DeviceType::CPU);

  rel =
      rel.to_gpu()   // (trait=[Pelago.[].packed.NVPTX.homRandom.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return lt(arg["ol_delivery_d"],
                      expressions::DateConstant(1577836800000));
          })  // (condition=[<($3, 2020-01-01 00:00:00:TIMESTAMP(3))],
              // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle],
              // isS=[false])
          .join(
              rel2508,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["o_w_id"].as("PelagoJoin#2513", "bk_2"),
                    build_arg["o_d_id"].as("PelagoJoin#2513", "bk_1"),
                    build_arg["o_id"].as("PelagoJoin#2513", "bk_0")}
                    .as("PelagoJoin#2513", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["ol_w_id"].as("PelagoJoin#2513", "pk_2"),
                    probe_arg["ol_d_id"].as("PelagoJoin#2513", "pk_1"),
                    probe_arg["ol_o_id"].as("PelagoJoin#2513", "pk_0")}
                    .as("PelagoJoin#2513", "pk");
              },
              28,
              3000000)  // (condition=[AND(=($9, $2), =($8, $1), =($7, $0))],
                        // joinType=[inner], rowcnt=[3.072E9],
                        // maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
                        // build=[RecordType(BIGINT o_id, INTEGER o_d_id,
                        // INTEGER o_w_id, TIMESTAMP(0) o_entry_d, INTEGER
                        // o_ol_cnt, INTEGER CASE, INTEGER CASE6)],
                        // lcount=[5.116125192468789E11], rcount=[7.68E9],
                        // buildcountrow=[3.072E9], probecountrow=[7.68E9])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$7"]).as("PelagoProject#2514", "$0"),
                    (arg["$8"]).as("PelagoProject#2514", "$1"),
                    (arg["$9"]).as("PelagoProject#2514", "$2"),
                    (arg["$10"]).as("PelagoProject#2514", "$3"),
                    (arg["$0"]).as("PelagoProject#2514", "$4"),
                    (arg["$1"]).as("PelagoProject#2514", "$5"),
                    (arg["$2"]).as("PelagoProject#2514", "$6"),
                    (arg["$3"]).as("PelagoProject#2514", "$7"),
                    (arg["$4"]).as("PelagoProject#2514", "$8"),
                    (arg["$5"]).as("PelagoProject#2514", "$9"),
                    (arg["$6"]).as("PelagoProject#2514", "$10")};
          })  // (ol_o_id=[$7], ol_d_id=[$8], ol_w_id=[$9], ol_delivery_d=[$10],
              // o_id=[$0], o_d_id=[$1], o_w_id=[$2], o_entry_d=[$3],
              // o_ol_cnt=[$4], CASE=[$5], CASE6=[$6],
              // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return le(arg["o_entry_d"], arg["ol_delivery_d"]);
          })  // (condition=[<=($7, $3)],
              // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle],
              // isS=[false])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_ol_cnt"].as("PelagoAggregate#2517", "$0")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{cond((eq(arg["o_carrier_id"], 1) |
                                             eq(arg["o_carrier_id"], 2)),
                                            1, 0)
                                           .as("PelagoAggregate#2517", "$1"),
                                       1, 0, SUM},
                        GpuAggrMatExpr{cond((ne(arg["o_carrier_id"], 1) &
                                             ne(arg["o_carrier_id"], 2)),
                                            1, 0)
                                           .as("PelagoAggregate#2517", "$2"),
                                       2, 0, SUM}};
              },
              10,
              131072)  // (group=[{0}], high_line_count=[SUM($1)],
                       // low_line_count=[SUM($2)],
                       // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .to_cpu()    // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
          .router(
              DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU,
              aff_reduce())  // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"].as("PelagoAggregate#2523", "$0")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#2523", "$1"),
                                   1, 0, SUM},
                    GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#2523", "$2"),
                                   2, 0, SUM}};
              },
              10,
              131072)  // (group=[{0}], high_line_count=[SUM($1)],
                       // low_line_count=[SUM($2)],
                       // trait=[Pelago.[].unpckd.NVPTX.homSingle.hetSingle])
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"], arg["$1"], arg["$2"]};
              },
              {direction::ASC, direction::NONE,
               direction::
                   NONE})  // (sort0=[$0], dir0=[ASC],
                           // trait=[Pelago.[0].unpckd.X86_64.homSingle.hetSingle])
          .print(
              [&](const auto &arg,
                  std::string outrel) -> std::vector<expression_t> {
                return {arg["$0"].as(outrel, "o_ol_cnt"),
                        arg["$1"].as(outrel, "high_line_count"),
                        arg["$2"].as(outrel, "low_line_count")};
              },
              std::string{query} +
                  (memmv
                       ? "mv"
                       : "nmv"))  // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle])
      ;

  return rel.prepare();
}
