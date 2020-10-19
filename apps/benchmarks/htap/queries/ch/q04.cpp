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
#include "q04.hpp"

static int q_instance = 10;

PreparedStatement Q_4_cpar(DegreeOfParallelism dop, const aff_t &aff_parallel,
                           const aff_t &aff_reduce, DeviceType dev,
                           const scan_t &scan) {
  assert(dev == DeviceType::CPU);
  auto rel2027 =
      scan(
          "tpcc_order",
          {"o_id", "o_d_id", "o_w_id",
           "o_entry_d"})  // (table=[[SSB, ch100w_order]], fields=[[0, 1, 2,
                          // 4]],
                          // traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homBrdcst.hetSingle.none])
          .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homBrdcst.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (
                ge(arg["o_entry_d"], expressions::DateConstant(915148800000)) &
                lt(arg["o_entry_d"], expressions::DateConstant(1644162414000)));
          })  // (condition=[AND(>=($3, 2007-01-02
              // 00:00:00:TIMESTAMP(3)), <($3, 2012-01-02
              // 00:00:00:TIMESTAMP(3)))],
              // trait=[Pelago.[].X86_64.unpckd.homBrdcst.hetSingle.cX86_64],
              // isS=[false])
      ;
  auto rel2036 =
      scan(
          "tpcc_orderline", {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"})  // (table=[[SSB, ch100w_orderline]], fields=[[0, 1, 2, 6]], traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
          .router(
              dop, 16, RoutingPolicy::LOCAL, dev,
              aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homRandom.hetSingle.none])
          .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64])
          .join(
              rel2027,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["o_id"].as("PelagoJoin#2030", "bk_0"),
                    build_arg["o_w_id"].as("PelagoJoin#2030", "bk_2"),
                    build_arg["o_d_id"].as("PelagoJoin#2030", "bk_1")}
                    .as("PelagoJoin#2030", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["ol_o_id"].as("PelagoJoin#2030", "pk_0"),
                    probe_arg["ol_w_id"].as("PelagoJoin#2030", "pk_2"),
                    probe_arg["ol_d_id"].as("PelagoJoin#2030", "pk_1")}
                    .as("PelagoJoin#2030", "pk");
              },
              20,
              80000000)  // (condition=[AND(=($0, $4), =($2, $6), =($1, $5))],
                         // joinType=[inner], rowcnt=[7.68E8],
                         // maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
                         // build=[RecordType(BIGINT o_id, INTEGER o_d_id,
                         // INTEGER o_w_id, TIMESTAMP(0) o_entry_d)],
                         // lcount=[6.7109666771656204E10], rcount=[1.536E10],
                         // buildcountrow=[7.68E8], probecountrow=[1.536E10])
          .filter([&](const auto &arg) -> expression_t {
            return ge(arg["ol_delivery_d"], arg["o_entry_d"]);
          })  // (condition=[>=($7, $3)],
              // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
              // isS=[false])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_id"], arg["o_d_id"], arg["o_w_id"],
                        arg["o_entry_d"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {};
              },
              20,
              800000000)  // (group=[{0, 1, 2, 3}],
                          // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
                          // global=[false])
          .router(
              DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU,
              aff_reduce())  // (trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"].as("PelagoAggregate#2035", "$0"),
                        arg["$1"].as("PelagoAggregate#2035", "$1"),
                        arg["$2"].as("PelagoAggregate#2035", "$2"),
                        arg["$3"].as("PelagoAggregate#2035", "$3")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {};
              },
              20,
              80000000)  // (group=[{0, 1, 2, 3}],
                         // trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64],
                         // global=[true])
          .pack()
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].X86_64.unpckd.homBrdcst.hetSingle.cX86_64])
          .unpack();
  return scan("tpcc_order", {"o_id",
                             "o_d_id", "o_w_id", "o_entry_d", "o_ol_cnt"})  // (table=[[SSB, ch100w_order]], fields=[[0, 1, 2, 4, 6]], traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
      .router(
          dop, 16, RoutingPolicy::LOCAL, dev,
          aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homRandom.hetSingle.none])
      .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return (ge(arg["o_entry_d"], expressions::DateConstant(1167696000000)) &
                lt(arg["o_entry_d"], expressions::DateConstant(1325462400000)));
      })  // (condition=[AND(>=($3, 2007-01-02 00:00:00:TIMESTAMP(3)), <($3,
          // 2012-01-02 00:00:00:TIMESTAMP(3)))],
          // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
          // isS=[false])
      .join(
          rel2036,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$0"].as("PelagoJoin#2040", "bk_0"),
                build_arg["$1"].as("PelagoJoin#2040", "bk_1"),
                build_arg["$2"].as("PelagoJoin#2040", "bk_2"),
                build_arg["$3"].as("PelagoJoin#2040", "bk_3")}
                .as("PelagoJoin#2040", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$0"].as("PelagoJoin#2040", "pk_0"),
                probe_arg["$1"].as("PelagoJoin#2040", "pk_1"),
                probe_arg["$2"].as("PelagoJoin#2040", "pk_2"),
                probe_arg["$3"].as("PelagoJoin#2040", "pk_3")}
                .as("PelagoJoin#2040", "pk");
          },
          10, 600121200)  // (condition=[AND(=($4, $0), =($5, $1), =($6, $2),
                          // =($7, $3))], joinType=[inner], rowcnt=[3.84E8],
                          // maxrow=[9.0E13], maxEst=[6.7108864E7], h_bits=[28],
                          // build=[RecordType(BIGINT o_id, INTEGER o_d_id,
                          // INTEGER o_w_id, TIMESTAMP(0) o_entry_d)],
                          // lcount=[3.2490159316488026E10], rcount=[3.84E8],
                          // buildcountrow=[3.84E8], probecountrow=[3.84E8])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["o_ol_cnt"].as("PelagoAggregate#2043", "o_ol_cnt")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (expression_t{1}).as("PelagoAggregate#2043", "order_count"), 1,
                0, SUM}};
          },
          10,
          600121200)  // (group=[{0}], order_count=[COUNT()],
                      // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
                      // global=[false])
      .router(
          DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM, DeviceType::CPU,
          aff_reduce())  // (trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["o_ol_cnt"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["order_count"], 1, 0, SUM}};
          },
          10,
          600121200)  // (group=[{0}], order_count=[$SUM0($1)],
                      // trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64],
                      // global=[true])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["o_ol_cnt"], arg["order_count"]};
          },
          {
              direction::NONE,
              direction::ASC,
          })  // (sort0=[$0], dir0=[ASC],
              // trait=[Pelago.[0].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .print(pg{"pm-csv"})
      .prepare();
}
