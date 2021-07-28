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
#include "q19.hpp"

static int q_instance = 30;

PreparedStatement Q_19_cpar(DegreeOfParallelism dop, const aff_t &aff_parallel,
                            const aff_t &aff_reduce, DeviceType dev,
                            const scan_t &scan) {
  assert(dev == DeviceType::CPU);

  auto rel1618 =
      scan(
          "tpcc_item",
          {"i_id",
           "i_price"})  // (table=[[SSB, ch100w_item]], fields=[[0, 3]],
                        // traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              dop, 128, dev,
              aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homBrdcst.hetSingle.none])
          .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homBrdcst.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return ge(arg["i_price"], ((double)1)) &
                   le(arg["i_price"], ((double)400000));
          })  // (condition=[AND($1, $2)],
              // trait=[Pelago.[].X86_64.unpckd.homBrdcst.hetSingle.cX86_64],
              // isS=[false])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["i_id"]};
          });
  return scan(
             "tpcc_orderline",
             {"ol_w_id", "ol_i_id", "ol_quantity",
              "ol_amount"})  // (table=[[SSB, ch100w_orderline]], fields=[[2, 4,
                             // 7, 8]],
                             // traits=[Pelago.[].X86_64.packed.homSingle.hetSingle.none])
      .router(
          dop, 16, RoutingPolicy::LOCAL, dev,
          aff_parallel())  // (trait=[Pelago.[].X86_64.packed.homRandom.hetSingle.none])
      .unpack()  // (trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return (ge(arg["ol_quantity"], 1) & le(arg["ol_quantity"], 10) &
                (eq(arg["$0"], 1) | eq(arg["$0"], 2) | eq(arg["$0"], 3) |
                 eq(arg["$0"], 4) | eq(arg["$0"], 5)));
      })  // (condition=[AND($2, $3, OR($4, $5,
          // $6))],
          // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
          // isS=[false])
      .join(
          rel1618,
          [&](const auto &build_arg) -> expression_t {
            return build_arg["$0"].as("PelagoJoin#1623", "bk_0");
          },
          [&](const auto &probe_arg) -> expression_t {
            return probe_arg["ol_i_id"].as("PelagoJoin#1623", "pk_0");
          },
          20,
          128 * 1024)  // (condition=[=($3, $0)],
                       // joinType=[inner],
                       // rowcnt=[2.56E7],
                       // maxrow=[100000.0],
                       // maxEst=[100000.0], h_bits=[27],
                       // build=[RecordType(INTEGER i_id,
                       // BOOLEAN >=, BOOLEAN <=)],
                       // lcount=[1.3944357272154548E9],
                       // rcount=[3.84E9],
                       // buildcountrow=[2.56E7],
                       // probecountrow=[3.84E9])
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["ol_amount"]};
          },
          {SUM})  // (group=[{}], revenue=[SUM($0)],
                  // trait=[Pelago.[].X86_64.unpckd.homRandom.hetSingle.cX86_64],
                  // global=[false])
      .router(
          DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM, DeviceType::CPU,
          aff_reduce())  // (trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64])
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoAggregate#1627", "revenue")};
          },
          {SUM})  // (group=[{}], revenue=[SUM($0)],
                  // trait=[Pelago.[].X86_64.unpckd.homSingle.hetSingle.cX86_64],
                  // global=[true])
      .print(pg{"pm-csv"})
      .prepare();
}
