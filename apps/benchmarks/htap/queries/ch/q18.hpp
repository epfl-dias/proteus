/*
    Harmonia -- High-performance elastic HTAP on heterogeneous hardware.

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

#ifndef HARMONIA_QUERIES_CH_Q18_HPP_
#define HARMONIA_QUERIES_CH_Q18_HPP_

#include "../queries.hpp"

template <>
template <typename Tplugin>
PreparedStatement Q<18>::c1t() {
  std::string revenue = "revenue";
  auto ctx = new ParallelContext("ch_Q" + std::to_string(Qid), false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel55895 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_customer",
                         {"c_id", "c_d_id", "c_w_id", "c_last"}, catalog)
          // (table=[[SSB, tpcc_customer]], fields=[[0, 1, 2, 5]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  auto rel55898 =
      RelBuilder{ctx}
          .scan<Tplugin>(
              "tpcc_order",
              {"o_id", "o_d_id", "o_w_id", "o_c_id", "o_entry_d", "o_ol_cnt"},
              catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 3, 4, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .join(
              rel55895,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["$0"].as("PelagoJoin#55897", "bk_0"),
                    build_arg["$2"].as("PelagoJoin#55897", "bk_2"),
                    build_arg["$1"].as("PelagoJoin#55897", "bk_1")}
                    .as("PelagoJoin#55897", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["$3"].as("PelagoJoin#55897", "pk_3"),
                    probe_arg["$2"].as("PelagoJoin#55897", "pk_2"),
                    probe_arg["$1"].as("PelagoJoin#55897", "pk_1")}
                    .as("PelagoJoin#55897", "pk");
              },
              10, 1024 * 1024)
          // (condition=[AND(=($0, $7), =($2, $6), =($1, $5))],
          // joinType=[inner], rowcnt=[3.072E9], maxrow=[3000000.0],
          // maxEst=[3000000.0], h_bits=[28], build=[RecordType(INTEGER c_id,
          // INTEGER c_d_id, INTEGER c_w_id, VARCHAR c_last)],
          // lcount=[2.85473452196066E11], rcount=[3.072E9],
          // buildcountrow=[3.072E9], probecountrow=[3.072E9])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#55898", "$0"),
                    (arg["$3"]).as("PelagoProject#55898", "$1"),
                    (arg["$4"]).as("PelagoProject#55898", "$2"),
                    (arg["$5"]).as("PelagoProject#55898", "$3"),
                    (arg["$6"]).as("PelagoProject#55898", "$4"),
                    (arg["$8"]).as("PelagoProject#55898", "$5"),
                    (arg["$9"]).as("PelagoProject#55898", "$6")};
          })
      // (c_id=[$0], c_last=[$3], o_id=[$4], o_d_id=[$5], o_w_id=[$6],
      // o_entry_d=[$8], o_ol_cnt=[$9],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_amount"}, catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 8]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .join(
          rel55898,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$4"].as("PelagoJoin#55900", "bk_4"),
                build_arg["$3"].as("PelagoJoin#55900", "bk_3"),
                build_arg["$2"].as("PelagoJoin#55900", "bk_2")}
                .as("PelagoJoin#55900", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$2"].as("PelagoJoin#55900", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#55900", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#55900", "pk_0")}
                .as("PelagoJoin#55900", "pk");
          },
          10, 1024 * 1024)
      // (condition=[AND(=($9, $4), =($8, $3), =($7, $2))], joinType=[inner],
      // rowcnt=[3.072E9], maxrow=[9.0E12], maxEst=[6.7108864E7], h_bits=[28],
      // build=[RecordType(INTEGER c_id, VARCHAR c_last, BIGINT o_id, INTEGER
      // o_d_id, INTEGER o_w_id, TIMESTAMP(0) o_entry_d, INTEGER o_ol_cnt)],
      // lcount=[5.116125192468789E11], rcount=[3.072E10],
      // buildcountrow=[3.072E9], probecountrow=[3.072E10])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$2"]).as("PelagoProject#55901", "$0"),
                (arg["$4"]).as("PelagoProject#55901", "$1"),
                (arg["$3"]).as("PelagoProject#55901", "$2"),
                (arg["$0"]).as("PelagoProject#55901", "$3"),
                (arg["$1"]).as("PelagoProject#55901", "$4"),
                (arg["$5"]).as("PelagoProject#55901", "$5"),
                (arg["$6"]).as("PelagoProject#55901", "$6"),
                (arg["$10"]).as("PelagoProject#55901", "$7")};
      })
      // (o_id=[$2], o_w_id=[$4], o_d_id=[$3], $f3=[$0], c_last=[$1],
      // o_entry_d=[$5], o_ol_cnt=[$6], ol_amount=[$10],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#55902", "$0"),
                    arg["$1"].as("PelagoAggregate#55902", "$1"),
                    arg["$2"].as("PelagoAggregate#55902", "$2"),
                    arg["$3"].as("PelagoAggregate#55902", "$3"),
                    arg["$4"].as("PelagoAggregate#55902", "$4"),
                    arg["$5"].as("PelagoAggregate#55902", "$5"),
                    arg["$6"].as("PelagoAggregate#55902", "$6")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (arg["$7"]).as("PelagoAggregate#55902", "$7"), 1, 0, SUM}};
          },
          10, 1024 * 1024)
      // (group=[{0, 1, 2, 3, 4, 5, 6}], EXPR$4=[SUM($7)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg["$7"], ((double)200));
      })
      // (condition=[>($7, 200)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$3"]).as("PelagoProject#55904", "$0"),
                (arg["$4"]).as("PelagoProject#55904", "$1"),
                (arg["$5"]).as("PelagoProject#55904", "$2"),
                (arg["$6"]).as("PelagoProject#55904", "$3"),
                (arg["$7"]).as("PelagoProject#55904", "$4")};
      })
      // ($f3=[$3], c_last=[$4], o_entry_d=[$5], o_ol_cnt=[$6], EXPR$4=[$7],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$1"]).as("PelagoProject#55905", "$0"),
                (arg["$0"]).as("PelagoProject#55905", "$1"),
                (arg["$2"]).as("PelagoProject#55905", "$2"),
                (arg["$3"]).as("PelagoProject#55905", "$3"),
                (arg["$4"]).as("PelagoProject#55905", "$4")};
      })
      // (c_last=[$1], o_id=[$0], o_entry_d=[$2], o_ol_cnt=[$3], EXPR$4=[$4],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"], arg["$2"], arg["$3"], arg["$4"]};
          },
          {direction::DESC, direction::ASC, direction::NONE, direction::NONE,
           direction::NONE})
      // (sort0=[$4], sort1=[$2], dir0=[DESC], dir1=[ASC], trait=[Pelago.[4
      // DESC, 2].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "c_last"), arg["$1"].as(outrel, "o_id"),
                arg["$2"].as(outrel, "o_entry_d"),
                arg["$3"].as(outrel, "o_ol_cnt"),
                arg["$4"].as(outrel, "EXPR$4")};
      })
      // (trait=[ENUMERABLE.[4 DESC,
      // 2].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<18>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                              Tr aff_reduce) {
  auto ctx = new ParallelContext(
      "ch_Q" + std::to_string(Qid) + "_" + typeid(Tplugin).name(), false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel56251 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_customer",
                         {"c_id", "c_d_id", "c_w_id", "c_last"}, catalog)
          // (table=[[SSB, tpcc_customer]], fields=[[0, 1, 2, 5]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  auto rel56257 =
      RelBuilder{ctx}
          .scan<Tplugin>(
              "tpcc_order",
              {"o_id", "o_d_id", "o_w_id", "o_c_id", "o_entry_d", "o_ol_cnt"},
              catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 3, 4, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .join(
              rel56251,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["$0"].as("PelagoJoin#56253", "bk_0"),
                    build_arg["$2"].as("PelagoJoin#56253", "bk_2"),
                    build_arg["$1"].as("PelagoJoin#56253", "bk_1")}
                    .as("PelagoJoin#56253", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["$3"].as("PelagoJoin#56253", "pk_3"),
                    probe_arg["$2"].as("PelagoJoin#56253", "pk_2"),
                    probe_arg["$1"].as("PelagoJoin#56253", "pk_1")}
                    .as("PelagoJoin#56253", "pk");
              },
              24, 16 * 1024 * 1024)
          // (condition=[AND(=($0, $7), =($2, $6), =($1, $5))],
          // joinType=[inner], rowcnt=[3.072E9], maxrow=[3000000.0],
          // maxEst=[3000000.0], h_bits=[28], build=[RecordType(INTEGER c_id,
          // INTEGER c_d_id, INTEGER c_w_id, VARCHAR c_last)],
          // lcount=[2.85473452196066E11], rcount=[3.072E9],
          // buildcountrow=[3.072E9], probecountrow=[3.072E9])
          .pack()
          // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64],
          // intrait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
          // inputRows=[3.072E9], cost=[{11731.720000000001 rows, 11720.0 cpu,
          // 0.0 io}])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, DeviceType::CPU,
              aff_parallel())
          // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle.cX86_64])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#56257", "$0"),
                    (arg["$3"]).as("PelagoProject#56257", "$1"),
                    (arg["$4"]).as("PelagoProject#56257", "$2"),
                    (arg["$5"]).as("PelagoProject#56257", "$3"),
                    (arg["$6"]).as("PelagoProject#56257", "$4"),
                    (arg["$8"]).as("PelagoProject#56257", "$5"),
                    (arg["$9"]).as("PelagoProject#56257", "$6")};
          })
      // (c_id=[$0], c_last=[$3], o_id=[$4], o_d_id=[$5], o_w_id=[$6],
      // o_entry_d=[$8], o_ol_cnt=[$9],
      // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
      ;
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_amount"}, catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 8]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU, aff_parallel())
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel56257,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$4"].as("PelagoJoin#56260", "bk_4"),
                build_arg["$3"].as("PelagoJoin#56260", "bk_3"),
                build_arg["$2"].as("PelagoJoin#56260", "bk_2")}
                .as("PelagoJoin#56260", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$2"].as("PelagoJoin#56260", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#56260", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#56260", "pk_0")}
                .as("PelagoJoin#56260", "pk");
          },
          28, 1024 * 1024 * 1024)
      // (condition=[AND(=($9, $4), =($8, $3), =($7, $2))], joinType=[inner],
      // rowcnt=[3.07232768E9], maxrow=[9.0E12], maxEst=[6.7108864E7],
      // h_bits=[28], build=[RecordType(INTEGER c_id, VARCHAR c_last, BIGINT
      // o_id, INTEGER o_d_id, INTEGER o_w_id, TIMESTAMP(0) o_entry_d, INTEGER
      // o_ol_cnt)], lcount=[5.116693851312614E11], rcount=[1.536E10],
      // buildcountrow=[3.07232768E9], probecountrow=[1.536E10])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$2"]).as("PelagoProject#56261", "$0"),
                (arg["$4"]).as("PelagoProject#56261", "$1"),
                (arg["$3"]).as("PelagoProject#56261", "$2"),
                (arg["$0"]).as("PelagoProject#56261", "$3"),
                (arg["$1"]).as("PelagoProject#56261", "$4"),
                (arg["$5"]).as("PelagoProject#56261", "$5"),
                (arg["$6"]).as("PelagoProject#56261", "$6"),
                (arg["$10"]).as("PelagoProject#56261", "$7")};
      })
      // (o_id=[$2], o_w_id=[$4], o_d_id=[$3], $f3=[$0], c_last=[$1],
      // o_entry_d=[$5], o_ol_cnt=[$6], ol_amount=[$10],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#56262", "$0"),
                    arg["$1"].as("PelagoAggregate#56262", "$1"),
                    arg["$2"].as("PelagoAggregate#56262", "$2"),
                    arg["$3"].as("PelagoAggregate#56262", "$3"),
                    arg["$4"].as("PelagoAggregate#56262", "$4"),
                    arg["$5"].as("PelagoAggregate#56262", "$5"),
                    arg["$6"].as("PelagoAggregate#56262", "$6")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (arg["$7"]).as("PelagoAggregate#56262", "$7"), 1, 0, SUM}};
          },
          20, 256 * 1024 * 1024)
      // (group=[{0, 1, 2, 3, 4, 5, 6}], EXPR$4=[SUM($7)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .pack()
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.cX86_64],
      // intrait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // inputRows=[1.536E9], cost=[{4692.688 rows, 4688.0 cpu, 0.0 io}])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#56266", "$0"),
                    arg["$1"].as("PelagoAggregate#56266", "$1"),
                    arg["$2"].as("PelagoAggregate#56266", "$2"),
                    arg["$3"].as("PelagoAggregate#56266", "$3"),
                    arg["$4"].as("PelagoAggregate#56266", "$4"),
                    arg["$5"].as("PelagoAggregate#56266", "$5"),
                    arg["$6"].as("PelagoAggregate#56266", "$6")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (arg["$7"]).as("PelagoAggregate#56266", "$7"), 1, 0, SUM}};
          },
          20, 1024 * 1024 * 1024)
      // (group=[{0, 1, 2, 3, 4, 5, 6}], EXPR$4=[SUM($7)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg["$7"], ((double)200));
      })
      // (condition=[>($7, 200)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$3"]).as("PelagoProject#56268", "$0"),
                (arg["$4"]).as("PelagoProject#56268", "$1"),
                (arg["$5"]).as("PelagoProject#56268", "$2"),
                (arg["$6"]).as("PelagoProject#56268", "$3"),
                (arg["$7"]).as("PelagoProject#56268", "$4")};
      })
      // ($f3=[$3], c_last=[$4], o_entry_d=[$5], o_ol_cnt=[$6], EXPR$4=[$7],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$1"]).as("PelagoProject#56269", "$0"),
                (arg["$0"]).as("PelagoProject#56269", "$1"),
                (arg["$2"]).as("PelagoProject#56269", "$2"),
                (arg["$3"]).as("PelagoProject#56269", "$3"),
                (arg["$4"]).as("PelagoProject#56269", "$4")};
      })
      // (c_last=[$1], o_id=[$0], o_entry_d=[$2], o_ol_cnt=[$3], EXPR$4=[$4],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"], arg["$2"], arg["$3"], arg["$4"]};
          },
          {direction::DESC, direction::ASC, direction::NONE, direction::NONE,
           direction::NONE})
      // (sort0=[$4], sort1=[$2], dir0=[DESC], dir1=[ASC], trait=[Pelago.[4
      // DESC, 2].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "c_last"), arg["$1"].as(outrel, "o_id"),
                arg["$2"].as(outrel, "o_entry_d"),
                arg["$3"].as(outrel, "o_ol_cnt"),
                arg["$4"].as(outrel, "EXPR$4")};
      })
      // (trait=[ENUMERABLE.[4 DESC,
      // 2].unpckd.X86_64.homSingle.hetSingle.cX86_64])

      .prepare();
}

#endif /* HARMONIA_QUERIES_CH_Q18_HPP_ */
