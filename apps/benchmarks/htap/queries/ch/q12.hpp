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

#ifndef HARMONIA_QUERIES_CH_Q12_HPP_
#define HARMONIA_QUERIES_CH_Q12_HPP_

#include "../queries.hpp"

template <typename Tplugin>
PreparedStatement q_ch12_c1t() {
  std::string revenue = "revenue";
  RelBuilderFactory ctx{__FUNCTION__};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel22964 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d",
                          "o_carrier_id", "o_ol_cnt"},
                         catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4, 5, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#22964", "$0"),
                    (arg["$1"]).as("PelagoProject#22964", "$1"),
                    (arg["$2"]).as("PelagoProject#22964", "$2"),
                    (arg["$3"]).as("PelagoProject#22964", "$3"),
                    (arg["$5"]).as("PelagoProject#22964", "$4"),
                    (cond((eq(arg["$4"], 1) | eq(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#22964", "$5"),
                    (cond((ne(arg["$4"], 1) & ne(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#22964", "$6")};
          })
      // (o_id=[$0], o_d_id=[$1], o_w_id=[$2], o_entry_d=[$3], o_ol_cnt=[$5],
      // CASE=[CASE(OR(=($4, 1), =($4, 2)), 1, 0)], CASE6=[CASE(AND(<>($4, 1),
      // <>($4, 2)), 1, 0)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  return ctx.getBuilder()
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 6]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return lt(arg["$3"], expressions::DateConstant(1577836800000));
      })
      // (condition=[<($3, 2020-01-01 00:00:00:TIMESTAMP(3))],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .join(
          rel22964,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$2"].as("PelagoJoin#22967", "bk_2"),
                build_arg["$1"].as("PelagoJoin#22967", "bk_1"),
                build_arg["$0"].as("PelagoJoin#22967", "bk_0")}
                .as("PelagoJoin#22967", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$2"].as("PelagoJoin#22967", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#22967", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#22967", "pk_0")}
                .as("PelagoJoin#22967", "pk");
          },
          10, 1024 * 1024)
      // (condition=[AND(=($9, $2), =($8, $1), =($7, $0))], joinType=[inner],
      // rowcnt=[3.072E9], maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
      // build=[RecordType(BIGINT o_id, INTEGER o_d_id, INTEGER o_w_id,
      // TIMESTAMP(0) o_entry_d, INTEGER o_ol_cnt, INTEGER CASE, INTEGER
      // CASE6)], lcount=[5.116125192468789E11], rcount=[1.536E10],
      // buildcountrow=[3.072E9], probecountrow=[1.536E10])
      .filter([&](const auto &arg) -> expression_t {
        return le(arg["$3"], arg["$10"]);
      })
      // (condition=[<=($3, $10)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$4"]).as("PelagoProject#22969", "$0"),
                (arg["$5"]).as("PelagoProject#22969", "$1"),
                (arg["$6"]).as("PelagoProject#22969", "$2")};
      })
      // (o_ol_cnt=[$4], $f1=[$5], $f2=[$6],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#22970", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#22970", "$1"), 1,
                               0, SUM},
                GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#22970", "$2"), 2,
                               0, SUM}};
          },
          10, 1024 * 1024)
      // (group=[{0}], high_line_count=[SUM($1)], low_line_count=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"], arg["$2"]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      // (sort0=[$0], dir0=[ASC],
      // trait=[Pelago.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "o_ol_cnt"),
                arg["$1"].as(outrel, "high_line_count"),
                arg["$2"].as(outrel, "low_line_count")};
      })
      // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<12>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                              Tr aff_reduce, DeviceType dev) {
  assert(dev == DeviceType::CPU);
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  std::string revenue = "revenue";
  auto rel24766 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d",
                          "o_carrier_id", "o_ol_cnt"},
                         catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4, 5, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, DeviceType::CPU,
              aff_parallel())
          // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#24766", "$0"),
                    (arg["$1"]).as("PelagoProject#24766", "$1"),
                    (arg["$2"]).as("PelagoProject#24766", "$2"),
                    (arg["$3"]).as("PelagoProject#24766", "$3"),
                    (arg["$5"]).as("PelagoProject#24766", "$4"),
                    (cond((eq(arg["$4"], 1) | eq(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#24766", "$5"),
                    (cond((ne(arg["$4"], 1) & ne(arg["$4"], 2)), 1, 0))
                        .as("PelagoProject#24766", "$6")};
          })
      // (o_id=[$0], o_d_id=[$1], o_w_id=[$2], o_entry_d=[$3], o_ol_cnt=[$5],
      // CASE=[CASE(OR(=($4, 1), =($4, 2)), 1, 0)], CASE6=[CASE(AND(<>($4, 1),
      // <>($4, 2)), 1, 0)],
      // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
      ;
  return ctx.getBuilder()
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 6]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU, aff_parallel())
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return lt(arg["$3"], expressions::DateConstant(1577836800000));
      })
      // (condition=[<($3, 2020-01-01 00:00:00:TIMESTAMP(3))],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // isS=[false])
      .join(
          rel24766,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$2"].as("PelagoJoin#24770", "bk_2"),
                build_arg["$1"].as("PelagoJoin#24770", "bk_1"),
                build_arg["$0"].as("PelagoJoin#24770", "bk_0")}
                .as("PelagoJoin#24770", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$2"].as("PelagoJoin#24770", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#24770", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#24770", "pk_0")}
                .as("PelagoJoin#24770", "pk");
          },
          24, 256 * 1024 * 1024)
      // (condition=[AND(=($9, $2), =($8, $1), =($7, $0))], joinType=[inner],
      // rowcnt=[3.072E9], maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
      // build=[RecordType(BIGINT o_id, INTEGER o_d_id, INTEGER o_w_id,
      // TIMESTAMP(0) o_entry_d, INTEGER o_ol_cnt, INTEGER CASE, INTEGER
      // CASE6)], lcount=[5.116125192468789E11], rcount=[7.68E9],
      // buildcountrow=[3.072E9], probecountrow=[7.68E9])
      .filter([&](const auto &arg) -> expression_t {
        return le(arg["$3"], arg["$10"]);
      })
      // (condition=[<=($3, $10)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$4"]).as("PelagoProject#24772", "$0"),
                (arg["$5"]).as("PelagoProject#24772", "$1"),
                (arg["$6"]).as("PelagoProject#24772", "$2")};
      })
      // (o_ol_cnt=[$4], $f1=[$5], $f2=[$6],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#24773", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#24773", "$1"), 1,
                               0, SUM},
                GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#24773", "$2"), 2,
                               0, SUM}};
          },
          10, 128)
      // (group=[{0}], high_line_count=[SUM($1)], low_line_count=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .pack()
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.cX86_64],
      // intrait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // inputRows=[3.84E8], cost=[{440.84040000000005 rows, 440.40000000000003
      // cpu, 0.0 io}])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#24777", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#24777", "$1"), 1,
                               0, SUM},
                GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#24777", "$2"), 2,
                               0, SUM}};
          },
          10, 128)
      // (group=[{0}], high_line_count=[SUM($1)], low_line_count=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"], arg["$2"]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      // (sort0=[$0], dir0=[ASC],
      // trait=[Pelago.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "o_ol_cnt"),
                arg["$1"].as(outrel, "high_line_count"),
                arg["$2"].as(outrel, "low_line_count")};
      })
      // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

#endif /* HARMONIA_QUERIES_CH_Q12_HPP_ */
