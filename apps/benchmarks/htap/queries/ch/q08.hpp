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

#ifndef HARMONIA_QUERIES_CH_Q8_HPP_
#define HARMONIA_QUERIES_CH_Q8_HPP_

#include "../queries.hpp"

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<8>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                             Tr aff_reduce, DeviceType dev) {
  assert(dev == DeviceType::CPU);
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel13131 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_nation", {"n_nationkey", "n_name"}, catalog)
          // (table=[[SSB, tpcc_nation]], fields=[[0, 1]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#13131", "$0"),
                    (eq(arg["$1"], expression_t{"'Germany':VARCHAR", nullptr}))
                        .as("PelagoProject#13131", "$1")};
          })
      // (n_nationkey=[$0], ==[=($1, 'Germany')],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  auto rel13137 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_supplier", {"su_suppkey", "su_nationkey"},
                         catalog)
          // (table=[[SSB, tpcc_supplier]], fields=[[0, 3]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .join(
              rel13131,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["$0"].as("PelagoJoin#13133", "bk_0");
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["$1"].as("PelagoJoin#13133", "pk_1");
              },
              20, 256 * 1024 * 1024)
          // (condition=[=($3, $0)], joinType=[inner], rowcnt=[1048576.0],
          // maxrow=[62.0], maxEst=[62.0], h_bits=[22],
          // build=[RecordType(INTEGER n_nationkey, BOOLEAN =)],
          // lcount=[3.052633491611866E7], rcount=[1.024E7],
          // buildcountrow=[1048576.0], probecountrow=[1.024E7])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$2"]).as("PelagoProject#13134", "$0"),
                    (arg["$1"]).as("PelagoProject#13134", "$1")};
          })
          // (su_suppkey=[$2], ==[$1],
          // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .pack()
          // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64],
          // intrait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
          // inputRows=[1.024E7], cost=[{8.008000000000001 rows, 8.0 cpu, 0.0
          // io}])
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
      ;
  auto rel13139 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_item", {"i_id"}, catalog)
          // (table=[[SSB, tpcc_item]], fields=[[0]],
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
      ;
  auto rel13141 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_customer", {"c_id", "c_d_id", "c_w_id"}, catalog)
          // (table=[[SSB, tpcc_customer]], fields=[[0, 1, 2]],
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
      ;
  auto rel13144 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_stock", {"s_i_id", "s_w_id"}, catalog)
          // (table=[[SSB, tpcc_stock]], fields=[[0, 1]],
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
            return {(arg["$0"]).as("PelagoProject#13144", "$0"),
                    (arg["$1"]).as("PelagoProject#13144", "$1"),
                    (((arg["$1"] * arg["$0"]) % 10000))
                        .as("PelagoProject#13144", "$2")};
          })
      // (s_i_id=[$0], s_w_id=[$1], MOD=[MOD(*($1, $0), 10000)],
      // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
      ;
  auto rel13148 =
      ctx.getBuilder()
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_c_id", "o_entry_d"},
                         catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 3, 4]],
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
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$4"], expressions::DateConstant(1167696000000)) &
                    le(arg["$4"], expressions::DateConstant(1325462400000)));
          })
          // (condition=[AND(>=($4, 2007-01-02 00:00:00:TIMESTAMP(3)), <=($4,
          // 2012-01-02 00:00:00:TIMESTAMP(3)))],
          // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64],
          // isS=[false])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#13148", "$0"),
                    (arg["$1"]).as("PelagoProject#13148", "$1"),
                    (arg["$2"]).as("PelagoProject#13148", "$2"),
                    (arg["$3"]).as("PelagoProject#13148", "$3"),
                    (expressions::ExtractExpression{
                         arg["$4"], expressions::extract_unit::YEAR})
                        .as("PelagoProject#13148", "$4")};
          })
      // (o_id=[$0], o_d_id=[$1], o_w_id=[$2], o_c_id=[$3],
      // EXTRACT=[EXTRACT(FLAG(YEAR), $4)],
      // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
      ;
  return ctx.getBuilder()
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_i_id",
                      "ol_supply_w_id", "ol_amount"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 4, 5, 8]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU, aff_parallel())
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .filter(
          [&](const auto &arg) -> expression_t { return lt(arg["$3"], 1000); })
      // (condition=[<($3, 1000)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // isS=[false])
      .join(
          rel13148,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$2"].as("PelagoJoin#13152", "bk_2"),
                build_arg["$1"].as("PelagoJoin#13152", "bk_1"),
                build_arg["$0"].as("PelagoJoin#13152", "bk_0")}
                .as("PelagoJoin#13152", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$2"].as("PelagoJoin#13152", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#13152", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#13152", "pk_0")}
                .as("PelagoJoin#13152", "pk");
          },
          20, 256 * 1024 * 1024)
      // (condition=[AND(=($7, $2), =($6, $1), =($5, $0))], joinType=[inner],
      // rowcnt=[7.68E8], maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
      // build=[RecordType(BIGINT o_id, INTEGER o_d_id, INTEGER o_w_id, INTEGER
      // o_c_id, BIGINT EXTRACT)], lcount=[8.474395470161682E10],
      // rcount=[7.68E9], buildcountrow=[7.68E8], probecountrow=[7.68E9])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$8"]).as("PelagoProject#13153", "$0"),
                (arg["$9"]).as("PelagoProject#13153", "$1"),
                (arg["$10"]).as("PelagoProject#13153", "$2"),
                (arg["$1"]).as("PelagoProject#13153", "$3"),
                (arg["$2"]).as("PelagoProject#13153", "$4"),
                (arg["$3"]).as("PelagoProject#13153", "$5"),
                (arg["$4"]).as("PelagoProject#13153", "$6")};
      })
      // (ol_i_id=[$8], ol_supply_w_id=[$9], ol_amount=[$10], o_d_id=[$1],
      // o_w_id=[$2], o_c_id=[$3], EXTRACT=[$4],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel13144,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$0"].as("PelagoJoin#13154", "bk_0"),
                build_arg["$1"].as("PelagoJoin#13154", "bk_1")}
                .as("PelagoJoin#13154", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$0"].as("PelagoJoin#13154", "pk_0"),
                probe_arg["$1"].as("PelagoJoin#13154", "pk_1")}
                .as("PelagoJoin#13154", "pk");
          },
          20, 256 * 1024 * 1024)
      // (condition=[AND(=($3, $0), =($4, $1))], joinType=[inner],
      // rowcnt=[1.024E10], maxrow=[1.0E7], maxEst=[1.0E7], h_bits=[28],
      // build=[RecordType(INTEGER s_i_id, INTEGER s_w_id, INTEGER MOD)],
      // lcount=[7.418320817733391E11], rcount=[1.92E9],
      // buildcountrow=[1.024E10], probecountrow=[1.92E9])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$0"]).as("PelagoProject#13155", "$0"),
                (arg["$3"]).as("PelagoProject#13155", "$1"),
                (arg["$5"]).as("PelagoProject#13155", "$2"),
                (arg["$6"]).as("PelagoProject#13155", "$3"),
                (arg["$7"]).as("PelagoProject#13155", "$4"),
                (arg["$8"]).as("PelagoProject#13155", "$5"),
                (arg["$2"]).as("PelagoProject#13155", "$6"),
                (arg["$9"]).as("PelagoProject#13155", "$7")};
      })
      // (s_i_id=[$0], ol_i_id=[$3], ol_amount=[$5], o_d_id=[$6], o_w_id=[$7],
      // o_c_id=[$8], MOD=[$2], EXTRACT=[$9],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel13141,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$0"].as("PelagoJoin#13156", "bk_0"),
                build_arg["$2"].as("PelagoJoin#13156", "bk_2"),
                build_arg["$1"].as("PelagoJoin#13156", "bk_1")}
                .as("PelagoJoin#13156", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$5"].as("PelagoJoin#13156", "pk_5"),
                probe_arg["$4"].as("PelagoJoin#13156", "pk_4"),
                probe_arg["$3"].as("PelagoJoin#13156", "pk_3")}
                .as("PelagoJoin#13156", "pk");
          },
          20, 256 * 1024 * 1024)
      // (condition=[AND(=($0, $8), =($2, $7), =($1, $6))], joinType=[inner],
      // rowcnt=[3.072E9], maxrow=[3000000.0], maxEst=[3000000.0], h_bits=[28],
      // build=[RecordType(INTEGER c_id, INTEGER c_d_id, INTEGER c_w_id)],
      // lcount=[2.114538111673339E11], rcount=[1.28E9],
      // buildcountrow=[3.072E9], probecountrow=[1.28E9])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$3"]).as("PelagoProject#13157", "$0"),
                (arg["$4"]).as("PelagoProject#13157", "$1"),
                (arg["$5"]).as("PelagoProject#13157", "$2"),
                (arg["$9"]).as("PelagoProject#13157", "$3"),
                (arg["$10"]).as("PelagoProject#13157", "$4")};
      })
      // (s_i_id=[$3], ol_i_id=[$4], ol_amount=[$5], MOD=[$9], EXTRACT=[$10],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel13139,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$0"].as("PelagoJoin#13158", "bk_0"),
                build_arg["$0"].as("PelagoJoin#13158", "bk_0")}
                .as("PelagoJoin#13158", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$1"].as("PelagoJoin#13158", "pk_1"),
                probe_arg["$0"].as("PelagoJoin#13158", "pk_0")}
                .as("PelagoJoin#13158", "pk");
          },
          20, 256 * 1024 * 1024)
      // (condition=[AND(=($0, $2), =($0, $1))], joinType=[inner],
      // rowcnt=[1.024E8], maxrow=[100000.0], maxEst=[100000.0], h_bits=[28],
      // build=[RecordType(INTEGER i_id)], lcount=[1.8887062805063353E9],
      // rcount=[3.84E8], buildcountrow=[1.024E8], probecountrow=[3.84E8])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$3"]).as("PelagoProject#13159", "$0"),
                (arg["$4"]).as("PelagoProject#13159", "$1"),
                (arg["$5"]).as("PelagoProject#13159", "$2")};
      })
      // (ol_amount=[$3], MOD=[$4], EXTRACT=[$5],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel13137,
          [&](const auto &build_arg) -> expression_t {
            return build_arg["$0"].as("PelagoJoin#13160", "bk_0");
          },
          [&](const auto &probe_arg) -> expression_t {
            return probe_arg["$1"].as("PelagoJoin#13160", "pk_1");
          },
          20, 256 * 1024 * 1024)
      // (condition=[=($3, $0)], joinType=[inner], rowcnt=[1.048576E7],
      // maxrow=[620544.0], maxEst=[620544.0], h_bits=[26],
      // build=[RecordType(INTEGER su_suppkey, BOOLEAN =)],
      // lcount=[3.535520584906131E8], rcount=[3.84E8],
      // buildcountrow=[1.048576E7], probecountrow=[3.84E8])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$4"]).as("PelagoProject#13161", "$0"),
                (cond(arg["$1"], arg["$2"], ((double)0)))
                    .as("PelagoProject#13161", "$1"),
                (arg["$2"]).as("PelagoProject#13161", "$2")};
      })
      // (l_year=[$4], $f1=[CASE($1, $2, 0:DOUBLE)], ol_amount=[$2],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#13162", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#13162", "$1"), 1,
                               0, SUM},
                GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#13162", "$2"), 2,
                               0, SUM}};
          },
          20, 256 * 1024 * 1024)
      // (group=[{0}], agg#0=[SUM($1)], agg#1=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .pack()
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.cX86_64],
      // intrait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // inputRows=[3.84E7], cost=[{44.4444 rows, 44.400000000000006 cpu, 0.0
      // io}])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#13166", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{(arg["$1"]).as("PelagoAggregate#13166", "$1"), 1,
                               0, SUM},
                GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#13166", "$2"), 2,
                               0, SUM}};
          },
          20, 256 * 1024 * 1024)
      // (group=[{0}], agg#0=[SUM($1)], agg#1=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$0"]).as("PelagoProject#13167", "$0"),
                ((arg["$1"] / arg["$2"])).as("PelagoProject#13167", "$1")};
      })
      // (l_year=[$0], mkt_share=[/($1, $2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"], arg["$1"]};
          },
          {direction::ASC, direction::NONE})
      // (sort0=[$0], dir0=[ASC],
      // trait=[Pelago.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "l_year"),
                arg["$1"].as(outrel, "mkt_share")};
      })
      // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

#endif /* HARMONIA_QUERIES_CH_Q8_HPP_ */
