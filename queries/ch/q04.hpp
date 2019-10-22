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

#ifndef HARMONIA_QUERIES_CH_Q4_HPP_
#define HARMONIA_QUERIES_CH_Q4_HPP_

#include <gflags/gflags.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "adaptors/aeolus-plugin.hpp"
#include "benchmarks/tpcc_64.hpp"
#include "benchmarks/ycsb.hpp"
#include "codegen/communication/comm-manager.hpp"
#include "codegen/memory/block-manager.hpp"
#include "codegen/memory/memory-manager.hpp"
#include "codegen/operators/relbuilder.hpp"
#include "codegen/plan/plan-parser.hpp"
#include "codegen/plan/prepared-statement.hpp"
#include "codegen/plugins/binary-block-plugin.hpp"
#include "codegen/storage/storage-manager.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/util/jit/pipeline.hpp"
#include "codegen/util/parallel-context.hpp"
#include "codegen/util/profiling.hpp"
#include "codegen/util/timing.hpp"
#include "interfaces/bench.hpp"
#include "llvm/Support/DynamicLibrary.h"
#include "queries/queries.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

template <typename Tplugin>
PreparedStatement q_ch4_c1t() {
  std::string revenue = "revenue";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel13550 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d"}, catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$3"], expressions::DateConstant(1325462400000)) &
                    lt(arg["$3"], expressions::DateConstant(1609545600000)));
          })
      // (condition=[AND(>=($3, 2007-01-02 00:00:00:TIMESTAMP(3)), <($3,
      // 2012-01-02 00:00:00:TIMESTAMP(3)))],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      ;
  auto rel13554 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d", "o_ol_cnt"},
                         catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$3"], expressions::DateConstant(1325462400000)) &
                    lt(arg["$3"], expressions::DateConstant(1609545600000)));
          })
          // (condition=[AND(>=($3, 2007-01-02 00:00:00:TIMESTAMP(3)), <($3,
          // 2012-01-02 00:00:00:TIMESTAMP(3)))],
          // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
          // isS=[false])
          .join(
              rel13550,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["$0"].as("PelagoJoin#13553", "bk_0"),
                    build_arg["$1"].as("PelagoJoin#13553", "bk_1"),
                    build_arg["$2"].as("PelagoJoin#13553", "bk_2"),
                    build_arg["$3"].as("PelagoJoin#13553", "bk_3")}
                    .as("PelagoJoin#13553", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["$0"].as("PelagoJoin#13553", "pk_0"),
                    probe_arg["$1"].as("PelagoJoin#13553", "pk_1"),
                    probe_arg["$2"].as("PelagoJoin#13553", "pk_2"),
                    probe_arg["$3"].as("PelagoJoin#13553", "pk_3")}
                    .as("PelagoJoin#13553", "pk");
              },
              10, 256 * 1024 * 1024)
          // (condition=[AND(=($4, $0), =($5, $1), =($6, $2), =($7, $3))],
          // joinType=[inner], rowcnt=[7.68E8], maxrow=[3000000.0],
          // maxEst=[3000000.0], h_bits=[28], build=[RecordType(BIGINT o_id,
          // INTEGER o_d_id, INTEGER o_w_id, TIMESTAMP(0) o_entry_d)],
          // lcount=[6.7109666771656204E10], rcount=[7.68E8],
          // buildcountrow=[7.68E8], probecountrow=[7.68E8])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$8"]).as("PelagoProject#13554", "$0"),
                    (arg["$0"]).as("PelagoProject#13554", "$1"),
                    (arg["$1"]).as("PelagoProject#13554", "$2"),
                    (arg["$2"]).as("PelagoProject#13554", "$3"),
                    (arg["$3"]).as("PelagoProject#13554", "$4")};
          })
      // (o_ol_cnt=[$8], o_id0=[$0], o_d_id0=[$1], o_w_id0=[$2],
      // o_entry_d0=[$3],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      ;
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 6]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .join(
          rel13554,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$1"].as("PelagoJoin#13556", "bk_1"),
                build_arg["$3"].as("PelagoJoin#13556", "bk_3"),
                build_arg["$2"].as("PelagoJoin#13556", "bk_2")}
                .as("PelagoJoin#13556", "bk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$0"].as("PelagoJoin#13556", "pk_0"),
                probe_arg["$2"].as("PelagoJoin#13556", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#13556", "pk_1")}
                .as("PelagoJoin#13556", "pk");
          },
          10, 256 * 1024 * 1024)
      // (condition=[AND(=($1, $5), =($3, $7), =($2, $6))], joinType=[inner],
      // rowcnt=[1.92E8], maxrow=[9.0E12], maxEst=[6.7108864E7], h_bits=[28],
      // build=[RecordType(INTEGER o_ol_cnt, BIGINT o_id0, INTEGER o_d_id0,
      // INTEGER o_w_id0, TIMESTAMP(0) o_entry_d0)],
      // lcount=[1.985514608872911E10], rcount=[3.072E10],
      // buildcountrow=[1.92E8], probecountrow=[3.072E10])
      .filter([&](const auto &arg) -> expression_t {
        return ge(arg["$8"], arg["$4"]);
      })
      // (condition=[>=($8, $4)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$0"]).as("PelagoProject#13558", "$0")};
      })
      // (o_ol_cnt=[$0],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#13559", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (expression_t{1}).as("PelagoAggregate#13559", "$1"), 1, 0,
                SUM}};
          },
          10, 1024 * 1024)
      // (group=[{0}], order_count=[COUNT()],
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
        return {arg["$0"].as(outrel, "o_ol_cnt"),
                arg["$1"].as(outrel, "order_count")};
      })
      // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<4>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                             Tr aff_reduce) {
  std::string revenue = "revenue";
  auto ctx = new ParallelContext(
      "ch_Q" + std::to_string(Qid) + "_" + typeid(Tplugin).name(), false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel13948 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d"}, catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$3"], expressions::DateConstant(1325462400000)) &
                    lt(arg["$3"], expressions::DateConstant(1609545600000)));
          })
      // (condition=[AND(>=($3, 2007-01-02 00:00:00:TIMESTAMP(3)), <($3,
      // 2012-01-02 00:00:00:TIMESTAMP(3)))],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      ;
  auto rel13955 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_order",
                         {"o_id", "o_d_id", "o_w_id", "o_entry_d", "o_ol_cnt"},
                         catalog)
          // (table=[[SSB, tpcc_order]], fields=[[0, 1, 2, 4, 6]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$3"], expressions::DateConstant(1325462400000)) &
                    lt(arg["$3"], expressions::DateConstant(1609545600000)));
          })
          // (condition=[AND(>=($3, 2007-01-02 00:00:00:TIMESTAMP(3)), <($3,
          // 2012-01-02 00:00:00:TIMESTAMP(3)))],
          // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
          // isS=[false])
          .join(
              rel13948,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["$0"].as("PelagoJoin#13951", "bk_0"),
                    build_arg["$1"].as("PelagoJoin#13951", "bk_1"),
                    build_arg["$2"].as("PelagoJoin#13951", "bk_2"),
                    build_arg["$3"].as("PelagoJoin#13951", "bk_3")}
                    .as("PelagoJoin#13951", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["$0"].as("PelagoJoin#13951", "pk_0"),
                    probe_arg["$1"].as("PelagoJoin#13951", "pk_1"),
                    probe_arg["$2"].as("PelagoJoin#13951", "pk_2"),
                    probe_arg["$3"].as("PelagoJoin#13951", "pk_3")}
                    .as("PelagoJoin#13951", "pk");
              },
              24, 256 * 1024 * 1024)
          // (condition=[AND(=($4, $0), =($5, $1), =($6, $2), =($7, $3))],
          // joinType=[inner], rowcnt=[7.68E8], maxrow=[3000000.0],
          // maxEst=[3000000.0], h_bits=[28], build=[RecordType(BIGINT o_id,
          // INTEGER o_d_id, INTEGER o_w_id, TIMESTAMP(0) o_entry_d)],
          // lcount=[6.7109666771656204E10], rcount=[7.68E8],
          // buildcountrow=[7.68E8], probecountrow=[7.68E8])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$8"]).as("PelagoProject#13952", "$0"),
                    (arg["$0"]).as("PelagoProject#13952", "$1"),
                    (arg["$1"]).as("PelagoProject#13952", "$2"),
                    (arg["$2"]).as("PelagoProject#13952", "$3"),
                    (arg["$3"]).as("PelagoProject#13952", "$4")};
          })
          // (o_ol_cnt=[$8], o_id0=[$0], o_d_id0=[$1], o_w_id0=[$2],
          // o_entry_d0=[$3],
          // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .pack()
          // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64],
          // intrait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
          // inputRows=[1.92E8], cost=[{368.368 rows, 368.0 cpu, 0.0 io}])
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
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_o_id", "ol_d_id", "ol_w_id", "ol_delivery_d"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[0, 1, 2, 6]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU, aff_parallel())
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .join(
          rel13955,
          [&](const auto &build_arg) -> expression_t {
            return expressions::RecordConstruction{
                build_arg["$1"].as("PelagoJoin#13958", "bk_1"),
                build_arg["$3"].as("PelagoJoin#13958", "bk_3"),
                build_arg["$2"].as("PelagoJoin#13958", "bk_2")}
                .as("PelagoJoin#13958", "pk");
          },
          [&](const auto &probe_arg) -> expression_t {
            return expressions::RecordConstruction{
                probe_arg["$0"].as("PelagoJoin#13958", "pk_0"),
                probe_arg["$2"].as("PelagoJoin#13958", "pk_2"),
                probe_arg["$1"].as("PelagoJoin#13958", "pk_1")}
                .as("PelagoJoin#13958", "bk");
          },
          24, 256 * 1024 * 1024)
      // (condition=[AND(=($1, $5), =($3, $7), =($2, $6))], joinType=[inner],
      // rowcnt=[1.92937984E8], maxrow=[9.0E12], maxEst=[6.7108864E7],
      // h_bits=[28], build=[RecordType(INTEGER o_ol_cnt, BIGINT o_id0, INTEGER
      // o_d_id0, INTEGER o_w_id0, TIMESTAMP(0) o_entry_d0)],
      // lcount=[1.9956846453055954E10], rcount=[1.536E10],
      // buildcountrow=[1.92937984E8], probecountrow=[1.536E10])
      .filter([&](const auto &arg) -> expression_t {
        return ge(arg["$8"], arg["$4"]);
      })
      // (condition=[>=($8, $4)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // isS=[false])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$0"]).as("PelagoProject#13960", "$0")};
      })
      // (o_ol_cnt=[$0],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#13961", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (expression_t{1}).as("PelagoAggregate#13961", "$1"), 1, 0,
                SUM}};
          },
          10, 1024 * 1024)
      // (group=[{0}], order_count=[COUNT()],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .pack()
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.cX86_64],
      // intrait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // inputRows=[4.8E7], cost=[{36.836800000000004 rows, 36.800000000000004
      // cpu, 0.0 io}])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#13965", "$0")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{
                (arg["$1"]).as("PelagoAggregate#13965", "$1"), 1, 0, SUM}};
          },
          10, 1024 * 1024)
      // (group=[{0}], order_count=[$SUM0($1)],
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
        return {arg["$0"].as(outrel, "o_ol_cnt"),
                arg["$1"].as(outrel, "order_count")};
      })
      // (trait=[ENUMERABLE.[0].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

#endif /* HARMONIA_QUERIES_CH_Q4_HPP_ */
