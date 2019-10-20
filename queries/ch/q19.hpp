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

#ifndef HARMONIA_QUERIES_CH_Q19_HPP_
#define HARMONIA_QUERIES_CH_Q19_HPP_

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
PreparedStatement q_ch19_c1t() {
  std::string revenue = "revenue";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel2560 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_item", {"i_id", "i_price"}, catalog)
          // (table=[[SSB, tpcc_item]], fields=[[0, 3]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#2559", "$0"),
                    (ge(arg["$1"], ((double)1))).as("PelagoProject#2559", "$1"),
                    (le(arg["$1"], ((double)400000)))
                        .as("PelagoProject#2559", "$2")};
          })
          // (i_id=[$0], >==[>=($1, 1)], <==[<=($1, 400000)],
          // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (arg["$1"] & arg["$2"]);
          })
      // (condition=[AND($1, $2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      ;
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_w_id", "ol_i_id", "ol_quantity", "ol_amount"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[2, 4, 7, 8]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$1"]).as("PelagoProject#2562", "$0"),
                (arg["$3"]).as("PelagoProject#2562", "$1"),
                (ge(arg["$2"], 1)).as("PelagoProject#2562", "$2"),
                (le(arg["$2"], 10)).as("PelagoProject#2562", "$3"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 2) | eq(arg["$0"], 3)))
                    .as("PelagoProject#2562", "$4"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 2) | eq(arg["$0"], 4)))
                    .as("PelagoProject#2562", "$5"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 5) | eq(arg["$0"], 3)))
                    .as("PelagoProject#2562", "$6")};
      })
      // (ol_i_id=[$1], ol_amount=[$3], >==[>=($2, 1)], <==[<=($2, 10)],
      // OR=[OR(=($0, 1), =($0, 2), =($0, 3))], OR5=[OR(=($0, 1), =($0, 2),
      // =($0, 4))], OR6=[OR(=($0, 1), =($0, 5), =($0, 3))],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return (arg["$2"] & arg["$3"] & (arg["$4"] | arg["$5"] | arg["$6"]));
      })
      // (condition=[AND($2, $3, OR($4, $5, $6))],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64],
      // isS=[false])
      .join(
          rel2560,
          [&](const auto &build_arg) -> expression_t {
            return build_arg["$0"].as("PelagoJoin#2564", "bk_0");
          },
          [&](const auto &probe_arg) -> expression_t {
            return probe_arg["$0"].as("PelagoJoin#2564", "pk_0");
          },
          10, 1024 * 1024)
      // (condition=[=($3, $0)], joinType=[inner], rowcnt=[2.56E7],
      // maxrow=[100000.0], maxEst=[100000.0], h_bits=[27],
      // build=[RecordType(INTEGER i_id, BOOLEAN >=, BOOLEAN <=)],
      // lcount=[1.3944357272154548E9], rcount=[7.68E9], buildcountrow=[2.56E7],
      // probecountrow=[7.68E9])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$4"]).as("PelagoProject#2565", "$0")};
      })
      // (ol_amount=[$4],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoAggregate#2566", "$0")};
          },
          {SUM})
      // (group=[{}], revenue=[SUM($0)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "revenue")};
      })
      // (trait=[ENUMERABLE.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])

      .prepare();
}

template <typename Tplugin>
PreparedStatement q_ch19_cpar(DegreeOfParallelism dop,
                              std::unique_ptr<Affinitizer> aff_parallel,
                              std::unique_ptr<Affinitizer> aff_parallel2,
                              std::unique_ptr<Affinitizer> aff_reduce) {
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel4073 =
      RelBuilder{ctx}
          .scan<Tplugin>("tpcc_item", {"i_id", "i_price"}, catalog)
          // (table=[[SSB, tpcc_item]], fields=[[0, 3]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, DeviceType::CPU,
              std::move(aff_parallel2))
          // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle.none])
          .unpack()
          // (trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#4072", "$0"),
                    (ge(arg["$1"], ((double)1))).as("PelagoProject#4072", "$1"),
                    (le(arg["$1"], ((double)400000)))
                        .as("PelagoProject#4072", "$2")};
          })
          // (i_id=[$0], >==[>=($1, 1)], <==[<=($1, 400000)],
          // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64])
          .filter([&](const auto &arg) -> expression_t {
            return (arg["$1"] & arg["$2"]);
          })
      // (condition=[AND($1, $2)],
      // trait=[Pelago.[].unpckd.X86_64.homBrdcst.hetSingle.cX86_64],
      // isS=[false])
      ;
  return RelBuilder{ctx}
      .scan<Tplugin>("tpcc_orderline",
                     {"ol_w_id", "ol_i_id", "ol_quantity", "ol_amount"},
                     catalog)
      // (table=[[SSB, tpcc_orderline]], fields=[[2, 4, 7, 8]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU,
              std::move(aff_parallel))
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$1"]).as("PelagoProject#4076", "$0"),
                (arg["$3"]).as("PelagoProject#4076", "$1"),
                (ge(arg["$2"], 1)).as("PelagoProject#4076", "$2"),
                (le(arg["$2"], 10)).as("PelagoProject#4076", "$3"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 2) | eq(arg["$0"], 3)))
                    .as("PelagoProject#4076", "$4"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 2) | eq(arg["$0"], 4)))
                    .as("PelagoProject#4076", "$5"),
                ((eq(arg["$0"], 1) | eq(arg["$0"], 5) | eq(arg["$0"], 3)))
                    .as("PelagoProject#4076", "$6")};
      })
      // (ol_i_id=[$1], ol_amount=[$3], >==[>=($2, 1)], <==[<=($2, 10)],
      // OR=[OR(=($0, 1), =($0, 2), =($0, 3))], OR5=[OR(=($0, 1), =($0, 2),
      // =($0, 4))], OR6=[OR(=($0, 1), =($0, 5), =($0, 3))],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .filter([&](const auto &arg) -> expression_t {
        return (arg["$2"] & arg["$3"] & (arg["$4"] | arg["$5"] | arg["$6"]));
      })
      // (condition=[AND($2, $3, OR($4, $5, $6))],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // isS=[false])
      .join(
          rel4073,
          [&](const auto &build_arg) -> expression_t {
            return build_arg["$0"].as("PelagoJoin#4078", "bk_0");
          },
          [&](const auto &probe_arg) -> expression_t {
            return probe_arg["$0"].as("PelagoJoin#4078", "pk_0");
          },
          18, 1024 * 1024)
      // (condition=[=($3, $0)], joinType=[inner], rowcnt=[2.56E7],
      // maxrow=[100000.0], maxEst=[100000.0], h_bits=[27],
      // build=[RecordType(INTEGER i_id, BOOLEAN >=, BOOLEAN <=)],
      // lcount=[1.3944357272154548E9], rcount=[3.84E9], buildcountrow=[2.56E7],
      // probecountrow=[3.84E9])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$4"]).as("PelagoProject#4079", "$0")};
      })
      // (ol_amount=[$4],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoAggregate#4080", "$0")};
          },
          {SUM})
      // (group=[{}], revenue=[SUM($0)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, std::move(aff_reduce))
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoAggregate#4082", "$0")};
          },
          {SUM})
      // (group=[{}], revenue=[SUM($0)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "revenue")};
      })
      // (trait=[ENUMERABLE.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

// template <typename Tp, typename Tr>
// PreparedStatement q_ch19(DegreeOfParallelism dop, Tp aff_parallel,
//                          Tr aff_reduce) {
//   if (dop == DegreeOfParallelism{1}) return q_ch19_c1t();
//   return q_ch19_cpar(dop, aff_parallel(), aff_parallel(), aff_reduce());
// }

#endif /* HARMONIA_QUERIES_CH_Q19_HPP_ */
