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
  auto ctx = new ParallelContext("q4", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline, {ol_delivery_d, ol_quantity, ol_amount},
                     catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
               ge(arg[ol_delivery_d], expressions::DateConstant(915148800000)) &
               lt(arg[ol_delivery_d], expressions::DateConstant(1577836800000));
      })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_amount]};
          },
          {SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_amount].as(outrel, revenue)};
      })
      .prepare();
}

// template <typename Tplugin>
// PreparedStatement q_ch4_cpar(DegreeOfParallelism dop,
//                              std::unique_ptr<Affinitizer> aff_parallel,
//                              std::unique_ptr<Affinitizer> aff_reduce) {
//   std::string revenue = "revenue";
//   auto ctx = new ParallelContext("q6", false);
//   CatalogParser &catalog = CatalogParser::getInstance();
//   return RelBuilder{ctx}
//       .scan<Tplugin>(tpcc_orderline,
//                                {ol_delivery_d, ol_quantity, ol_amount},
//                                catalog)
//       .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
//               std::move(aff_parallel))
//       .unpack()
//       .filter([&](const auto &arg) -> expression_t {
//         return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
//                ge(arg[ol_delivery_d],
//                expressions::DateConstant(915148800000)) &
//                lt(arg[ol_delivery_d],
//                expressions::DateConstant(1577836800000));
//       })
//       .reduce(
//           [&](const auto &arg) -> std::vector<expression_t> {
//             return {arg[ol_amount]};
//           },
//           {SUM})
//       .router(DegreeOfParallelism{1}, 1, RoutingPolicy::RANDOM,
//       DeviceType::CPU,
//               std::move(aff_reduce))
//       .reduce(
//           [&](const auto &arg) -> std::vector<expression_t> {
//             return {arg[ol_amount]};
//           },
//           {SUM})
// .print([&](const auto &arg,
//            std::string outrel) -> std::vector<expression_t> {
//   return {arg[ol_amount].as(outrel, revenue)};
// })
//       .prepare();
// }

template <typename Tplugin>
PreparedStatement q_ch4(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel,
                        std::unique_ptr<Affinitizer> aff_reduce) {
  // if (dop == DegreeOfParallelism{1})
  return q_ch4_c1t<Tplugin>();
  // return q_ch4_cpar(dop, std::move(aff_parallel), std::move(aff_reduce));
}

#endif /* HARMONIA_QUERIES_CH_Q4_HPP_ */
