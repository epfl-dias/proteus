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

PreparedStatement q_sum_c1t() {
  auto ctx = new ParallelContext("main2", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<AeolusCowPlugin>(tpcc_orderline, {ol_o_id}, catalog)
      .unpack()
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .print([&](const auto &arg) -> std::vector<expression_t> {
        auto reg_as2 = new RecordAttribute(0, "t2", ol_o_id,
                                           arg[ol_o_id].getExpressionType());
        assert(reg_as2 && "Error registering expression as attribute");

        InputInfo *datasetInfo =
            catalog.getOrCreateInputInfo(reg_as2->getRelationName(), ctx);
        datasetInfo->exprType =
            new BagType{RecordType{std::vector<RecordAttribute *>{reg_as2}}};

        return {arg[ol_o_id].as(reg_as2)};
      })
      .prepare();
}

PreparedStatement q_sum_cpar(DegreeOfParallelism dop,
                             std::unique_ptr<Affinitizer> aff_parallel,
                             std::unique_ptr<Affinitizer> aff_reduce) {
  auto ctx = new ParallelContext("main3", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<AeolusCowPlugin>(tpcc_orderline, {ol_o_id}, catalog)
      .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_parallel))
      .unpack()
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .router(DegreeOfParallelism{1}, 16, RoutingPolicy::LOCAL, DeviceType::CPU,
              std::move(aff_reduce))
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .print([&](const auto &arg) -> std::vector<expression_t> {
        auto reg_as2 = new RecordAttribute(0, "t2", ol_o_id,
                                           arg[ol_o_id].getExpressionType());
        assert(reg_as2 && "Error registering expression as attribute");

        InputInfo *datasetInfo =
            catalog.getOrCreateInputInfo(reg_as2->getRelationName(), ctx);
        datasetInfo->exprType =
            new BagType{RecordType{std::vector<RecordAttribute *>{reg_as2}}};

        return {arg[ol_o_id].as(reg_as2)};
      })
      .prepare();
}

PreparedStatement q_sum(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel,
                        std::unique_ptr<Affinitizer> aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_sum_c1t();
  return q_sum_cpar(dop, std::move(aff_parallel), std::move(aff_reduce));
}
