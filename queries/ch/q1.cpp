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

PreparedStatement q_ch_c1t() {
  std::string count_order = "count_order";
  auto ctx = new ParallelContext("main2", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<AeolusCowPlugin>(tpcc_orderline,
                             {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                             catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg[ol_delivery_d],
                  expressions::DateConstant(/*FIX*/ 904694400000));
      })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    expression_t{1}.as(tpcc_orderline, count_order)};
          },
          {SUM /*fix*/, SUM, SUM, SUM})
      .print([&](const auto &arg) -> std::vector<expression_t> {
        std::string outrel = "out";

        std::vector<expression_t> ret{
            arg[ol_number].as(outrel, ol_number),
            arg[ol_quantity].as(outrel, "sum_qty"),
            arg[ol_amount].as(outrel, "sum_amount"),
            (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
            (arg[ol_amount] / arg[count_order].template as<FloatType>())
                .as(outrel, "avg_amount"),
            arg[count_order]};

        std::vector<RecordAttribute *> args;
        args.reserve(ret.size());

        for (const auto &e : ret) {
          args.emplace_back(new RecordAttribute{e.getRegisteredAs()});
        }

        InputInfo *datasetInfo = catalog.getOrCreateInputInfo(outrel, ctx);
        datasetInfo->exprType =
            new BagType{RecordType{std::vector<RecordAttribute *>{args}}};

        return ret;
      })
      .prepare();
}

PreparedStatement q_ch2_c1t() {
  return PreparedStatement::from(
      "/scratch/chrysoge/pelago_sigmod2020_htap/src/htap/ch-plans/q1.json",
      "main2");
}
