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

#ifndef HARMONIA_QUERIES_CH_Q1_HPP_
#define HARMONIA_QUERIES_CH_Q1_HPP_

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
PreparedStatement q_ch_c1t() {
  std::string count_order = "count_order";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline,
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
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        std::vector<expression_t> ret{
            arg[ol_number].as(outrel, ol_number),
            arg[ol_quantity].as(outrel, "sum_qty"),
            arg[ol_amount].as(outrel, "sum_amount"),
            (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
            (arg[ol_amount] / arg[count_order].template as<FloatType>())
                .as(outrel, "avg_amount"),
            arg[count_order].as(outrel, count_order)};
        return ret;
      })
      .prepare();
}

template <typename Tplugin>
PreparedStatement q_ch_cpar(DegreeOfParallelism dop,
                            std::unique_ptr<Affinitizer> aff_parallel,
                            std::unique_ptr<Affinitizer> aff_reduce) {
  std::string count_order = "count_order";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline,
                     {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                     catalog)
      .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_parallel))
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
      .router(DegreeOfParallelism{1}, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_reduce))
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    arg[count_order]};
          },
          {SUM /*fix*/, SUM, SUM, SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        std::vector<expression_t> ret{
            arg[ol_number].as(outrel, ol_number),
            arg[ol_quantity].as(outrel, "sum_qty"),
            arg[ol_amount].as(outrel, "sum_amount"),
            (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
            (arg[ol_amount] / arg[count_order].template as<FloatType>())
                .as(outrel, "avg_amount"),
            arg[count_order].as(outrel, count_order)};
        return ret;
      })
      .prepare();
}

// PreparedStatement q_ch2_c1t() {
//   return PreparedStatement::from(
//       "/scratch/chrysoge/pelago_sigmod2020_htap/src/htap/ch-plans/q1.json",
//       __FUNCTION__);
// }

template <typename Tplugin>
PreparedStatement q_ch(DegreeOfParallelism dop,
                       std::unique_ptr<Affinitizer> aff_parallel,
                       std::unique_ptr<Affinitizer> aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch_c1t<Tplugin>();
  return q_ch_cpar<Tplugin>(dop, std::move(aff_parallel),
                            std::move(aff_reduce));
}

template <typename Tplugin>
PreparedStatement q_ch1_c1t() {
  std::string count_order = "count_order";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline,
                     {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                     catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg[ol_delivery_d],
                  expressions::DateConstant(/*FIX*/ 904694400000));
      })
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                GpuAggrMatExpr{expression_t{1}.as(tpcc_orderline, count_order),
                               3, 0, SUM}};
          },
          10, 1024 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    arg[count_order]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_number].as(outrel, ol_number),
                arg[ol_quantity].as(outrel, "sum_qty"),
                arg[ol_amount].as(outrel, "sum_amount"),
                (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
                (arg[ol_amount] / arg[count_order].template as<FloatType>())
                    .as(outrel, "avg_amount"),
                arg[count_order].as(outrel, count_order)};
      })
      .prepare();
}

template <typename Tplugin>
PreparedStatement q_ch1_cpar(DegreeOfParallelism dop,
                             std::unique_ptr<Affinitizer> aff_parallel,
                             std::unique_ptr<Affinitizer> aff_reduce) {
  std::string count_order = "count_order";
  auto ctx = new ParallelContext(__FUNCTION__, false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline,
                     {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                     catalog)
      .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_parallel))
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg[ol_delivery_d],
                  expressions::DateConstant(/*FIX*/ 904694400000));
      })
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                GpuAggrMatExpr{expression_t{1}.as(tpcc_orderline, count_order),
                               3, 0, SUM}};
          },
          10, 1024 * 1024)
      .router(DegreeOfParallelism{1}, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_reduce))
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                GpuAggrMatExpr{expression_t{1}.as(tpcc_orderline, count_order),
                               3, 0, SUM}};
          },
          10, 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    arg[count_order]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_number].as(outrel, ol_number),
                arg[ol_quantity].as(outrel, "sum_qty"),
                arg[ol_amount].as(outrel, "sum_amount"),
                (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
                (arg[ol_amount] / arg[count_order].template as<FloatType>())
                    .as(outrel, "avg_amount"),
                arg[count_order].as(outrel, count_order)};
      })
      .prepare();
}

template <typename Tplugin>
PreparedStatement q_ch1(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel,
                        std::unique_ptr<Affinitizer> aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch1_c1t<Tplugin>();
  return q_ch1_cpar<Tplugin>(dop, std::move(aff_parallel),
                             std::move(aff_reduce));
}

#endif /* HARMONIA_QUERIES_CH_Q1_HPP_ */
