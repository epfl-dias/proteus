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
#include "q01.hpp"

#include "ch-queries.hpp"

static int q_instance = 0;

PreparedStatement Q_1_cpar(DegreeOfParallelism dop, const aff_t &aff_parallel,
                           const aff_t &aff_reduce, DeviceType dev,
                           const scan_t &scan) {
  std::string count_order = "count_order";

  auto rel =
      scan(tpcc_orderline, {ol_delivery_d, ol_number, ol_amount, ol_quantity})
          .router(dop, 8, RoutingPolicy::RANDOM, dev, aff_parallel())
      //      .memmove(8, dev == DeviceType::CPU)
      ;

  if (dev == DeviceType::GPU) {
    rel = rel.memmove(8, dev == DeviceType::CPU).to_gpu();
  }

  rel = rel.unpack()
            // .filter([&](const auto &arg) -> expression_t {
            //   return gt(arg[ol_delivery_d],
            //             expressions::DateConstant(/*FIX*/ 904694400000));
            // })
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                          GpuAggrMatExpr{expression_t{1}.as(
                                             arg[ol_number].getRelationName(),
                                             count_order),
                                         1, 32, SUM},
                          GpuAggrMatExpr{arg[ol_amount], 1, 64, SUM}};
                },
                5, 128 * 1024);

  if (dev == DeviceType::GPU) {
    rel = rel.to_cpu();
  }

  rel = rel.router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                   DeviceType::CPU, aff_reduce())
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                          GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                          GpuAggrMatExpr{arg[count_order], 3, 0, SUM}};
                },
                5, 128)
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                          arg[count_order]};
                },
                {direction::ASC, direction::NONE, direction::NONE})
            .print(
                [&](const auto &arg,
                    std::string outrel) -> std::vector<expression_t> {
                  return {arg[ol_number].as(outrel, ol_number),
                          arg[ol_quantity].as(outrel, "sum_qty"),
                          arg[ol_amount].as(outrel, "sum_amount"),
                          (arg[ol_quantity] / (arg[count_order] + 1))
                              .as(outrel, "avg_qty"),
                          (arg[ol_amount] /
                           (arg[count_order] + 1).template as<FloatType>())
                              .as(outrel, "avg_amount"),
                          arg[count_order].as(outrel, count_order)};
                },
                std::string{"CH_Q_01"} + std::to_string(q_instance++));

  return rel.prepare();
}
