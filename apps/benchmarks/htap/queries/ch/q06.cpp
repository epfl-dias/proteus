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
#include "q06.hpp"

#include "ch-queries.hpp"

static int q_instance = 20;

PreparedStatement Q_6_cpar(DegreeOfParallelism dop, const aff_t &aff_parallel,
                           const aff_t &aff_reduce, DeviceType dev,
                           const scan_t &scan) {
  std::string revenue = "revenue";

  auto rel = scan(tpcc_orderline, {ol_delivery_d, ol_quantity, ol_amount})
                 .router(dop, 32, RoutingPolicy::LOCAL, dev, aff_parallel());

  if (dev == DeviceType::GPU) {
    rel = rel.memmove(8, dev).to_gpu();
  }

  rel = rel.unpack()
            .filter([&](const auto &arg) -> expression_t {
              return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
                     ge(arg[ol_delivery_d],
                        expressions::DateConstant(915148800000)) &
                     lt(arg[ol_delivery_d],
                        expressions::DateConstant(1644162414000));
            })
            .project([&](const auto &arg) -> std::vector<expression_t> {
              return {(arg["$0"]).as("PelagoProject#1621", "ol_delivery_d"),
                      (arg["$1"]).as("PelagoProject#1621", "ol_quantity"),
                      (arg["$2"]).as("PelagoProject#1621", "ol_amount")};
            })

            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_amount]};
                },
                {SUM});

  if (dev == DeviceType::GPU) {
    rel = rel.to_cpu();
  }

  rel = rel.router(DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM,
                   DeviceType::CPU, aff_reduce())
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_amount].as("tmp", ol_amount)};
                },
                {SUM})
            .print(pg{"csv-pm"},
                   std::string{"CH_Q_06"} + std::to_string(q_instance++));

  return rel.prepare();
}
