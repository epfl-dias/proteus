/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2025
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

#include "ssb-synthetic-queries.hpp"

constexpr auto query = "synth_join_zero";
PreparedStatement synthetic_join_zero(SynthJoinQueryArgs args) {
  args.morph->setQueryName(query);
  auto &topo = topology::getInstance();
  CHECK_GE(args.lo_quantity_le_predicate, 1);
  CHECK_LE(args.lo_quantity_le_predicate, 50);

  auto probe_pipeline =
      args.morph
          ->scan("lineorder", {"lo_custkey", "lo_partkey", "lo_suppkey",
                               "lo_orderdate", "lo_quantity", "lo_revenue"})
          .router(DegreeOfParallelism(topo.getGpuCount() *
                                      args.scan_memmove_dop_per_gpu),
                  args.scan_router_slack, RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<GPUAffinitizer>())
          .memmove(args.scan_memmove_slack, DeviceType::GPU)
          .router(DegreeOfParallelism(topo.getGpuCount()), 32,
                  RoutingPolicy::LOCAL, DeviceType::CPU,
                  std::make_unique<GPUAffinitizer>())
          .to_gpu()
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(
                le(arg["lo_quantity"], args.lo_quantity_le_predicate),
                expressions::Selectivity((args.lo_quantity_le_predicate * 2.0) /
                                         100.0));
          })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["lo_revenue"]).as("tmp", "revenue")};
              },
              {SUM})
          .to_cpu()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU, std::make_unique<GPUAffinitizer>())
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["revenue"])};
              },
              {SUM})
          .print(pg{"pm-csv"})
          .prepare();

  return probe_pipeline;
}
