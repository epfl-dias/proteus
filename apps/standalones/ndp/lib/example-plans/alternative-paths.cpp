/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <lib/examples.hpp>

/**
 * Generate a plan in which: each server, reduces 50% of the data locally
 * and send the rest 50% of the data to the ResultServer, to be aggregated
 * there. Then, all the partial aggregates are combined by the ResultServer
 * to produce the final result.
 *
 * In this plan, two paths are taken, either local or remote filter+reduce.
 * The two paths are the logically the same.
 */
RelBuilder generateAlternativePathsSymmetricPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager) {
  auto builder2 = builder
                      /**
                       * The scan will return empty results in compute nodes
                       * where there are no data.
                       */
                      .scan(
                          /* relName */ "inputs/ssbm100/date.bin",
                          /* relAttrs */ {"d_datekey", "d_year"},
                          /* catalog */ catalog,
                          /* pg */ pg{"distributed-block"})
                      .router_scaleout(
                          /* hash */
                          [&](const auto &arg) -> std::optional<expression_t> {
                            /* select at random either to do local reduction or
                             * remote reduction. The optimal ratio can be tuned
                             * based xpected processing speed and network
                             */
                            return cond(eq(expressions::rand() % 2, 1),
                                        clusterManager.getResultServerId(),
                                        clusterManager.getLocalServerId());
                          },
                          /* fanout */
                          DegreeOfParallelism{clusterManager.getNumExecutors()},
                          /* slack */ 8,
                          /* p */ RoutingPolicy::HASH_BASED,
                          /* target */ DeviceType::CPU)
                      /*
                       * If the current task came from another node, this
                       * memmove will pull the data locally
                       */
                      .memmove_scaleout(8);

  /*
   * Now reduce in each node and send the result to ResultServer for the
   * final aggregation.
   */
  return naivelyParallelize(builder2, generateLocalReductionTask)
      .router_scaleout(
          /* hash */
          [&](const auto &arg) -> std::optional<expression_t> {
            return clusterManager.getResultServerId();
          },
          /* fanout */
          DegreeOfParallelism{clusterManager.getNumExecutors()},
          /* slack */ 8,
          /* p */ RoutingPolicy::HASH_BASED,
          /* target */ DeviceType::CPU)
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["year_sq_sum"], arg["year_sum"], arg["cnt"]};
          },
          {SUM, SUM, SUM})
      .project(
          /* expr */
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["cnt"].as("out", "cnt"),
                    (arg["year_sum"] / arg["cnt"]).as("out", "avg"),
                    ((arg["year_sq_sum"] / arg["cnt"]) -
                     ((arg["year_sum"] / arg["cnt"]) *
                      (arg["year_sum"] / arg["cnt"])))
                        .as("out", "var")};
          })
      .print(/* pgType */ pg{"pm-csv"});
}
