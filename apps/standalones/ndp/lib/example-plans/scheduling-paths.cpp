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
 * The two paths are the logically different:
 *  1. Path one: performs full reduction in local node,
 *  2. Path two: filters the data locally and sends them to a random node
 *     for the aggregation.
 */
RelBuilder generateAlternativePathsNonSymmetricPlan(
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
          /**
           * Splits the data flow into multiple ones to allow
           * different alternatives.
           * The policy will determine which alternative is selected
           * for each input, in combination with the target type and
           * the affinitization policies.
           * The slack provides load balancing and backpressure.
           *
           * @param     alternatives    number of output flows
           * @param     slack           max number of on-the-fly
           *                            tasks on each flow
           * @param     p               policy determine how
           *                            the target flow is selected
           * @param     target          the target processing units
           */
      .split(
          /* alternatives */ 2,
          /* slack */ 8000,
          /* p */ RoutingPolicy::RANDOM,
          /* target */ DeviceType::CPU);

  auto alt1 = naivelyParallelize(builder2, generateLocalReductionTask)
      .router_scaleout(
          /* hash */
          [&](const auto &arg) -> std::optional<expression_t> {
            return clusterManager.getResultServerId();
          },
          /* fanout */
          DegreeOfParallelism{clusterManager.getNumExecutors()},
          /* slack */ 8,
          /* p */ RoutingPolicy::HASH_BASED,
          /* target */ DeviceType::CPU);

  auto alt2 =
      naivelyParallelize(builder2,
                         [&](RelBuilder b) {
                           return b.unpack()
                               .filter([&](const auto &arg) -> expression_t {
                                 return gt(arg["d_datekey"] % 30, 15);
                               })
                               .pack();
                         })
          .router_scaleout(
              DegreeOfParallelism{clusterManager.getNumExecutors()}, 8,
              /*
               * Send to any random node, including Compute or Storage nodes
               * Other more dynamic policies will be used for the next
               * deliverables, although the slack already provides load
               * balancing capabilites: using the random policy and a slack
               * of 8 means that if a node has already send 8 unfinished
               * packets to another node, then it will try to find another
               * node.
               */
              RoutingPolicy::RANDOM, DeviceType::CPU)
          .memmove_scaleout(8)
          .unpack()
              /*
               * The router above distributed blocks, so this reduce will provide
               * worker-local results, we need another router afterwards to
               * combine them.
               */
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {
                    (arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
                    arg["d_year"].as("tmp", "year_sum"),
                    expression_t{1}.as("tmp", "cnt")};
              },
              {SUM, SUM, SUM})
          .router_scaleout(
              /* hash */
              [&](const auto &arg) -> std::optional<expression_t> {
                return clusterManager.getResultServerId();
              },
              /* fanout */
              DegreeOfParallelism{clusterManager.getNumExecutors()},
              /* slack */ 8,
              /* p */ RoutingPolicy::HASH_BASED,
              /* target */ DeviceType::CPU);

  /* Now combine the two paths */

  return alt1
      /**
       * Union the items from the current flow and the others
       *
       * @param     others  flows to unify with current one
       */
      .unionAll(/* others */ {alt2})
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
                    arg["d_year"].as("tmp", "year_sum"),
                    expression_t{1}.as("tmp", "cnt")};
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
