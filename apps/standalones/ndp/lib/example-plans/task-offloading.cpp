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
 * Generate a simple plan to demonstrate a scale-out query that pulls the data
 * from all the servers, into the server indicated as the ResultServer (the
 * server that will return the result to the application)
 */
RelBuilder generateMultiServerPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager) {
  auto builder2 =
      builder
          /**
           * The scan will return empty results in compute nodes where there are
           * no data.
           */
          .scan(
              /* relName */ "inputs/ssbm100/date.bin",
              /* relAttrs */ {"d_datekey", "d_year"},
              /* catalog */ catalog,
              /* pg */ pg{"distributed-block"})
          /**
           * Similar to router, but the scale-out case.
           *
           * RouterScaleout distributes the inputs to @p fanout executors
           * (machines)
           *
           * How inputs are distributed across executors depends on the
           * RoutingPolicies.
           *
           * The RoutingPolicy::HASH_BASED specifies that we are going to send
           * the data to a specific node, given by the @p hash expression.
           *
           * Note that routers do not transfer data but only commands, such as
           * which data are going to be pulled from the other side.
           *
           * @param     hash    expression used for directing the routing,
           * either to specific machines through a constant, or using a data
           * property (for example for hash-based routing)
           * @param     fanout  Number of target executors
           * @param     slack   Slack in each pipe, used for load-balancing and
           *                    backpressure. Limits the number of on-the-fly
           *                    requests.
           */
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
          /**
           * Pull data into the local machine, if they are not already here.
           */
          .memmove_scaleout(8);

  /* Now a single thread reduces in the ResultServer the data.
   * The other servers will receive nothing and stay idle.
   */
  return generateLocalReductionTask(builder2)
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

/**
 * Generate a simple plan to demonstrate a scale-out query that pulls the data
 * from all the servers and executes the reduction in parallel, into the server
 * indicated as the ResultServer (the server that will return the result to the
 * application)
 */
RelBuilder generateMultiServerParallelReductionPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager) {
  auto builder2 =
      builder
          /**
           * The scan will return empty results in compute nodes where there are
           * no data.
           */
          .scan(
              /* relName */ "inputs/ssbm100/date.bin",
              /* relAttrs */ {"d_datekey", "d_year"},
              /* catalog */ catalog,
              /* pg */ pg{"distributed-block"})
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
          /**
           * Pull data into the local machine, if they are not already here.
           */
          .memmove_scaleout(8);

  /* Local reduction will effectively run on the compute node that will
   * return the results
   */
  return naivelyParallelize(builder2, generateLocalReductionTask)
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

/**
 * Generate a plan that filters the data at their source location
 */
RelBuilder generatePushFilterToStorageNodesPlan(
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
                          /* pg */ pg{"distributed-block"});
  return
      /*
       * Use multiple threads in each machine to filter the data.
       *
       * Machines with no data, like the compute nodes will be idle in
       * that part of the query.
       */
      naivelyParallelize(builder2,
                         [](RelBuilder b) {
                           return b.unpack()
                               .filter([&](const auto &arg) -> expression_t {
                                 return gt(arg["d_datekey"] % 30, 15);
                               })
                               /**
                                * Pack tuples to blocks
                                *
                                * @see Chrysogelos et al, VLDB2019
                                */
                               .pack();
                         })
          /**
           * Move filtered data to the ResultServer
           * (their corresponding tasks to be exact)
           */
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
          /**
           * and pull that data there.
           */
          .memmove_scaleout(8)
          .unpack()
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {
                    (arg["d_year"] * arg["d_year"]).as("tmp", "year_sq_sum"),
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

/**
 * Generate a plan that filters the data at their source location
 */
RelBuilder generatePushFullTaskToStorageNodesPlan(
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
                          /* pg */ pg{"distributed-block"});
  return
      /*
       * Use multiple threads in each machine to filter the data.
       *
       * Machines with no data, like the compute nodes will be idle in
       * that part of the query.
       */
      naivelyParallelize(builder2, generateLocalReductionTask)
          /**
           * Move filtered data to the ResultServer
           * (their corresponding tasks to be exact)
           */
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
          /**
           * The task description contains a single tuple; there are no data to
           * transfer/unpack (the tuple is similar to the block handles
           * transferred as part of the task description in previous queries).
           */
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
