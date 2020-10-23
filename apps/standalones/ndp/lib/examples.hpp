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

#ifndef PROTEUS_EXAMPLES_HPP
#define PROTEUS_EXAMPLES_HPP

#include <distributed-runtime/cluster-manager.hpp>
#include <olap/operators/relbuilder.hpp>

/**
 * Generate a simple plan to demonstrate the usage of the RelBuilder API
 *
 * Plans are generally produced through Proteus' query optimizer and both
 * operators and expressions are serialized as JSONs.
 *
 * (Proteus' query optimizer is build on top of Calcite)
 */
RelBuilder generateSingleThreadedPlan(RelBuilder builder,
                                      CatalogParser &catalog);

/**
 * Generate a simple parallel plan to demonstrate the usage of the RelBuilder
 *
 * Proteus' query optimizer auto-parallelizes the plans using the appropriate
 * rules.
 */
RelBuilder generateMultiThreadedPlan(RelBuilder builder,
                                     CatalogParser &catalog);

/**
 * Simple task for demonstration purposes and annotations
 */
RelBuilder generateLocalReductionTask(RelBuilder builder);
/**
 * Helper to naively parallelize inside a node
 */
RelBuilder naivelyParallelize(RelBuilder builder,
                              const std::function<RelBuilder(RelBuilder)> &f);

/**
 * Generate a simple parallel plan to demonstrate the usage of the RelBuilder
 * using multiple subtasks.
 *
 * Proteus' query optimizer auto-parallelizes the plans using the appropriate
 * rules.
 */
RelBuilder generatePlanComposition(RelBuilder builder, CatalogParser &catalog);

/**
 * Generate a simple plan to demonstrate a scale-out query that pulls the data
 * from all the servers, into the server indicated as the ResultServer (the
 * server that will return the result to the application)
 */
RelBuilder generateMultiServerPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager);

/**
 * Generate a simple plan to demonstrate a scale-out query that pulls the data
 * from all the servers and executes the reduction in parallel, into the server
 * indicated as the ResultServer (the server that will return the result to the
 * application)
 */
RelBuilder generateMultiServerParallelReductionPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager);

/**
 * Generate a plan that filters the data at their source location
 */
RelBuilder generatePushFilterToStorageNodesPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager);

/**
 * Generate a plan that filters and reduces data at their source location
 */
RelBuilder generatePushFullTaskToStorageNodesPlan(
    RelBuilder builder, CatalogParser &catalog,
    proteus::distributed::ClusterManager &clusterManager);

#endif  // PROTEUS_EXAMPLES_HPP
