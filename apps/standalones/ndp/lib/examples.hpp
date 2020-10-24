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

/**
 * Functionalities
 * =
 *
 * The following examples demonstrate the expressiveness and validity of the
 * NDP protocol, by show casing the following functionalities:
 *
 * 1. Shipping offloading tasks from compute to storage nodes as well as to
 * offload subtasks from storage node to the internal storage hierarchy.
 *
 *      `generateSingleThreadedPlan` shows how simple tasks are described,
 *      `generateMultiThreadedPlan` shows how such tasks are parallelized,
 *      `generatePlanComposition` shows how subtasks are composed.
 *      All these functions show how DeviceType and Routing policies are
 *      attached to nodes so that they allow offloading to other node-internal
 *      device (types) like the internal PCIe devices.
 *      Furthermore, while these plans are demosntrational and use
 *      proteus' RelBuilder, a trivial interpreter can process our automatically
 *      generated JSON plans into the related RelBuilder calls to automate the
 *      parsing.
 *
 * 2. Supporting task dynamic offloading, making NDP tasks not limited to
 * pre-defined operators.
 *
 *      `generateMultiServerPlan` shows how plans cross across the servers,
 *      `generateMultiServerParallelReductionPlan` combines multi-server with
 *      multiple internal execution instances,
 *      `generatePushFilterToStorageNodesPlan` and
 *      `generatePushFullTaskToStorageNodesPlan` show how plans express
 *      different static task placements.
 *      `generateAlternativePathsSymmetricPlan` and
 *      `generateAlternativePathsNonSymmetricPlan` show how more dynamic flows
 *      and load balancing is possible using more complex operator DAGs.
 *
 *      The relational/monoid operators described by the plans allow different
 *      operations to be composed from relational and monoid verbs, allowing
 *      generic data analytic tasks to be offload to the system nodes.
 *
 * 3. Sending the location and format of the data that need to be processed by
 * offloading tasks, to read and parse data.
 *
 *      All the plans use plugins to provide the information about the type of
 *      the inputs and allow the operators to specialize themselves to the input
 *      nodes. Furthermore, plans pack/unpack data to optimize for data
 *      transfers and parallelization. The pack can also receive more complex
 *      expressions or register through the ".as" functions intermediate
 *      expressions to different formats.
 *
 * 4. Sending task-specific configuration data of the offloading task.
 *
 *      All the operators are parameterizable and configurable through an
 *      operator-specific set of parameters.
 *      This set of parameters is passed as:
 *      i.   RelBuilder/JSON parameters
 *      ii.  Calls to the ClusterManager and/or CatalogParser
 *      iii. Through the session objects
 *
 * 5. Returning results to computing nodes.
 *
 *      All plans use a router_scaleout to gather the flow into the
 *      ResultServer. As routers* support multiple targets as well,
 *      trivially these plans can be extended to save the results
 *      into multiple compute nodes (or even storage nodes).
 *
 *
 *
 * Fundamental NDP protocol components
 * =
 *
 * To support the aforementioned functionalities the provided examples also show
 * the fundamental operations required for the generic NDP protocol.
 *
 * 1. Dynamic offloading through compute-aware query plans for task placement
 *
 *      The plans in `generateMultiServerPlan`,
 *      `generateMultiServerParallelReductionPlan`,
 *      `generatePushFilterToStorageNodesPlan` and
 *      `generatePushFullTaskToStorageNodesPlan` show how different task
 *      placements are achieved through the plans.
 *      This provides the required mechanisms for task placement near the data
 *      and control over the placement of subtasks in a granural as well as
 *      flexible manner.
 *
 * 2. NDP-aware query through computational primitives for task descriptions
 *
 *      The plans in `generateSingleThreadedPlan`,
 *      `generateMultiThreadedPlan` and
 *      `generatePlanComposition` show different task descriptions and the
 *      many of our verbs (relational operators and monoids as well as
 *      attribute-level expressions).
 *      Using the relational verbs, task description can be componsed and
 *      using the operators for the task placement, specific tasks can
 *      be placed on specific nodes.
 *
 * 3. Synergetic execution through fluid plans for storage-compute cooperation
 *
 *      The plans in `generateAlternativePathsSymmetricPlan` and
 *      `generateAlternativePathsNonSymmetricPlan` show how paths can have
 *      different paths to not only alternate but also provide different
 *      alternative paths across storage and compute nodes.
 *      Using such alternatives enables minimizing the data transfers and/or
 *      load of each device as well as choke points.
 *      Furthermore, it reduces the potential of bad planning decision to
 *      impact performance, as alternative paths can be specified to allow
 *      fall-backs based on the load.
 *
 * 4. Adaptive scheduling through policies for int(er/ra)-cluster load balancing
 *
 *      All the plans contain slack parameters in the nodes that contains
 *      asynchronous consumer-producer pipes (like router* and memmove* nodes).
 *      This slack allows propagating information across the nodes to notify
 *      producers that consumers are running slow and change paths.
 *
 *      The routing policies and the target device/affinitization provided in
 *      all the routing points allow flexible scheduling and load balancing.
 *      We showcase the case with enum policies, but proteus' router operators
 *      use classes that are customized based on the affinities and the target
 *      devices to produce different placement policies at runtime based on the
 *      allowed devices. Thus, the enums can be replaced by class instances that
 *      teach the routers different policies.
 *
 *      Furthermore, the split/union operators are extensions of the routers
 *      that allow data flows to split and reconverge to provide alternative
 *      paths and thus increase the opportunities for pressure relief and
 *      load distribution based on runtime behaviors.
 */

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
    proteus::distributed::ClusterManager &clusterManager);

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
    proteus::distributed::ClusterManager &clusterManager);

#endif  // PROTEUS_EXAMPLES_HPP
