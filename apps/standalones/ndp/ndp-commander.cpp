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

#include <cli-flags.hpp>
#include <ndp/ndp-common.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <olap/plan/query-result.hpp>
#include <util/glog.hpp>

#include "ndp-cli-flags.hpp"

namespace proteus {

class NDPCommandProvider : public CommandProvider {};

}  // namespace proteus

int main(int argc, char *argv[]) {
  /*
   * Phase 0. Self-discovery and initialization
   */

  /*
   * Phase 0a. Initialize the platform and warm-up the managers
   *
   * During this initialization, dynamic libraries available in the machine
   * are discovered and registered with the ctx.
   *
   * The rest of the infrastructure will query the Context to find the
   * appropriate providers.
   */
  auto ctx = proteus::from_cli::ndp("NDP", &argc, &argv);

  /*
   * Phase 1. Cluster discovery
   *
   * When an instance comes up, it communicates with the local component of the
   * cluster manager to register itself to the network and locate its peers.
   */

  /*
   * Phase 1a. Initialize cluster and detect primary
   *
   * If the instance is the primary, it checks if the JDBC listener and
   * query optimizer is up (in the same node). If not, it spawns them.
   *
   * If the instance is a secondary, it spawns up a listening thread for control
   * messages regarding the communication with the primary (heartbeats, topology
   * information, peer-discovery, etc)
   */
  ctx.getClusterManager().connect(FLAGS_primary, FLAGS_url, FLAGS_port);

  /*
   * Phase 2. Serving queries
   */
  try {
    while (true) {
      // Wait for a query to be send by the primary.
      auto query = ctx.getClusterManager().getQuery();

      ctx.getThreadPool().enqueue([query = std::move(query), &ctx]() {
        try {
          // The query contains the query plan that describes the global
          // execution
          // plan and the nodes participating in the query execution.

          /*
           * Deserialize the query plan into a DAG of operators, represented by
           * a RelBuilder instance.
           */
          auto builder =
              ctx.getPlanParser().parse(query.getQueryPlan(), query.getUUID());

          /*
           * The RelBuilder will then start invoking the plan preparation.
           * In Proteus, as it's a JIT-based engine, this will spawn the code
           * generation sequence.
           * For an non-JITed engine, it would do any final transformations
           * and/or initializations required before the query plan can be
           * executed.
           */
          auto preparedStatement = builder.prepare();

          /*
           * Notify the cluster manager that we are ready to start the query
           * execution.
           *
           * For now this is just a notification for organizational purposes,
           * but it can also give one chance to the ClusterManager to delay
           * the execution thread from starting the query or even canceling it
           * if one of the participating nodes complained.
           */
          ctx.getClusterManager().notifyReady(query.getUUID());

          /*
           * When notify ready returns, we can start query execution by pushing
           * the preparedStatement into the executor.
           *
           * Query parameters are inside the prepapred statement and the
           * generated code. They contain configuration parameters related to
           * running the query and which nodes participate in the execution.
           */
          auto result = ctx.getExecutor().run(preparedStatement);

          /*
           * The plan already contains the infomration on how to ship and save
           * the result set, but here we are lso notifying the ClusterManager
           * that the query execution in this node finished.
           *
           * If the result shipping from the saved location to the jdbc
           * connection is blocking, that's the point where the primary will
           * notify the JDBC connection to read the data. Similarly, if the
           * results are save on disk, this is the point where the
           * ClusterManager of the primary will notify the user that the results
           * are ready.
           *
           * Note that the plan will take care of collecting all the results in
           * the nodes marked as output nodes through the final router
           * operation. Thus, ClusterManagers on non-output nodes can ignore
           * this call.
           *
           * Usually, the only output node will be the primary. Even if we flush
           * multiple partitions across multiple output nodes, the primary will
           * collect through the last router the location of those files to
           * register them with the application/user.
           */
          ctx.getClusterManager().notifyFinished(query.getUUID(),
                                                 std::move(result));
        } catch (proteus::query_interrupt_command &) {
          LOG(INFO) << "Query interrupted by user";
        }
      });
    }
  } catch (proteus::shutdown_command &) {
    LOG(INFO) << "REPL terminating due to shutdown command";
  }

  /*
   * As ctx gets destroyed it hierarchically calls into the different components
   * to allow them to deinitialize and cleanup their resources
   */
  return 0;
}
