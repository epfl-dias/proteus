
Useful documentation
=

[examples.hpp](lib/examples.hpp) contains the explanation and high-level overview of the example plans.
Also, it makes the connection between the showcased examples,
the NDP protocol functionalities and the validation of its expressiveness.
The examples show how proteus' RelBuilders are used and the main verbs of the protocol is using.
RelBuilder invocations are represented by JSON plans that are communicated across the nodes through RPC.

The [MockPlanParser](./lib/mock-plan-parser.cpp) replaces Proteus' PlanParser to showcase the normal flow without requiring code generation details.

The [ClusterCommandProvier](./include/ndp/cluster_command_provider.hpp) provides a CommandProvider to relay command from the query optimizer to the system nodes.

The [ndp](./include/ndp/ndp-common.hpp) provides the entry point for the configurations.

The main executable entry point is the [ndp.cpp](ndp.cpp) which has two modes:

* __primary__ where it acts as the command receiver from the query optimizer.
    Commands are received through stdin and contains a statement preparation, statement execution or
    execution of a prepared statement directive. These directives are relayed to the secondary nodes for the actual execution. Additionally, the primary is located in the same node as the main ResultServer (which is a secondary) to facilitate data exchange through shared memory.
    The primary also waits for secondary nodes to register with it and exchange necessary configuration parameters, thus in the current implementation it should be started first.
* __secondary__ where it follows the directives send by the primary.
    Secondary nodes wait for RPC directives from the primary and join the query execution as soon
    as the primary directs them.



Main execution flow
=

Queries arrive to the JDBC connection and get optimized. The final, NDP-aware plan is serialized as a 
JSON file and communicated to the primary. Then the primary prepares the statement/query by notifying 
the participating nodes (specified by the ClusterManager, which is customizable and either use all 
the registered nodes (default) or consult the catalog and resource manager to target specific nodes).
When the primary gets notified that the nodes are ready to start executing the query, it sends them 
the execution command and waits for them to report when they have finished executing.
As soon as all participating nodes complete, the result is returned to the user/JDBC connection.

Linearizing and simplifying the execution of a single query the flow is similar to the following:

```c++
  /*
   * Phase 0. Self-discovery and initialization
   */
  /*
   * Phase 0a. Initialize the platform and warm-up the managers
   * (memory manager, infiniband manager, cluster manager etc)
   */
  auto ctx = proteus::from_cli::ndp("NDP", &argc, &argv);
  /*
   * Phase 0b. Discover dynamic libraries
   */
  // Load the execution lib
  ctx.loadExecutionLibraries();
  // Load the lib*.so with policies
  ctx.loadRoutingPolicies();
  // Load the lib*.so with load balancing policies
  ctx.loadLoadBalancingPolicies();
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
  ctx.getClusterManager().connect();
  /*
   * Phase 2. Serving queries
   */
  try {
    while (true) {
      try {
        // Wait for a query to be send by the primary.
        auto query = ctx.getClusterManager().getQuery();
        // The query contains the query plan that describes the global execution
        // plan and the nodes participating in the query execution.
        /*
         * Deserialize the query plan into a DAG of operators, represented by a
         * RelBuilder instance.
         */
        auto builder = ctx.getPlanParser().parse(query.getQueryPlan());
        /*
         * The RelBuilder will then start invoking the plan preparation.
         * In Proteus, as it's a JIT-based engine, this will spawn the code
         * generation sequence.
         * For an non-JITed engine, it would do any final transformations and/or
         * initializations required before the query plan can be executed.
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
        ctx.getClusterManager().notifyReady(query.getId());
        /*
         * When notify ready returns, we can start query execution by pushing
         * the preparedStatement into the executor.
         *
         * QueryParameters contains configuration parameters related to running
         * the query and which nodes participate in the execution.
         */
        auto result = ctx.getExecutor().run(preparedStatement);
        /*
         * The plan already contains the information on how to ship and save the
         * result set, but here we are lso notifying the ClusterManager that the
         * query execution in this node finished.
         *
         * If the result shipping from the saved location to the jdbc connection
         * is blocking, that's the point where the primary will notify the JDBC
         * connection to read the data.
         * Similarly, if the results are save on disk, this is the point where
         * the ClusterManager of the primary will notify the user that the
         * results are ready.
         *
         * Note that the plan will take care of collecting all the results in
         * the nodes marked as output nodes through the final router operation.
         * Thus, ClusterManagers on non-output nodes can ignore this call.
         *
         * Usually, the only output node will be the primary. Even if we flush
         * multiple partitions across multiple output nodes, the primary will
         * collect through the last router the location of those files to
         * register them with the application/user.
         */
        ctx.getClusterManager().notifyFinished(result);
      } catch (proteus::query_interrupt_command &) {
        LOG(INFO) << "Query interrupted by user";
      }
    }
  } catch (proteus::shutdown_command &) {
    LOG(INFO) << "REPL terminating due to shutdown command";
  }
  /*
   * As ctx gets destroyed it hierarchically calls into the different components
   * to allow them to deinitialize and cleanup their resources
   */
  return 0;
```   
