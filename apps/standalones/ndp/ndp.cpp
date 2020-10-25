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

#include "command-provider/local-command-provider.hpp"
#include "lib/examples.hpp"
#include "ndp-cli-flags.hpp"
#include "ndp/cluster_command_provider.hpp"
#include "ndp/ndp-common.hpp"

namespace proteus {

class NDPCommandProvider : public ClusterCommandProvider {
 public:
  explicit NDPCommandProvider(proteus::ndp &ctx)
      : ClusterCommandProvider(ctx) {}
};

class MockLocalCommandProvider : public LocalCommandProvider {
 private:
  ndp *engine;
  std::map<std::string, PreparedStatement> stmts{};

 public:
  void prepareStatement(const std::string &label,
                        const std::span<const std::byte> &plan) override {
    LOG(INFO) << "Mock prepare Query";
    static size_t plan_id = 0;

    /* The query contains the query plan that describes the global execution
     * plan and the nodes participating in the query execution.
     *
     * Deserialize the query plan into a DAG of operators, represented by a
     * RelBuilder instance.
     */

    // auto builder = engine->getPlanParser().parse(plan, label);

    // sample-plans-hardcoded.
    //

    auto _builder = RelBuilderFactory{__FUNCTION__}.getBuilder();
    auto &_catalog = CatalogParser::getInstance();
    auto &_cluster_mgr = this->engine->getClusterManager();

    static std::vector<RelBuilder> plans{
        generateSingleThreadedPlan(_builder, _catalog),             // Q0
        generateMultiThreadedPlan(_builder, _catalog),              // Q1
        generateSingleThreadedPlan(_builder, _catalog),             // Q2
        generateMultiThreadedPlan(_builder, _catalog),              // Q3
        generatePlanComposition(_builder, _catalog),                // Q4
        generateMultiServerPlan(_builder, _catalog, _cluster_mgr),  // Q5
        generateMultiServerParallelReductionPlan(_builder, _catalog,
                                                 _cluster_mgr),  // Q6
        generatePushFilterToStorageNodesPlan(_builder, _catalog,
                                             _cluster_mgr),  // Q7
        generatePushFullTaskToStorageNodesPlan(_builder, _catalog,
                                               _cluster_mgr),  // Q8
        generateAlternativePathsSymmetricPlan(_builder, _catalog,
                                              _cluster_mgr),  // Q9
        generateAlternativePathsNonSymmetricPlan(_builder, _catalog,
                                                 _cluster_mgr),  // Q10
    };
    stmts.emplace(label, plans[plan_id++ % plans.size()].prepare());
    /*
     * The RelBuilder will then start invoking the plan preparation.
     * In Proteus, as it's a JIT-based engine, this will spawn the code
     * generation sequence.
     * For an non-JITed engine, it would do any final transformations and/or
     * initializations required before the query plan can be executed.
     */
    // stmts.emplace(label, builder.prepare());
    /*
     * After successful preparation of query statements, ClusterManager will
     * send notification to primary-node from each secondary node, stating that
     * query has been successfully prepared, or error otherwise.
     * */
  }

  fs::path runPreparedStatement(const std::string &label, bool echo) override {
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
    auto result = engine->getExecutor().run(stmts.at(label));
    this->store(std::move(result));
    return fs::path("/dev/shm/" + label);
    /*
     * After successful execution of query statements, ClusterManager will
     * send notification to primary-node from each secondary node, stating that
     * query has been successfully executed, or error otherwise.
     * */
  }

  explicit MockLocalCommandProvider(ndp *engine) : engine(engine) {}
  ~MockLocalCommandProvider() override = default;
};
}  // namespace proteus

// https://stackoverflow.com/a/25829178/1237824
std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) return str;
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

// https://stackoverflow.com/a/7756105/1237824
bool starts_with(const std::string &s1, const std::string &s2) {
  return s2.size() <= s1.size() && s1.compare(0, s2.size(), s2) == 0;
}

constexpr size_t clen(const char *str) {
  return (*str == 0) ? 0 : clen(str + 1) + 1;
}

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

  // Primary-node act as an interface to OLAP engine.
  if (FLAGS_primary) {
    /*
     * In case of primary-node, the process will run a REPL loop, parse
     * incoming commands from query planner, and broadcast/relay those query
     * preparation/execution commands to secondary executors through the
     * NDPCommandProvider, which in turns, uses interfaces provided by
     * the ClusterManager.
     * */
    auto primaryProvider = std::make_shared<proteus::NDPCommandProvider>(ctx);
    ctx.getClusterManager().setCommandProvider(primaryProvider);

    while (ctx.getClusterManager().getNumExecutors() >= FLAGS_min_executors)
      ;

    LOG(INFO) << "Ready.";

    std::string line;
    while (std::getline(std::cin, line)) {
      std::string cmd = trim(line);

      LOG(INFO) << "Command received: " << cmd;

      if (cmd == "quit") {
        // Being primary node, disconnect will issue shutdown commands to
        // secondary nodes and shutdowns entire cluster gracefully.
        ctx.getClusterManager().disconnect();
        break;

      } else if (starts_with(cmd, "prepare plan ")) {
        if (starts_with(cmd, "prepare plan from file ")) {
          constexpr size_t prefix_size = clen("prepare plan from file ");
          std::string plan = cmd.substr(prefix_size);
          std::string label = primaryProvider->prepareStatement(fs::path{plan});
          std::cout << "prepared statement with label " << label << std::endl;

        } else {
          std::cout << "error (command not supported)" << std::endl;
        }
      } else if (starts_with(cmd, "execute plan ")) {
        if (starts_with(cmd, "execute plan from file ")) {
          constexpr size_t prefix_size = clen("execute plan from file ");
          std::string plan = cmd.substr(prefix_size);
          std::string path = primaryProvider->runStatement(plan, false);
          std::cout << "result in file " << path << std::endl;

        } else if (starts_with(cmd, "execute plan from statement ")) {
          constexpr size_t prefix_size = clen("execute plan from statement ");
          std::string plan = cmd.substr(prefix_size);
          std::string path = primaryProvider->runPreparedStatement(plan, false);
          std::cout << "result in file " << path << std::endl;

        } else {
          std::cout << "error (command not supported)" << std::endl;
        }
      }
    }

  } else {
    /*
     * Secondary nodes initializes the command provider and load it into
     * the cluster manager. From there on, secondary-nodes acts on events,
     * generated by the primary node. for each command by primary-node,
     * secondary node handler is called by the cluster-manager and processed
     * accordingly.
     *
     * The main thread waits indefinitely until the primary-node commands to
     * shutdown to all secondary nodes.
     * */

    // LocalCommandProvider provides functionalities to parse and execute plans,
    // regardless of operators inside the plan are single-server or
    // runs in a distributed fashion.
    // auto secondaryProvider = std::make_shared<LocalCommandProvider>();

    auto secondaryProvider =
        std::make_shared<proteus::MockLocalCommandProvider>(&ctx);

    ctx.getClusterManager().setCommandProvider(secondaryProvider);

    ctx.getClusterManager().waitUntilShutdown();
  }

  return 0;
}
