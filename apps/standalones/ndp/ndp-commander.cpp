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
#include "ndp-cli-flags.hpp"
#include "ndp/cluster_command_provider.hpp"
#include "ndp/ndp-common.hpp"

namespace proteus {

class NDPCommandProvider : public ClusterCommandProvider {};

class MockLocalCommandProvider : public LocalCommandProvider {
 private:
  ndp engine;
  std::map<std::string, PreparedStatement> stmts;

 public:
  void prepareStatement(const std::string &label,
                        const std::span<const std::byte> &plan) override {
    LOG(INFO) << "Mock prepare Query";
    auto builder = engine.getPlanParser().parse(plan, label);
    stmts.emplace(label, builder.prepare());
  }

  fs::path runPreparedStatement(const std::string &label, bool echo) override {
    auto result = engine.getExecutor().run(stmts.at(label));
    this->store(std::move(result));
    return fs::path("/dev/shm/" + label);
  }

  MockLocalCommandProvider() = default;
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

  // Primary-node act as an interface to OLAP engine.
  if (FLAGS_primary) {
    auto primaryProvider = std::make_shared<proteus::NDPCommandProvider>();
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
          std::string label = primaryProvider->runStatement(plan, false);

          //          std::cout << "result in file " << label << std::endl;
        } else if (starts_with(cmd, "execute plan from statement ")) {
          constexpr size_t prefix_size = clen("execute plan from statement ");
          std::string plan = cmd.substr(prefix_size);
          std::string label =
              primaryProvider->runPreparedStatement(plan, false);

          //          std::cout << "result in file " << label << std::endl;
        } else {
          std::cout << "error (command not supported)" << std::endl;
        }
      }
    }

  } else {
    // LocalCommandProvider provides functionalities to parse and execute plans,
    // regardless of operators inside the plan are single-server or
    // runs in a distributed fashion.
    // auto secondaryProvider = std::make_shared<LocalCommandProvider>();

    auto secondaryProvider =
        std::make_shared<proteus::MockLocalCommandProvider>();

    ctx.getClusterManager().setCommandProvider(secondaryProvider);

    ctx.getClusterManager().waitUntilShutdown();
  }

  return 0;
}
