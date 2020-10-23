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
#include <distributed-runtime/cluster-manager.hpp>
#include <iostream>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>

#include "ndp-cli-flags.hpp"

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
  //  auto ctx = proteus::from_cli::olap("NDP", &argc, &argv);
  //  set_exec_location_on_scope
  //  exec(topology::getInstance().getCpuNumaNodes()[0]);
  //
  //  auto &clusterManager =
  //  proteus::distributed::ClusterManager::getInstance();
  //  clusterManager.connect(FLAGS_primary, FLAGS_url, FLAGS_port);
  //
  //  bool echo = false;
  //
  //  // ------- Wait for atleast X # of executors join.
  //  LOG(INFO) << "Waiting for at least " << FLAGS_min_executors
  //            << " to register..";
  //  if (FLAGS_primary) {
  //    while (clusterManager.getNumExecutors() >= FLAGS_min_executors)
  //      ;
  //
  //    LOG(INFO) << "Starting primary REPL loop";
  //
  //    std::string line;
  //    while (std::getline(std::cin, line)) {
  //      std::string cmd = trim(line);
  //
  //      LOG(INFO) << "Command received: " << cmd;
  //
  //      if (cmd == "quit") {
  //        std::cout << "quiting..." << std::endl;
  //        break;
  //      } else if (starts_with(cmd, "prepare plan ")) {
  //        if (starts_with(cmd, "prepare plan from file ")) {
  //          constexpr size_t prefix_size = clen("prepare plan from file ");
  //          std::string plan = cmd.substr(prefix_size);
  //          std::string label = provider.prepareStatement(fs::path{plan});
  //
  //          std::cout << "prepared statement with label " << label <<
  //          std::endl;
  //        } else {
  //          std::cout << "error (command not supported)" << std::endl;
  //        }
  //      } else if (starts_with(cmd, "execute plan ")) {
  //        if (starts_with(cmd, "execute plan from file ")) {
  //          constexpr size_t prefix_size = clen("execute plan from file ");
  //          std::string plan = cmd.substr(prefix_size);
  //          std::string label = provider.runStatement(plan, echo);
  //
  //          std::cout << "result in file " << label << std::endl;
  //        } else if (starts_with(cmd, "execute plan from statement ")) {
  //          constexpr size_t prefix_size = clen("execute plan from statement
  //          "); std::string plan = cmd.substr(prefix_size); std::string label
  //          = provider.runPreparedStatement(plan, echo);
  //
  //          std::cout << "result in file " << label << std::endl;
  //        } else {
  //          std::cout << "error (command not supported)" << std::endl;
  //        }
  //      } else if (starts_with(cmd, "echo")) {
  //        if (cmd == "echo results on") {
  //          echo = true;
  //        } else if (cmd == "echo results off") {
  //          echo = false;
  //        } else {
  //          std::cout << "error (unknown echo, please specify what to echo)"
  //                    << std::endl;
  //        }
  //      }
  //      //      else if (cmd == "unloadall") {
  //      //        StorageManager::getInstance().unloadAll();
  //      //        std::cout << "done" << std::endl;
  //      //      }
  //    }
  //
  //  } else {
  //    // Secondary-node
  //    // Currently, the engine-runs query-at-a-time only.
  //
  //    LOG(INFO) << "Waiting for query";
  //    auto query = clusterManager.getQuery();
  //
  //    LOG(INFO) << "Received query with UUID: " << query.getUUID();
  //
  //    // prepare-query
  //    provider.prepareStatement(query.getQueryPlan());
  //
  //    // execute-query
  //  }
  //
  //  // shutdown-procedure
  //
  //  if (FLAGS_primary) {
  //    while (clusterManager.getNumExecutors() < 1)
  //      ;
  //    LOG(INFO) << "Primary broadcasting query";
  //    clusterManager.broadcastQuery({"qq", "qq"});
  //  } else {
  //    LOG(INFO) << "Waiting for query now";
  //    auto q = clusterManager.getQuery();
  //    LOG(INFO) << "secondary got query with uuid: " << q.getUUID();
  //  }
  //
  //  clusterManager.disconnect();

  return 0;
}
