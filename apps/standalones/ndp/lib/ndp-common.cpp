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

#include <gflags/gflags.h>

#include <cli-flags.hpp>
#include <ndp/ndp-common.hpp>
#include <olap/common/olap-common.hpp>

#include "mock-plan-parser.hpp"

namespace proteus {

class ndp::impl {
 private:
  proteus::olap olap;
  MockPlanParser parser;
  CommandExecutor executor;

 public:
  PlanParser &getPlanParser() { return parser; }
  CommandExecutor &getExecutor() { return executor; }

  // Load the execution lib
  void loadExecutionLibraries() {}
  // Load the lib*.so with policies
  void loadRoutingPolicies() {}
  // Load the lib*.so with load balancing policies
  void loadLoadBalancingPolicies() {}
  // Detect local files and configurations
  void discoverLocalCatalog() {}

  impl(const string &usage, int *argc, char ***argv)
      : olap(proteus::from_cli::olap(usage, argc, argv)) {
    /*
     * For now, all libraries are pre-linked, but the following function can be
     * extended to load and configure the different parts so that the context
     * returns libraries that are loaded dynamically.
     */

    // Load the execution lib
    loadExecutionLibraries();

    // Load the lib*.so with policies
    loadRoutingPolicies();

    // Load the lib*.so with load balancing policies
    loadLoadBalancingPolicies();

    // Detect local files and configurations
    discoverLocalCatalog();
  }
};

PlanParser &proteus::ndp::getPlanParser() { return p_impl->getPlanParser(); }
proteus::distributed::ClusterManager &ndp::getClusterManager() const {
  return proteus::distributed::ClusterManager::getInstance();
}
ThreadPool &ndp::getThreadPool() { return ThreadPool::getInstance(); }
CommandExecutor &ndp::getExecutor() { return p_impl->getExecutor(); }

ndp::ndp(const string &usage, int *argc, char ***argv)
    : p_impl(std::make_unique<ndp::impl>(usage, argc, argv)){};
ndp::~ndp() = default;

proteus::ndp from_cli::ndp(const string &usage, int *argc, char ***argv) {
  //  gflags::SetUsageMessage(usage);
  //  gflags::ParseCommandLineFlags(argc, argv, true);
  //
  //  google::InitGoogleLogging((*argv)[0]);
  //  FLAGS_logtostderr = true;

  return proteus::ndp{usage, argc, argv};
}
}  // namespace proteus
