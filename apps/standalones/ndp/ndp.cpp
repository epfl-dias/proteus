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
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>

#include "ndp-cli-flags.hpp"

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("NDP", &argc, &argv);
  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  auto &clusterManager = proteus::distributed::ClusterManager::getInstance();

  clusterManager.connect(FLAGS_primary, FLAGS_url, FLAGS_port);

  sleep(2);

  if (FLAGS_primary) {
    while (clusterManager.getNumExecutors() < 1)
      ;
    LOG(INFO) << "Primary broadcasting query";
    clusterManager.broadcastQuery({"qq", "qq"});
  } else {
    LOG(INFO) << "Waiting for query now";
    auto q = clusterManager.getQuery();
    LOG(INFO) << "secondary got query with uuid: " << q.query_uuid;
  }

  sleep(2);

  clusterManager.disconnect();

  return 0;
}
