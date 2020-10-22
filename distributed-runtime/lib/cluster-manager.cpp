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

#include "distributed-runtime/cluster-manager.hpp"

#include <arpa/inet.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <algorithm>
#include <thread>
#include <utility>

#include "cluster-control.hpp"
#include "clustercontrolplane.grpc.pb.h"
#include "common/common.hpp"
#include "common/error-handling.hpp"
#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "threadpool/thread.hpp"

namespace proteus::distributed {

// const std::string& server_addr,
//                                 int port,
//                                 bool is_primary_node,
//                                 const std::string& primary_node_addr,
//                                 int primary_control_port
void ClusterManager::connect(const std::string& self_server_addr,
                             int self_server_port, bool is_primary_node,
                             const std::string& primary_node_addr,
                             int primary_control_port) {
  if (initialized) {
    LOG(ERROR) << "ClusterManager already initialized.";
  }
  assert(self_server_port != -1 && "Port cannot be -1");

  self_server_address = self_server_addr;
  primary_server_address = primary_node_addr;

  this->is_primary = is_primary_node;
  this->terminate = false;

  if (!is_primary_node &&
      (primary_node_addr.empty() || primary_control_port <= 0)) {
    throw std::runtime_error("Invalid primary node server");
  }

  // start GRPC server.
  ClusterControl::getInstance().startServer(self_server_addr, self_server_port,
                                            is_primary_node, primary_node_addr,
                                            primary_control_port);
  initialized = true;
}

void ClusterManager::disconnect() {
  ClusterControl::getInstance().shutdownServer();
  this->terminate = true;
}

Query ClusterManager::getQuery() const {
  assert(initialized && "Cluster manager not connected.");
  return ClusterControl::getInstance().getQuery();
}
void ClusterManager::broadcastQuery(Query query) const {
  assert(initialized && "Cluster manager not connected.");
  return ClusterControl::getInstance().broadcastQuery(std::move(query));
}
size_t ClusterManager::getNumExecutors() {
  return ClusterControl::getInstance().getNumExecutors();
}

void ClusterManager::notifyReady(std::string query_uuid) {
  throw std::runtime_error("Unimplemented.");
}
void ClusterManager::notifyFinished(std::string query_uuid, void* result) {
  throw std::runtime_error("Unimplemented.");
}

}  // namespace proteus::distributed
