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

#include <sys/types.h>

#include <algorithm>
#include <olap/plan/query-result.hpp>
#include <utility>

#include "cluster-control.hpp"
#include "common/common.hpp"
#include "threadpool/thread.hpp"

namespace proteus::distributed {

void ClusterManager::connect(bool is_primary_node,
                             const std::string primary_node_addr,
                             int primary_control_port) {
  if (initialized) {
    LOG(ERROR) << "ClusterManager already initialized.";
  }

  this->is_primary = is_primary_node;
  this->terminate = false;

  if (!is_primary_node &&
      (primary_node_addr.empty() || primary_control_port <= 0)) {
    throw std::runtime_error("Invalid primary node server");
  }

  // start GRPC server.
  ClusterControl::getInstance().startServer(is_primary_node, primary_node_addr,
                                            primary_control_port);
  initialized = true;
}

void ClusterManager::disconnect() {
  if (!terminate) {
    this->terminate = true;
    ClusterControl::getInstance().shutdownServer();
  }
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
void ClusterManager::notifyFinished(std::string query_uuid,
                                    QueryResult result) {
  throw std::runtime_error("Unimplemented.");
}

const std::string &Query::getUUID() const { return query_uuid; }

struct decode_query {
  std::span<const std::byte> operator()(const std::string &s) {
    return std::span<const std::byte>{(const std::byte *)s.data(), s.size()};
  }
  std::span<const std::byte> operator()(const std::unique_ptr<mmap_file> &f) {
    return f->asSpan();
  }
};

std::span<const std::byte> Query::getQueryPlan() const {
  return std::visit(decode_query{}, query_plan);
}
}  // namespace proteus::distributed
