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

#ifndef PROTEUS_CLUSTER_MANAGER_HPP
#define PROTEUS_CLUSTER_MANAGER_HPP

#include <iostream>
#include <map>
#include <thread>
#include <vector>

namespace proteus::distributed {

struct Query {
  std::string query_uuid;
  std::string query_plan;
};

class ClusterManager {
 public:
  static inline ClusterManager &getInstance() {
    static ClusterManager instance;
    return instance;
  }
  ClusterManager(ClusterManager const &) = delete;
  void operator=(ClusterManager const &) = delete;
  void operator=(ClusterManager const &&) = delete;

  void connect(const std::string &self_server_addr, int self_server_port,
               bool is_primary_node, const std::string &primary_node_addr,
               int primary_control_port);
  void disconnect();

  Query getQuery() const;
  void broadcastQuery(Query query) const;
  void notifyReady(std::string query_uuid);
  void notifyFinished(std::string query_uuid, void *results = nullptr);

  size_t getNumExecutors();

 private:
  ClusterManager() : initialized(false), terminate(false), is_primary(false) {}

 private:
  bool terminate;
  bool initialized;
  bool is_primary;

  std::string self_server_address;
  std::string primary_server_address;
};

}  // namespace proteus::distributed

#endif  // PROTEUS_CLUSTER_MANAGER_HPP
