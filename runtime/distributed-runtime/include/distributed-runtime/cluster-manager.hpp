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

/*
 * CusterManager.hpp
 *
 * struct Query
 *    Provides a container for query plan, which can be communicated across
 *    executor processes.
 *
 *    Attributes:
 *    - `query_uuid`: Unique identifier for a query across cluster-runtime.
 *    - `query_plan`: Query plan in string which can be parsed, prepared and
 *          executed across executor nodes.
 *
 *
 *  ClusterManager
 *      Cluster Manager provides functionalities for inter-cluster
 *      communication protocol and provides following services. It provides
 *      control-plane layer for cluster-management, communication, and
 *      executor management.
 *
 *     - Node Discovery and Registration: primary-node starts a listening server
 *        to which all secondary nodes can communicate with. Secondary-nodes,
 *        registers with the primary-node, notifying presence of additional
 *        executor node and the fully-qualified-communication listener address.
 *     - Get/Broadcast Query: sending/receiving query plans across execution
 *        nodes.
 *     - Notify Ready/Finished: Distributed condition variable, intended to be
 *        used for control synchronization.
 *
 *      Implementation-specific: Underlying ClusterManager, proteus uses a
 *        gRPC server on each node. all nodes have a bi-directional link with
 *        the primary node. The communication primitives and protocol is defined
 *        in the `clustercontrolplan.proto` file.
 */

#include <command-provider/command-provider.hpp>
#include <iostream>
#include <map>
#include <olap/plan/query-result.hpp>
#include <span>
#include <storage/mmap-file.hpp>
#include <thread>
#include <variant>
#include <vector>

namespace proteus::distributed {

class Query {
 private:
  std::string query_uuid;
  std::variant<std::unique_ptr<mmap_file>, std::string> query_plan;

 public:
  Query() = default;
  Query(std::string query_uuid, std::unique_ptr<mmap_file> query_plan)
      : query_uuid(std::move(query_uuid)), query_plan(std::move(query_plan)) {}
  Query(std::string query_uuid, std::string query_plan)
      : query_uuid(std::move(query_uuid)), query_plan(std::move(query_plan)) {}

 public:
  /*
   * @returns plan blob with the lifetime as the Query object itself
   */
  [[nodiscard]] std::span<const std::byte> getQueryPlan() const;
  /*
   * @returns plan's cluster-wide unique identifier.
   */
  [[nodiscard]] const std::string &getUUID() const;
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

  virtual void connect(bool is_primary_node, std::string primary_node_addr,
                       int primary_control_port);
  virtual void disconnect();

  void setCommandProvider(std::shared_ptr<CommandProvider> provider) {
    cmdProvider = provider;
  }

  auto getCommandProvider() { return cmdProvider; }

  virtual void broadcastPrepareQuery(Query query);
  virtual void broadcastExecuteQuery(std::string query_uuid);
  virtual void broadcastPrepareExecuteQuery(Query query);

  void notifyReady(std::string query_uuid);
  void notifyFinished(std::string query_uuid, QueryResult result);

  virtual size_t getNumExecutors();
  virtual int32_t getResultServerId();
  int32_t getLocalServerId();
  auto getQueryStatus(std::string query_uuid);

  virtual void waitUntilShutdown();

  virtual ~ClusterManager() {
    if (!terminate) disconnect();
  }

 protected:
  ClusterManager() : initialized(false), terminate(false), is_primary(false) {}

 private:
  bool terminate;
  bool initialized;
  bool is_primary;
  std::shared_ptr<CommandProvider> cmdProvider;
};

}  // namespace proteus::distributed

#endif  // PROTEUS_CLUSTER_MANAGER_HPP
