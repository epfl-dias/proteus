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

#include "cluster-control.hpp"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <unistd.h>

#include <platform/common/common.hpp>
#include <platform/common/error-handling.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <utility>

#include "distributed-runtime/cluster-manager.hpp"

namespace proteus::distributed {

void ClusterControl::wait() {
  this->listener_thread.join();
  return;
}

void ClusterControl::startServer(bool is_primary_node,
                                 const std::string& primary_node_addr,
                                 int primary_control_port) {
  LOG(INFO) << "Starting Cluster Control Service.";

  // Sanity-check for primary-node address and port.
  assert(!primary_node_addr.empty());
  assert(primary_control_port > 0 && primary_control_port < 65536);

  // Form a socket-style address by concatenating listening port with address.
  this->is_primary = is_primary_node;
  this->primary_node_address = std::string{primary_node_addr} + ":" +
                               std::to_string(primary_control_port);

  if (is_primary_node) {
    this->server_address = primary_node_address;
    LOG(INFO) << "Primary Node:: " << is_primary_node;
    LOG(INFO) << "ServerAddress:: " << primary_node_address;
  }

  // start-listener-thread
  std::thread t1(&ClusterControl::listenerThread, this);
  this->listener_thread.swap(t1);
}
void ClusterControl::shutdownServer(bool rpc_initiated) {
  LOG(INFO) << "Shutting down Cluster Control Service.";

  if (is_primary) {
    // Notify all secondary to shutdown.

    proteus::distributed::NodeCommand request;
    request.set_command(proteus::distributed::NodeCommand::SHUTDOWN);

    for (auto& cl : this->client_conn) {
      LOG(INFO) << "Command:Shutdown to executor # " << cl.first;
      // loaders.emplace_back([request, ex_id, cl]() {
      proteus::distributed::NodeStatusUpdate queryReply;
      grpc::ClientContext context;
      grpc::Status status =
          cl.second->sendCommand(&context, request, &queryReply);

      if (status.ok()) {
        LOG(INFO) << "Exec-" << cl.first << " replied for shutting down.";
      } else {
        LOG(INFO) << "RPC Failed for exec- " << cl.first << ": "
                  << status.error_code() << ": " << status.error_message()
                  << std::endl;
      }
    }
    client_conn.clear();
  }

  if (!is_primary && !rpc_initiated) {
    // Notify primary that this-executor is shutting down.
    proteus::distributed::NodeStatusUpdate request;
    request.set_executor_id(this->self_executor_id);
    request.set_status(proteus::distributed::NodeStatusUpdate::SHUTDOWN);
    proteus::distributed::genericReply reply;

    grpc::ClientContext context;
    grpc::Status status =
        this->primary_conn->changeNodeStatus(&context, request, &reply);

    if (!status.ok()) {
      LOG(INFO) << "RPC Failed for exec- " << this->self_executor_id << ": "
                << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

  this->server->Shutdown();
  LOG(INFO) << "Shutdown procedure completed.";
}

void ClusterControl::registerSelfToPrimary() {
  // register itself to primary node.
  this->primary_conn = NodeControlPlane::NewStub(grpc::CreateChannel(
      primary_node_address, grpc::InsecureChannelCredentials()));

  proteus::distributed::NodeInfo registerRequest;
  registerRequest.set_control_address(server_address);
  proteus::distributed::NodeRegistrationReply registerReply;

  grpc::ClientContext context;
  grpc::Status status = this->primary_conn->registerExecutor(
      &context, registerRequest, &registerReply);

  if (status.ok()) {
    this->self_executor_id = registerReply.executor_id();
  } else {
    LOG(ERROR) << "Secondary registration failed.";
    throw std::runtime_error("Secondary registration failed.");
  }
}

void ClusterControl::listenerThread() {
  std::string secondary_hostname;

  if (!is_primary) {
    // if not primary node, get the hostname from system.

    char hostname_buffer[512];
    linux_run(gethostname(hostname_buffer, 512));
    secondary_hostname = std::string{hostname_buffer};
    this->server_address = secondary_hostname + ":" + std::to_string(0);
    LOG(INFO) << "Got Hostname: " << this->server_address;
  }

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  // if the port is 0, system will select an available port automatically,
  // and update the `sec_port` variable.

  int sec_port = 0;
  builder.AddListeningPort(this->server_address,
                           grpc::InsecureServerCredentials(), &sec_port);

  // Register "service" as the instance through which we'll communicate with
  // clients and finally, assemble and start the server.
  builder.RegisterService(&controlService);
  this->server = builder.BuildAndStart();

  // If not a primary-node, register itself to primary as a executor node.
  if (!is_primary) {
    this->server_address = secondary_hostname + ":" + std::to_string(sec_port);
    this->registerSelfToPrimary();
  }

  LOG(INFO) << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();

  LOG(INFO) << "Listener-thread completed.";
}

void ClusterControl::broadcastPrepareQuery(Query query) {
  proteus::distributed::QueryPlan request;
  request.set_query_uuid(query.getUUID());
  auto s = query.getQueryPlan();
  request.set_jsonplan({(const char*)s.data(), s.size_bytes()});

  LOG(INFO) << "clearing previous query statuses";
  this->_clearQueryStatus(query.getUUID());

  for (auto& cl : this->client_conn) {
    LOG(INFO) << "Broadcasting query to executor # " << cl.first;

    proteus::distributed::genericReply queryReply;
    grpc::ClientContext context;
    grpc::Status status =
        cl.second->prepareStatement(&context, request, &queryReply);

    if (status.ok()) {
      switch (queryReply.reply()) {
        case queryReply.ACK:
          LOG(INFO) << "broadcastPrepareQuery: executor#" << cl.first << " ACK";
          break;
        case queryReply.ERROR:
          LOG(INFO) << "broadcastPrepareQuery: executor#" << cl.first
                    << " ERROR";
          break;
        default:
          LOG(INFO) << "Unknown reply";
          break;
      }
    } else {
      LOG(INFO) << "RPC Failed for executor# " << cl.first << ": "
                << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }
}
void ClusterControl::broadcastExecuteQuery(std::string query_uuid) {
  proteus::distributed::QueryPlan request;
  request.set_query_uuid(query_uuid);

  LOG(INFO) << "clearing previous query statuses";
  this->_clearQueryStatus(query_uuid);

  for (auto& cl : this->client_conn) {
    LOG(INFO) << "Broadcasting query to executor # " << cl.first;

    proteus::distributed::genericReply queryReply;
    grpc::ClientContext context;
    grpc::Status status =
        cl.second->executeStatement(&context, request, &queryReply);

    if (status.ok()) {
      switch (queryReply.reply()) {
        case queryReply.ACK:
          LOG(INFO) << "broadcastExecuteQuery: executor#" << cl.first << " ACK";
          break;
        case queryReply.ERROR:
          LOG(INFO) << "broadcastExecuteQuery: executor#" << cl.first
                    << " ERROR";
          break;
        default:
          LOG(INFO) << "Unknown reply";
          break;
      }
    } else {
      LOG(INFO) << "RPC Failed for executor# " << cl.first << ": "
                << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }
}
void ClusterControl::broadcastPrepareExecuteQuery(Query query) {
  proteus::distributed::QueryPlan request;
  request.set_query_uuid(query.getUUID());
  auto s = query.getQueryPlan();
  request.set_jsonplan({(const char*)s.data(), s.size_bytes()});

  LOG(INFO) << "clearing previous query statuses";
  this->_clearQueryStatus(query.getUUID());

  for (auto& cl : this->client_conn) {
    LOG(INFO) << "Broadcasting query to executor # " << cl.first;

    proteus::distributed::genericReply queryReply;
    grpc::ClientContext context;
    grpc::Status status =
        cl.second->executeStatement(&context, request, &queryReply);

    if (status.ok()) {
      switch (queryReply.reply()) {
        case queryReply.ACK:
          LOG(INFO) << "broadcastPrepareExecuteQuery: executor#" << cl.first
                    << " ACK";
          break;
        case queryReply.ERROR:
          LOG(INFO) << "broadcastPrepareExecuteQuery: executor#" << cl.first
                    << " ERROR";
          break;
        default:
          LOG(INFO) << "Unknown reply";
          break;
      }
    } else {
      LOG(INFO) << "RPC Failed for executor#" << cl.first << ": "
                << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }
}

int ClusterControl::registerExecutor(
    const proteus::distributed::NodeInfo* node) {
  assert(is_primary && "secondary node getting executor register request??");

  std::unique_lock<std::mutex> safety_lock(this->registration_lock);
  this->executors.emplace_back(*node);
  auto exec_id = ++executor_id_ctr;
  this->executors.back().set_executor_id(exec_id);
  exec_id_map.insert({exec_id, &(this->executors.back())});
  exec_address_map.insert(
      {std::string{node->control_address()}, &(this->executors.back())});

  LOG(INFO) << "Executor node with address " << node->control_address()
            << " registered with ID: " << exec_id;

  // address in gRPC is a fully-qualified one.
  std::string target_addr = std::string{node->control_address()};
  client_conn.insert(
      {exec_id, NodeControlPlane::NewStub(grpc::CreateChannel(
                    target_addr, grpc::InsecureChannelCredentials()))});

  return exec_id;
}

int32_t ClusterControl::getResultServerId() {
  auto eid = exec_address_map[this->primary_node_address]->executor_id();
  LOG(INFO) << "Result server-id: " << eid;
  return eid;
}

void ClusterControl::updateNodeStatus(
    const proteus::distributed::NodeStatusUpdate* request) {
  assert(is_primary && "secondary node getting executor update request??");

  std::unique_lock<std::mutex> safety_lock(this->registration_lock);

  auto exec_id = request->executor_id();
  // remove from client_conn
  client_conn.erase(exec_id);

  // remove from exec_address_map
  exec_address_map.erase(exec_id_map[exec_id]->control_address());

  // remove from executors
  executors.erase(
      std::remove_if(executors.begin(), executors.end(),
                     [exec_id](auto x) { return x.executor_id() == exec_id; }),
      executors.end());

  // remove from exec_id_map
  exec_id_map.erase(exec_id);

  LOG(INFO) << "Removed executor-" << exec_id << ".";
}
void ClusterControl::notifyQueryStatus(
    proteus::distributed::QueryNotification&& notification) {
  if (notification.status() ==
      proteus::distributed::QueryNotification::EXECUTED) {
    notification.set_result_server_address(this->primary_node_address);
  }

  notification.set_sender_executor_id(this->self_executor_id);
  notification.set_timestamp(
      std::chrono::system_clock::now().time_since_epoch().count());
  proteus::distributed::genericReply queryReply;
  grpc::ClientContext context;
  grpc::Status status =
      this->primary_conn->QueryStatus(&context, notification, &queryReply);

  if (!status.ok()) {
    LOG(INFO) << "RPC Failed: " << status.error_code() << ": "
              << status.error_message();
  }
}

void ClusterControl::_updateQueryStatus(
    const proteus::distributed::QueryNotification& notification) {
  switch (notification.status()) {
    case (QueryNotification::Status::QueryNotification_Status_PREPARED):
      LOG(INFO) << "Query[" << notification.query_uuid()
                << "][Executor:" << notification.sender_executor_id()
                << "] status update: PREPARED";
      break;
    case (QueryNotification::Status::QueryNotification_Status_EXECUTED):
      LOG(INFO) << "Query[" << notification.query_uuid()
                << "][Executor:" << notification.sender_executor_id()
                << "] status update: EXECUTED";
      break;
    case (QueryNotification::Status::QueryNotification_Status_ERROR):
      LOG(INFO) << "Query[" << notification.query_uuid()
                << "][Executor:" << notification.sender_executor_id()
                << "] status update: ERROR: " << notification.error_message();
      break;
    default:
      LOG(INFO) << "Unknown query status update for query_uuid: "
                << notification.query_uuid()
                << " | status: " << notification.status();
      break;
  }
  {
    std::unique_lock<std::mutex> safety_lock(this->query_status_lock);
    if (query_status_map.contains(notification.query_uuid()) == true) {
      this->query_status_map.at(notification.query_uuid())
          .push_back(notification);
    } else {
      this->query_status_map.emplace(
          notification.query_uuid(),
          std::vector<QueryNotification>{notification});
    }
  }
}

//---------------------------------------
// gRPC Stub Handlers
//---------------------------------------

grpc::Status NodeControlServiceImpl::registerExecutor(
    grpc::ServerContext* context, const proteus::distributed::NodeInfo* request,
    proteus::distributed::NodeRegistrationReply* reply) {
  auto& exec_address = request->control_address();
  auto executor_id = ClusterControl::getInstance().registerExecutor(request);
  reply->set_executor_id(executor_id);
  return grpc::Status::OK;
}

grpc::Status NodeControlServiceImpl::prepareStatement(
    grpc::ServerContext* context,
    const proteus::distributed::QueryPlan* request,
    proteus::distributed::genericReply* reply) {
  LOG(INFO) << "Received query in the queue, Now preparing.";

  Query tmp{request->query_uuid(), request->jsonplan()};

  ThreadPool::getInstance().enqueue([tmp = std::move(tmp)]() {
    proteus::distributed::QueryNotification queryStatus;
    queryStatus.set_query_uuid(tmp.getUUID());

    LOG(INFO) << "Preparing query: " << tmp.getUUID();
    try {
      ClusterManager::getInstance().getCommandProvider()->prepareStatement(
          tmp.getUUID(), tmp.getQueryPlan());
      // ClusterControl::getInstance().query_queue.push(std::move(tmp));
      queryStatus.set_status(proteus::distributed::QueryNotification::PREPARED);
    } catch (std::exception& e) {
      queryStatus.set_status(proteus::distributed::QueryNotification::ERROR);
      queryStatus.set_error_message(e.what());
    }
    ClusterControl::getInstance().notifyQueryStatus(std::move(queryStatus));
  });

  reply->set_reply(proteus::distributed::genericReply::ACK);

  return grpc::Status::OK;
}
grpc::Status NodeControlServiceImpl::sendCommand(
    grpc::ServerContext* context,
    const proteus::distributed::NodeCommand* request,
    proteus::distributed::NodeStatusUpdate* reply) {
  if (request->command() == proteus::distributed::NodeCommand::SHUTDOWN) {
    ThreadPool::getInstance().enqueue([]() {
      // ClusterManager::getInstance().disconnect();
      ClusterControl::getInstance().shutdownServer(true);
    });

    reply->set_status(proteus::distributed::NodeStatusUpdate::SHUTDOWN);
    return grpc::Status::OK;
  } else {
    throw std::runtime_error("Unknown command received.");
  }
}

grpc::Status NodeControlServiceImpl::changeNodeStatus(
    grpc::ServerContext* context,
    const proteus::distributed::NodeStatusUpdate* request,
    proteus::distributed::genericReply* reply) {
  ClusterControl::getInstance().updateNodeStatus(request);
  reply->set_reply(proteus::distributed::genericReply::ACK);
  return grpc::Status::OK;
}

grpc::Status NodeControlServiceImpl::executeStatement(
    grpc::ServerContext* context,
    const proteus::distributed::QueryPlan* request,
    proteus::distributed::genericReply* reply) {
  Query tmp{request->query_uuid(), request->jsonplan()};

  ThreadPool::getInstance().enqueue([tmp = std::move(tmp)]() {
    proteus::distributed::QueryNotification queryStatus;
    queryStatus.set_query_uuid(tmp.getUUID());
    try {
      LOG(INFO) << "Executing query: " << tmp.getUUID();
      auto resultPath = ClusterManager::getInstance()
                            .getCommandProvider()
                            ->runPreparedStatement(tmp.getUUID());

      queryStatus.set_result_location(resultPath);
      queryStatus.set_status(proteus::distributed::QueryNotification::EXECUTED);

    } catch (proteus::unprepared_plan_execution& exception) {
      if (tmp.getQueryPlan().empty()) {
        queryStatus.set_status(proteus::distributed::QueryNotification::ERROR);
        queryStatus.set_error_message(
            "Execute called without preparation/query-plan :" +
            std::string{exception.what()});
        LOG(ERROR) << "Execute called without preparation/query-plan";
      }

      LOG(INFO) << "Preparing & Executing query: " << tmp.getUUID();
      ClusterManager::getInstance().getCommandProvider()->prepareStatement(
          tmp.getUUID(), tmp.getQueryPlan());
      auto resultPath = ClusterManager::getInstance()
                            .getCommandProvider()
                            ->runPreparedStatement(tmp.getUUID());
      queryStatus.set_result_location(resultPath);
      queryStatus.set_status(proteus::distributed::QueryNotification::EXECUTED);
    }
    ClusterControl::getInstance().notifyQueryStatus(std::move(queryStatus));
  });

  return grpc::Status::OK;
}

grpc::Status NodeControlServiceImpl::QueryStatus(
    grpc::ServerContext* context,
    const proteus::distributed::QueryNotification* request,
    proteus::distributed::genericReply* reply) {
  ClusterControl::getInstance()._updateQueryStatus({*request});

  reply->set_reply(genericReply::ACK);
  return grpc::Status::OK;
}

}  // namespace proteus::distributed
