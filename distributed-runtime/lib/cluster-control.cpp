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

#include <utility>

#include "common/common.hpp"
#include "distributed-runtime/cluster-manager.hpp"

namespace proteus::distributed {

// FIXME: AS both primary/secondary runs a gRRPC server, there can be a
//  conflict of ip/port pair if both are running on the same server.

void ClusterControl::startServer(const std::string& self_server_addr,
                                 int self_server_port, bool is_primary_node,
                                 const std::string& primary_node_addr,
                                 int primary_control_port) {
  LOG(INFO) << "Starting Cluster Control Service.";
  assert(self_server_port > 0 && self_server_port < 65536);

  std::string full_server_add =
      self_server_addr + ":" + std::to_string(self_server_port);

  this->server_address = std::move(full_server_add);
  this->network_control_port = self_server_port;
  this->is_primary = is_primary_node;
  this->primary_node_address = std::string{primary_node_addr} + ":" +
                               std::to_string(primary_control_port);

  LOG(INFO) << "ServerAddres:: " << server_address;
  LOG(INFO) << "PrimaryAddres:: " << primary_node_address;

  // start-listener-thread
  std::thread t1(&ClusterControl::listenerThread, this);
  this->listener_thread.swap(t1);

  if (!is_primary_node) {
    assert(!primary_node_addr.empty());
    assert(primary_control_port > 0 && primary_control_port < 65536);

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
      this->self_executor_id = registerReply.slave_id();
    } else {
      LOG(INFO) << "Secondary registration failed.";
      throw std::runtime_error("Secondary registration failed.");
    }
  }
}
void ClusterControl::shutdownServer() {
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
  } else {
    primary_conn.release();
  }

  this->server->Shutdown();
  this->listener_thread.join();
  query_queue.close();
  LOG(INFO) << "Shutdown procedure completed.";
}

void ClusterControl::listenerThread() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(this->server_address,
                           grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&controlService);
  // Finally assemble the server.
  this->server = builder.BuildAndStart();

  LOG(INFO) << "Server listening on " << server_address << std::endl;
  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

Query ClusterControl::getQuery() {
  // Blocks until a new query is available.
  Query tmp;
  LOG(INFO) << "waiting for query to arrive.";
  auto status = query_queue.pop(tmp);
  if (!status) {
    LOG(INFO) << "Query_queue returned false";
  }
  return tmp;
}

void ClusterControl::broadcastQuery(Query query) {
  proteus::distributed::QueryPlan request;
  request.set_query_uuid(query.query_uuid);
  request.set_jsonplan(query.query_plan);

  for (auto& cl : this->client_conn) {
    LOG(INFO) << "Broadcasting query to executor # " << cl.first;
    // loaders.emplace_back([request, ex_id, cl]() {
    proteus::distributed::genericReply queryReply;
    grpc::ClientContext context;
    grpc::Status status =
        cl.second->prepareStatement(&context, request, &queryReply);

    if (status.ok()) {
      switch (queryReply.reply()) {
        case queryReply.ACK:
          LOG(INFO) << "prepareStatement: exec-" << cl.first << " ACK";
          break;
        case queryReply.ERROR:
          LOG(INFO) << "prepareStatement: exec-" << cl.first << " ERROR";
          break;
        default:
          LOG(INFO) << "Unknown reply";
          break;
      }
    } else {
      LOG(INFO) << "RPC Failed for exec- " << cl.first << ": "
                << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
    //});
  }
}

int ClusterControl::registerExecutor(
    const proteus::distributed::NodeInfo* node) {
  assert(is_primary && "secondary node getting executor register request??");

  std::unique_lock<std::mutex> safety_lock(this->registration_lock);
  this->executors.emplace_back(*node);
  auto exec_id = ++executor_id_ctr;
  exec_id_map.insert({exec_id, &(this->executors.back())});
  exec_address_map.insert(
      {std::string{node->control_address()}, &(this->executors.back())});

  LOG(INFO) << "Executor node with address " << node->control_address()
            << " registered with ID: " << exec_id;

  // address in gRPC is a fully-qualified one.
  std::string target_addr = std::string{node->control_address()};
  //+ ":" +std::to_string(network_control_port);
  client_conn.insert(
      {exec_id, NodeControlPlane::NewStub(grpc::CreateChannel(
                    target_addr, grpc::InsecureChannelCredentials()))});

  return exec_id;
}

grpc::Status NodeControlServiceImpl::registerExecutor(
    grpc::ServerContext* context, const proteus::distributed::NodeInfo* request,
    proteus::distributed::NodeRegistrationReply* reply) {
  auto& exec_address = request->control_address();
  auto executor_id = ClusterControl::getInstance().registerExecutor(request);
  reply->set_slave_id(executor_id);
  return grpc::Status::OK;
}

grpc::Status NodeControlServiceImpl::prepareStatement(
    grpc::ServerContext* context,
    const proteus::distributed::QueryPlan* request,
    proteus::distributed::genericReply* reply) {
  Query tmp;
  tmp.query_plan = request->jsonplan();
  tmp.query_uuid = request->query_uuid();
  ClusterControl::getInstance().query_queue.push(tmp);
  LOG(INFO) << "Received query in the queue";
  reply->set_reply(proteus::distributed::genericReply::ACK);
  return grpc::Status::OK;
}
grpc::Status NodeControlServiceImpl::sendCommand(
    grpc::ServerContext* context,
    const proteus::distributed::NodeCommand* request,
    proteus::distributed::NodeStatusUpdate* reply) {
  ClusterManager::getInstance().disconnect();
  reply->set_status(proteus::distributed::NodeStatusUpdate::SHUTDOWN);
  return grpc::Status::OK;
}

grpc::Status NodeControlServiceImpl::executeStatement(
    grpc::ServerContext* context,
    const proteus::distributed::QueryPlan* request,
    proteus::distributed::genericReply* reply) {
  throw std::runtime_error("Unimplemented");

  return grpc::Status::OK;
}

}  // namespace proteus::distributed
