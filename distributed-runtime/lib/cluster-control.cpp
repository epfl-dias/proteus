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

#include <utility>

#include "common/common.hpp"
#include "common/error-handling.hpp"
#include "distributed-runtime/cluster-manager.hpp"

namespace proteus::distributed {

// FIXME: AS both primary/secondary runs a gRRPC server, there can be a
//  conflict of ip/port pair if both are running on the same server.

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
      cl.second.release();
      client_conn.erase(cl.first);
    }
    // Sanity-check. All client-connections should have been closed by now.
    assert(client_conn.empty());

  } else {
    this->primary_conn.release();
    this->server->Shutdown();
  }
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
    this->self_executor_id = registerReply.slave_id();
  } else {
    LOG(INFO) << "Secondary registration failed.";
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

  // Clean-up.
  query_queue.close();
  LOG(INFO) << "Listener-thread completed.";
}

Query ClusterControl::getQuery() {
  // Blocks until a new query is available.
  Query tmp;
  LOG(INFO) << "waiting for query to arrive.";
  auto status = this->query_queue.pop(tmp);
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
  client_conn.insert(
      {exec_id, NodeControlPlane::NewStub(grpc::CreateChannel(
                    target_addr, grpc::InsecureChannelCredentials()))});

  return exec_id;
}

//---------------------------------------
// gRPC Stub Handlers
//---------------------------------------

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
  // FIXME: need another thread to start shutdown, as RPC needs to reply
  //  also before shutting down.
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
