/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "infiniband-manager.hpp"

#include <arpa/inet.h>
#include <err.h>
#include <glog/logging.h>
#include <malloc.h>
#include <netdb.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>
#include <mutex>
#include <thread>

#include "infiniband-handler.hpp"

subscription::value_type subscription::wait() {
  std::unique_lock<std::mutex> lk{m};
  cv.wait(lk, [this] { return !q.empty(); });
  auto ret = q.front();
  q.pop();
  return ret;
}

void subscription::publish(void *data, size_t size) {
  std::unique_lock<std::mutex> lk{m};
  q.emplace(data, size);
  cv.notify_one();
}

IBHandler *InfiniBandManager::ib;

void InfiniBandManager::send(void *data, size_t bytes) {
  ib->send(data, bytes);
}

void InfiniBandManager::deinit() { delete ib; }

subscription &InfiniBandManager::subscribe() {
  return ib->register_subscriber();
}

void InfiniBandManager::unsubscribe(subscription &sub) {
  ib->unregister_subscriber(sub);
}

void InfiniBandManager::disconnectAll() { ib->disconnect(); }

IBHandler *createServer(uint16_t port, bool ipv4,
                        int cq_backlog = /* arbitrary */ 16);
IBHandler *createClient(const std::string &url, uint16_t port, bool ipv4,
                        int cq_backlog = /* arbitrary */ 16);

void InfiniBandManager::init(const std::string &url, uint16_t port,
                             bool primary, bool ipv4) {
  // int num_of_devices = 0;
  // ibv_context **devs = rdma_get_devices(&num_of_devices);
  // for (int i = 0; i < num_of_devices; ++i) {
  //   std::cout << "IB: " << devs[i]->device->dev_name << " "
  //             << devs[i]->device->dev_path << " " << devs[i]->device->name
  //             << " " << ibv_node_type_str(devs[i]->device->node_type)
  //             << std::endl;
  // }
  // rdma_free_devices(devs);
  if (primary) {
    ib = createServer(port, ipv4);
  } else {
    ib = createClient(url, port, ipv4);
  }

  ib->start();
}

#include <endian.h>
#include <linux/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid) {
  char tmp[9];
  int i;
  uint32_t tmp_gid[4];

  for (tmp[8] = 0, i = 0; i < 4; ++i) {
    memcpy(tmp, wgid + i * 8, 8);
    __be32 v32;
    sscanf(tmp, "%x", &v32);
    tmp_gid[i] = be32toh(v32);
  }
  memcpy(gid, tmp_gid, sizeof(*gid));
}

void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]) {
  uint32_t tmp_gid[4];
  int i;

  memcpy(tmp_gid, gid, sizeof(tmp_gid));
  for (i = 0; i < 4; ++i) sprintf(&wgid[i * 8], "%08x", htobe32(tmp_gid[i]));
}

void *InfiniBandManager::reg(void *mem, size_t bytes) {
  return ib->reg(mem, bytes);
}

void InfiniBandManager::unreg(void *mem) { ib->unreg(mem); }
