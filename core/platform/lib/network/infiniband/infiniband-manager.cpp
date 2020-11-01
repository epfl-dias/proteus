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
#include <mutex>
#include <platform/common/error-handling.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/network/infiniband/devices/ib.hpp>
#include <platform/network/infiniband/infiniband-handler.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <thread>

subscription::value_type subscription::wait() {
  std::unique_lock<std::mutex> lk{m};
  cv.wait(lk, [this] { return !q.empty(); });
  auto ret = std::move(q.front());
  q.pop();
  return ret;
}

void subscription::publish(proteus::managed_ptr data, size_t size
#ifndef NDEBUG
                           ,
                           decltype(__builtin_FILE()) file,
                           decltype(__builtin_LINE()) line
#endif
) {
  std::unique_lock<std::mutex> lk{m};
  q.emplace(std::move(data), size
#ifndef NDEBUG
            ,
            file, line
#endif
  );
  cv.notify_one();
}

std::vector<IBHandler *> InfiniBandManager::ib;
uint64_t InfiniBandManager::srv_id;

uint64_t InfiniBandManager::server_id() { return srv_id; }

void InfiniBandManager::send(void *data, size_t bytes) {
  assert(!ib.empty());
  ib.front()->send(data, bytes);  // FIXME
}

void InfiniBandManager::write(proteus::managed_ptr data, size_t bytes,
                              size_t sub_id) {
  ib.front()->write(std::move(data), bytes, sub_id);  // FIXME
}

void InfiniBandManager::write_to(proteus::managed_ptr data, size_t bytes,
                                 buffkey b) {
  ib.front()->write_to(std::move(data), bytes, b);  // FIXME
}

subscription *InfiniBandManager::write_silent(proteus::managed_ptr data,
                                              size_t bytes) {
  return ib.front()->write_silent(std::move(data), bytes);  // FIXME
}

subscription *InfiniBandManager::read(proteus::remote_managed_ptr data,
                                      size_t bytes) {
  return ib.front()->read(std::move(data), bytes);  // FIXME
}

subscription *InfiniBandManager::read_event() {
  return ib.front()->read_event();  // FIXME
}

void InfiniBandManager::flush() {
  ib.front()->flush();  // FIXME
}
void InfiniBandManager::flush_read() {
  ib.front()->flush_read();  // FIXME
}

buffkey InfiniBandManager::get_buffer() {
  return ib.front()->get_buffer();
  // FIXME
}

void InfiniBandManager::release_buffer(proteus::remote_managed_ptr p) {
  return ib.front()->release_buffer(std::move(p));
  // FIXME
}

void InfiniBandManager::deinit() {
  for (auto &i : ib) delete i;
}

subscription &InfiniBandManager::subscribe() {
  return ib.front()->register_subscriber();  // FIXME
}

subscription &InfiniBandManager::create_subscription() {
  return ib.front()->create_subscription();  // FIXME
}

void InfiniBandManager::unsubscribe(subscription &sub) {
  ib.front()->unregister_subscriber(sub);  // FIXME
}

void InfiniBandManager::disconnectAll() {
  for (auto &i : ib) i->disconnect();
  //  ib.front()->disconnect();  // FIXME
}

IBHandler *createServer(uint16_t port, bool ipv4,
                        int cq_backlog = /* arbitrary */ cq_ack_backlog);
IBHandler *createClient(size_t i, const std::string &url, uint16_t port,
                        bool ipv4,
                        int cq_backlog = /* arbitrary */ cq_ack_backlog);

void InfiniBandManager::init(const std::string &url, uint16_t port,
                             bool primary, bool ipv4) {
  for (const auto &ib : topology::getInstance().getIBs()) {
    BlockManager::reg(ib.getMemRegistry());
  }
  for (const auto &ib : topology::getInstance().getIBs()) {
    ib.ensure_listening();
  }

  if (primary) {
    srv_id = 0;
  } else {
    srv_id = 1;
  }
  if (topology::getInstance().getIBCount()) {
    auto addr = [&]() {
      if (primary) {
        return topology::getInstance().getIBs().front().get_client(port);
      } else {
        return topology::getInstance().getIBs().front().get_server(url, port);
      }
    }();
    LOG(INFO) << addr;

    ib.emplace_back(new IBHandler(8, topology::getInstance().getIBs().front()));
    ib.front()->start();
  }
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

buffkey InfiniBandManager::reg(const void *mem, size_t bytes) {
  if (!ib.empty()) return ib.front()->reg2(mem, bytes);
  return {nullptr, 0};
}

void InfiniBandManager::unreg(const void *mem) {
  if (!ib.empty()) ib.front()->unreg(mem);
}

subscription::value_type::~value_type() {
  if (line == 441) LOG(INFO) << file << " " << line;
  assert(data == nullptr);
  BlockManager::release_buffer(data.release());
}
