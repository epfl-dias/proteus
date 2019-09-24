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
#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>
#include <mutex>
#include <thread>

#include "infiniband-handler.hpp"

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid);
void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]);
void ib_connect(ibv_qp *qp, const ib_addr &rem, uint8_t ib_port, uint32_t psn,
                ibv_mtu mtu, uint8_t ib_sl, uint8_t sgid_idx);

int open_addr_exchange_connection(uint16_t port,
                                  const char *servername = nullptr);

static ib_addr get_server_addr(const std::string &server, ibv_qp *qp,
                               uint16_t port, const ib_addr &loc,
                               uint8_t ib_port, uint32_t psn, ibv_mtu mtu,
                               uint8_t ib_sl, uint8_t sgid_idx) {
  auto connfd = open_addr_exchange_connection(port, server.c_str());

  constexpr size_t gid_size = 32;
  constexpr size_t msg_size =
      sizeof("0000:000000:000000:") + gid_size;  // Includes \0
  char msg[msg_size];

  {
    char gid[gid_size + 1];
    gid_to_wire_gid(&loc.gid, gid);
    sprintf(msg, "%04" PRIx16 ":%06" PRIx32 ":%06" PRIx32 ":%s", loc.lid,
            loc.qpn, loc.psn, gid);
  }

  if (write(connfd, msg, msg_size) != msg_size) {
    close(connfd);
    auto msg = std::string{"Failed sending local IB address"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  {
    if (read(connfd, msg, msg_size) != msg_size ||
        write(connfd, "done", sizeof("done")) != sizeof("done")) {
      close(connfd);
      auto msg = std::string{"Failed reading remote IB address"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
  }

  close(connfd);

  ib_addr ret{};

  {
    char gid[gid_size + 1];
    sscanf(msg, "%" SCNx16 ":%" SCNx32 ":%" SCNx32 ":%s", &ret.lid, &ret.qpn,
           &ret.psn, gid);
    wire_gid_to_gid(gid, &ret.gid);
  }

  // Wait for above read before opening connection, to avoid race condition
  ib_connect(qp, ret, ib_port, psn, mtu, ib_sl, sgid_idx);

  return ret;
}

class IBHandlerClient : public IBHandler {
 public:
  IBHandlerClient(const std::string &url, uint16_t port, bool ipv4,
                  int cq_backlog = /* arbitrary */ 16)
      : IBHandler(cq_backlog) {
    rem_addr = get_server_addr(url, qp, port, addr, ib_port, addr.psn,
                               IBV_MTU_4096, ib_sl, ib_gidx);

    LOG(INFO) << "Remote IB address: " << rem_addr;
    active_connections.emplace(uint64_t{0}, nullptr);
  }
};

IBHandler *createClient(const std::string &url, uint16_t port, bool ipv4,
                        int cq_backlog = /* arbitrary */ 16) {
  return new IBHandlerClient(url, port, ipv4, cq_backlog);
}
