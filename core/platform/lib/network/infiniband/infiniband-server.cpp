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

#include <glog/logging.h>
#include <sys/socket.h>

#include <iostream>
#include <platform/common/error-handling.hpp>
#include <platform/network/infiniband/infiniband-handler.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>

int startpingpong(bool isserver);

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid);
void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]);

void ib_connect(ibv_qp *qp, const ib_addr &rem, uint8_t ib_port, uint32_t psn,
                ibv_mtu mtu, uint8_t ib_sl, uint8_t sgid_idx) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = mtu;
  attr.dest_qp_num = rem.qpn;
  attr.rq_psn = rem.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = rem.lid;
  attr.ah_attr.sl = ib_sl;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = ib_port;

  if (rem.gid.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = rem.gid;
    attr.ah_attr.grh.sgid_index = sgid_idx;
  }

  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
    auto msg = std::string{"Failed transitioning QP to RTR"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = psn;
  attr.max_rd_atomic = 1;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC)) {
    auto msg = std::string{"Failed transitioning QP to RTS"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}

int open_addr_exchange_connection(uint16_t port,
                                  const char *servername = nullptr) {
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = (servername) ? 0 : AI_PASSIVE;  // Use only on server-side

  addrinfo *addrlist;
  {
    auto sport = std::to_string(port);
    auto x = getaddrinfo(servername, sport.c_str(), &hints, &addrlist);
    if (x < 0) {
      auto msg =
          std::string{"Failed to get address info ("} + gai_strerror(x) + ')';
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
  }

  auto conn = (servername) ? connect : bind;

  int connfd = -1;
  for (auto addr = addrlist; addr; addr = addr->ai_next) {
    auto fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) continue;
    int enable = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    if (conn(fd, addr->ai_addr, addr->ai_addrlen) == 0) {
      connfd = fd;
      break;
    }

    close(fd);
  }

  freeaddrinfo(addrlist);

  if (connfd < 0) {
    auto msg = std::string{"Failed to connect to port"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  if (!servername) {
    auto sfd = connfd;
    linux_run(listen(sfd, /* just one, the client */ 1));
    connfd = accept(sfd, nullptr, nullptr);
    close(sfd);
    if (connfd < 0) linux_run(connfd);
  }

  return connfd;
}

ib_addr get_client_addr(ibv_qp *qp, uint16_t port, const ib_addr &loc,
                        uint8_t ib_port, uint32_t psn, ibv_mtu mtu,
                        uint8_t ib_sl, uint8_t sgid_idx) {
  auto connfd = open_addr_exchange_connection(port);

  constexpr size_t gid_size = 32;
  constexpr size_t msg_size =
      sizeof("0000:000000:000000:") + gid_size;  // Includes \0
  char msg[msg_size];

  {
    auto n = read(connfd, msg, msg_size);
    if (n != msg_size) {
      close(connfd);
      auto msg = std::string{"Failed reading remote IB address"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
  }

  ib_addr ret{};

  {
    char gid[gid_size + 1];
    sscanf(msg, "%" SCNx16 ":%" SCNx32 ":%" SCNx32 ":%s", &ret.lid, &ret.qpn,
           &ret.psn, gid);
    wire_gid_to_gid(gid, &ret.gid);
  }

  // Prepare this side of the connection before replying to avoid race condition
  ib_connect(qp, ret, ib_port, psn, mtu, ib_sl, sgid_idx);

  {
    char gid[gid_size + 1];
    gid_to_wire_gid(&loc.gid, gid);
    sprintf(msg, "%04" PRIx16 ":%06" PRIx32 ":%06" PRIx32 ":%s", loc.lid,
            loc.qpn, loc.psn, gid);
  }

  if (write(connfd, msg, msg_size) != msg_size ||
      read(connfd, msg, msg_size) != sizeof("done")) {
    close(connfd);
    auto msg = std::string{"Failed sending local IB address"};
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }

  close(connfd);
  return ret;
}
