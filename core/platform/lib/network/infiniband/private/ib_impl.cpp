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

#include "ib_impl.hpp"

#include <random>

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid);
void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]);

void ib_connect(ibv_qp *qp, const ib_addr &rem, uint8_t ib_port, uint32_t psn,
                ibv_mtu mtu, uint8_t ib_sl, uint8_t sgid_idx);

ib_addr get_server_addr(const std::string &server, ibv_qp *qp, uint16_t port,
                        const ib_addr &loc, uint8_t ib_port, uint32_t psn,
                        ibv_mtu mtu, uint8_t ib_sl, uint8_t sgid_idx);

ib_addr get_client_addr(ibv_qp *qp, uint16_t port, const ib_addr &loc,
                        uint8_t ib_port, uint32_t psn, ibv_mtu mtu,
                        uint8_t ib_sl, uint8_t sgid_idx);

ib_addr IBimpl::get_server(const std::string &server, uint16_t port) const {
  return get_server_addr(server, qp, port, addr, ib_port, addr.psn,
                         IBV_MTU_4096, ib_sl, ib_gidx);
}

ib_addr IBimpl::get_client(uint16_t port) const {
  return get_client_addr(qp, port, addr, ib_port, addr.psn, IBV_MTU_4096, ib_sl,
                         ib_gidx);
}

void IBimpl::post_recv() {
  // FIXME: should request a buffer local to the card
  constexpr size_t buff_size = BlockManager::buffer_size;
  void *recv_region = BlockManager::get_buffer().release();
  ibv_mr *recv_mr = (--reged_mem.upper_bound(recv_region))->second;
  ibv_sge sge{};
  sge.addr = (decltype(sge.addr))recv_region;
  sge.length = buff_size;
  sge.lkey = recv_mr->lkey;

  ibv_recv_wr wr{};
  wr.wr_id = (uintptr_t)recv_region;
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_recv_wr *bad_wr = nullptr;
  // FIXME: which connection?
  linux_run(ibv_post_recv(qp, &wr, &bad_wr));
}

void IBimpl::ensure_listening() {
  // Transition QP to RTR state
  post_recv();

  linux_run(ibv_req_notify_cq(cq, 0));
}

[[nodiscard]] ibv_device_attr IBimpl::query_device() const {
  ibv_device_attr attr{};
  linux_run(ibv_query_device(context.get(), &attr));
  return attr;
}

[[nodiscard]] ibv_port_attr IBimpl::query_port(
    decltype(ibv_qp_attr::port_num) ibport) const {
  ibv_port_attr pattr{};
  linux_run(ibv_query_port(context.get(), ibport, &pattr));
  return pattr;
}

[[nodiscard]] ibv_gid IBimpl::query_port_gid(
    decltype(ibv_qp_attr::port_num) ibport, int ibgidx) const {
  ibv_gid gid{};
  linux_run(ibv_query_gid(context.get(), ibport, ibgidx, &gid));
  return gid;
}

IBimpl::IBimpl(ibv_device *device)
    : context(linux_run(ibv_open_device(device))),
      // Initialize local PD
      pd(linux_run(ibv_alloc_pd(context.get()))),
      comp_channel(ibv_create_comp_channel(context.get())),
      addr(),
      ib_port(1),  // TODO: parameter
      ib_sl(0),    // TODO: parameter
      ib_gidx(0)   // TODO: parameter
{
  {
    ibv_device_attr attr = query_device();

    //    LOG(INFO) << "IB GUID: " << std::hex << attr.node_guid << std::dec
    //              << ", Max CQE: " << attr.max_cqe << ", Max SGE: " <<
    //              attr.max_sge
    //              << ", Ports: " << (uint32_t)attr.phys_port_cnt;

    // TODO: this is a dangerous place to LOG information: if it ends up in
    //  stdcout it will break topology discovery by cloth. The correct fix
    //  is to fix clotho's input expectation.
    LOG_IF(WARNING, attr.phys_port_cnt > 1) << "Unused IB physical ports";

    // Create completion channel to block on it for events
    int max_cqe = std::min(2048, attr.max_cqe);
    int max_sge = std::min(2048, attr.max_sge);
    assert(max_sge > 0);

    {
      //      ibv_cq_init_attr_ex ceq_attr{};
      //      ceq_attr.cqe = max_cqe;
      //      ceq_attr.cq_context = nullptr;
      //      ceq_attr.channel = comp_channel;
      //      ceq_attr.comp_vector = 0;
      //      ceq_attr.parent_domain = pd;
      cq = linux_run(
          ibv_create_cq(context.get(), max_cqe, nullptr, comp_channel, 0));
    }

    {
      ibv_qp_init_attr_ex init_attr{};
      init_attr.send_cq = cq;
      init_attr.recv_cq = cq;
      init_attr.cap.max_send_wr = max_cqe - 1;
      init_attr.cap.max_recv_wr = max_cqe - 1;
      init_attr.cap.max_send_sge = max_sge;
      init_attr.cap.max_recv_sge = max_sge;
      init_attr.qp_type = IBV_QPT_RC;
      init_attr.comp_mask |= IBV_QP_INIT_ATTR_PD;
      //      init_attr.comp_mask |=
      //          IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
      //      init_attr.send_ops_flags |=
      //          IBV_QP_EX_WITH_SEND | IBV_QP_EX_WITH_RDMA_WRITE |
      //          IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD
      //          | IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP;
      init_attr.pd = pd;

      qp = linux_run(ibv_create_qp_ex(context.get(), &init_attr));
    }
  }

  /**
   * To transition the QP into a state that can both send and receive messages
   * we have to go from the RST (reset) state, to the Init state, then to
   * RTR (Ready to Respond) and finally to RTS (ready to request or respond)
   * state
   */
  {
    // Transition QP to Init state
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    // attr.pkey_index = 0;
    attr.port_num = ib_port;
    attr.qp_access_flags = static_cast<unsigned int>(IBV_ACCESS_LOCAL_WRITE) |
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    linux_run(ibv_modify_qp(qp, &attr,
                            static_cast<unsigned int>(IBV_QP_STATE) |
                                IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                IBV_QP_ACCESS_FLAGS));
  }

  ibv_port_attr pattr = query_port(ib_port);

  assert((pattr.link_layer == IBV_LINK_LAYER_ETHERNET || pattr.lid) &&
         "Failed to get LID");

  assert(ib_gidx >= 0);

  std::random_device rd;
  std::mt19937 gen(rd());

  addr.lid = pattr.lid;
  addr.qpn = qp->qp_num;
  addr.psn = gen() & 0xFFFFFFu;
  addr.gid = query_port_gid(ib_port, ib_gidx);

  reged_mem.emplace(((void *)std::numeric_limits<uintptr_t>::max()), nullptr);

  //    BlockManager::reg(*this);
  //
  //    {
  //      // Transition QP to RTR state
  //      post_recv();
  //
  //      linux_run(ibv_req_notify_cq(cq, 0));
  //    }
}

IBimpl::~IBimpl() {
  linux_run(ibv_destroy_qp(qp));
  linux_run(ibv_destroy_cq(cq));
  linux_run(ibv_destroy_comp_channel(comp_channel));
  linux_run(ibv_dealloc_pd(pd));
}

void IBimpl::reg(const void *mem, size_t bytes) {
  ibv_mr *mr = ibv_reg_mr(pd, const_cast<void *>(mem), bytes,
                          static_cast<unsigned int>(IBV_ACCESS_LOCAL_WRITE) |
                              IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    LOG(INFO) << strerror(errno) << " " << errno << " " << ENOMEM;
    mr = ibv_reg_mr(pd, const_cast<void *>(mem), bytes,
                    static_cast<unsigned int>(IBV_ACCESS_LOCAL_WRITE) |
                        IBV_ACCESS_REMOTE_READ);
    if (!mr) {
      LOG(INFO) << strerror(errno) << " " << errno << " " << ENOMEM;
      mr = linux_run(ibv_reg_mr(pd, const_cast<void *>(mem), bytes,
                                IBV_ACCESS_REMOTE_READ));
      assert(errno != ENOMEM);
      LOG(INFO) << "reg (local): " << mem << "-"
                << ((void *)(((char *)mem) + bytes)) << "(ro)";

    } else {
      LOG(INFO) << "reg (local): " << mem << "-"
                << ((void *)(((char *)mem) + bytes)) << "(ro,local-write)";
    }
  } else {
    LOG(INFO) << "reg (local): " << mem << "-"
              << ((void *)(((char *)mem) + bytes)) << "(rw)";
  }
  assert(mr->addr == mem);

  {
    std::unique_lock<std::mutex> lock{m_reg};
    reged_mem.emplace(mem, mr);
  }
}

void IBimpl::unreg(const void *mem) {
  ibv_mr *mr;
  {
    std::unique_lock<std::mutex> lock{m_reg};
    auto it = reged_mem.find(mem);
    assert(it != reged_mem.end() && "Memory not registered to this handler");
    mr = it->second;
    reged_mem.erase(it);
  }

  linux_run(ibv_dereg_mr(mr));

  LOG(INFO) << "unreg: " << mem;
}
