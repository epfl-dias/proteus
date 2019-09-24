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

#include "infiniband-handler.hpp"

#include <arpa/inet.h>
#include <err.h>
#include <glog/logging.h>
#include <malloc.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cassert>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"

template <>
void **buffer_manager<int32_t>::h_h_buff_start;
template <>
size_t *buffer_manager<int32_t>::h_size;

std::ostream &operator<<(std::ostream &out, const ib_addr &addr) {
  out << "LID 0x" << std::setfill('0') << std::setw(4) << std::hex << addr.lid
      << " QPN 0x" << std::setfill('0') << std::setw(6) << std::hex << addr.qpn
      << " PSN 0x" << std::setfill('0') << std::setw(6) << std::hex << addr.psn
      << " GID " << addr.gid;
  return out;
}

std::ostream &operator<<(std::ostream &out, const ibv_gid &gid) {
  constexpr size_t max_ipv6_charlength = 32;
  char gidc[max_ipv6_charlength + 1];
  linux_run(inet_ntop(AF_INET6, &gid, gidc, max_ipv6_charlength + 1));
  return (out << gidc);
}

void IBHandler::post_recv(ibv_qp *qp) {
  // FIXME: should request a buffer local to the card
  // recv_region = BlockManager::get_buffer();
  constexpr size_t buff_size = BlockManager::buffer_size;
  // recv_region = BlockManager::h_get_buffer(1);
  // size_t buff_size = ((size_t)16) * 1024 * 1024 * 1024;
  // recv_region = MemoryManager::mallocPinned(buff_size);

  // for (const auto &gpu : topology::getInstance().getGpus()) {
  //   set_device_on_scope d{gpu};
  //   gpu_run(cudaMalloc(&recv_region, buff_size));

  //   // CUdeviceptr d_A;
  //   // gpu_run(cuMemAlloc(&d_A, BlockManager::buffer_size *
  //   sizeof(int32_t)));
  //   // OUT << "device ptr: " << hex << d_A << dec << endl;

  //   // unsigned int flag = 1;
  //   // gpu_run(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
  //   // d_A));

  //   // FIXME: Registering per recv will probably cause a huge overhead
  //   LOG(INFO) << recv_region;
  //   ibv_mr *recv_mr =                           // linux_run(
  //       ibv_reg_mr(pd, recv_region, buff_size,  // BlockManager::buffer_size,
  //                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);  //);
  //   LOG(INFO) << recv_region << " registered: " << recv_mr;
  // }

  // LOG(INFO) << "cu: ";
  // for (const auto &gpu : topology::getInstance().getGpus()) {
  //   set_device_on_scope d{gpu};
  //   CUdeviceptr r = 0;
  //   gpu_run(cuMemAlloc(&r, buff_size));

  //   // CUDA_POINTER_ATTRIBUTE_P2P_TOKENS return_data;
  //   // gpu_run(cuPointerGetAttribute(&return_data,
  //   // CU_POINTER_ATTRIBUTE_P2P_TOKENS,
  //   //                               r));
  //   // // CUdeviceptr d_A;
  //   // // gpu_run(cuMemAlloc(&d_A, BlockManager::buffer_size *
  //   // sizeof(int32_t)));
  //   // // OUT << "device ptr: " << hex << d_A << dec << endl;

  //   // // unsigned int flag = 1;
  //   // // gpu_run(cuPointerSetAttribute(&flag,
  //   CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
  //   // // d_A));

  //   unsigned int flag = 1;
  //   gpu_run(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
  //   r));

  //   int managed = 0;
  //   gpu_run(
  //       cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, r));
  //   LOG(INFO) << managed;

  //   gpu_run(cuPointerGetAttribute(&recv_region,
  //                                 CU_POINTER_ATTRIBUTE_DEVICE_POINTER, r));

  //   LOG(INFO) << recv_region;
  //   ibv_mr *recv_mr =                           // linux_run(
  //       ibv_reg_mr(pd, recv_region, buff_size,  // BlockManager::buffer_size,
  //                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);  //);
  //   LOG(INFO) << recv_region << " registered: " << recv_mr;

  //   CUDA_POINTER_ATTRIBUTE_P2P_TOKENS return_data;
  //   LOG(INFO) << cuPointerGetAttribute(&return_data,
  //                                      CU_POINTER_ATTRIBUTE_P2P_TOKENS, r)
  //             << " " << CUDA_SUCCESS;
  //   // FIXME: Registering per recv will probably cause a huge overhead
  //   LOG(INFO) << return_data.p2pToken << " " << return_data.vaSpaceToken;
  // }

  // exit(0);

  // if (topology::getInstance().getGpus().size() <= 2) {
  recv_region = BlockManager::get_buffer();
  // } else {
  //   recv_region = BlockManager::h_get_buffer(1);
  // }

  // ibv_mr *recv_mr = linux_run(
  //     ibv_reg_mr(pd, recv_region, buff_size,  // BlockManager::buffer_size,
  //                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

  // std::cout << recv_region << std::endl;
  // for (const auto &e : reged_mem) std::cout << e.first << std::endl;
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
  // LOG(INFO) << "Added receive buffer";
}

void IBHandler::run() {
  // assert(ec);
  // rdma_cm_event *event;
  // LOG(INFO) << "run";
  // while (rdma_get_cm_event(ec, &event) == 0) {
  //   LOG(INFO) << "event";
  //   // Create a local copy of event, as it gets invalidated by the ack
  //   rdma_cm_event local_event{*event};

  //   LOG(INFO) << "acking";
  //   rdma_ack_cm_event(event);

  //   LOG(INFO) << "IB:peer=" << rdma_get_peer_addr(connid) << ": received
  //   event "
  //             << rdma_event_str(local_event.event);

  //   if (handle_event(local_event)) break;
  // }

  // rdma_destroy_id(connid);
  // rdma_destroy_event_channel(ec);
}

IBHandler::IBHandler(int cq_backlog) : pending(0), recv_region(nullptr) {
  ibv_device **dev_list = linux_run(ibv_get_device_list(nullptr));

  ibv_device *ib_dev = dev_list[0];
  assert(ib_dev && "No IB devices detected");
  ib_port = 1;  // TODO: parameter
  ib_gidx = 0;  // strtol(optarg, nullptr, 0);   // TODO: parameter
  ib_sl = 0;    // TODO: parameter

  // Initialize local PD
  context = linux_run(ibv_open_device(ib_dev));
  pd = linux_run(ibv_alloc_pd(context));

  ibv_device_attr attr{};
  linux_run(ibv_query_device(context, &attr));

  LOG(INFO) << "IB GUID: " << std::hex << attr.node_guid << std::dec
            << ", Max CQE: " << attr.max_cqe
            << ", Ports: " << (uint32_t)attr.phys_port_cnt;

  // Create completion channel to block on it for events
  int max_cqe = std::min(128, attr.max_cqe);
  comp_channel = linux_run(ibv_create_comp_channel(context));
  cq = linux_run(ibv_create_cq(context, max_cqe, nullptr, comp_channel, 0));

  {
    ibv_qp_init_attr init_attr{};
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.cap.max_send_wr = max_cqe - 1;
    init_attr.cap.max_recv_wr = max_cqe - 1;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.qp_type = IBV_QPT_RC;

    qp = linux_run(ibv_create_qp(pd, &init_attr));
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
    // attr.qp_access_flags = 0;

    linux_run(ibv_modify_qp(
        qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  }

  reged_mem.emplace(((void *)std::numeric_limits<uintptr_t>::max()), nullptr);

  {
    for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
      char *ptr = (char *)BlockManager::h_h_buff_start[cpu.id];
      size_t bytes = BlockManager::h_size[cpu.id] * sizeof(int32_t) *
                     BlockManager::block_size;
      // for (auto t = ptr; t < ptr + bytes;
      //      t += (sizeof(int32_t) * BlockManager::h_size)) {
      //   reg(t, bytes);
      //   break;
      // }
      reg(ptr, bytes);
    }
  }

  {
    // Transition QP to RTR state

    post_recv(qp);

    linux_run(ibv_req_notify_cq(cq, 0));
  }

  ibv_port_attr pattr;
  linux_run(ibv_query_port(context, ib_port, &pattr));

  assert((pattr.link_layer == IBV_LINK_LAYER_ETHERNET || pattr.lid) &&
         "Failed to get LID");

  assert(ib_gidx >= 0);

  addr.lid = pattr.lid;
  addr.qpn = qp->qp_num;
  addr.psn = rand() & 0xffffff;
  ibv_gid gid{};
  linux_run(ibv_query_gid(context, ib_port, ib_gidx, &gid));
  addr.gid = gid;

  LOG(INFO) << "Local IB address: " << addr;

  active_connections.emplace(uint64_t{0}, nullptr);

  poller = std::thread(&IBHandler::poll_cq, this);
}

IBHandler::~IBHandler() {
  assert(listener.joinable());
  listener.join();

  assert(poller.joinable());
  poller.join();

  for (const auto &cpu : topology::getInstance().getCpuNumaNodes()) {
    InfiniBandManager::unreg(BlockManager::h_h_buff_start[cpu.id]);
  }
  assert(reged_mem.size() == 1 && "Should only contain ~nullptr");

  linux_run(ibv_destroy_qp(qp));
  linux_run(ibv_destroy_cq(cq));
  linux_run(ibv_destroy_comp_channel(comp_channel));

  linux_run(ibv_dealloc_pd(pd));
  linux_run(ibv_close_device(context));
}

void IBHandler::start() {
  //    run();
  listener = std::thread(&IBHandler::run, this);
}

subscription &IBHandler::register_subscriber() { return sub; }

void IBHandler::unregister_subscriber(subscription &) {}

void IBHandler::poll_cq() {
  set_exec_location_on_scope loc{topology::getInstance().getCpuNumaNodes()[0]};
  // FIXME: remove
  ibv_cq *cq;
  void *ctx;  // used to return consumer-supplied CQ context

  size_t i = 0;

  while (true) {
    linux_run(ibv_get_cq_event(comp_channel, &cq, &ctx));
    // if (++i == 16) {
    ibv_ack_cq_events(cq, i);
    i = 0;
    // }
    linux_run(ibv_req_notify_cq(cq, 0));

    bool ready2exit = false;

    ibv_wc wc{};
    while (ibv_poll_cq(cq, 1, &wc)) {
      if (wc.status != IBV_WC_SUCCESS) {
        if (wc.status == IBV_WC_RNR_RETRY_EXC_ERR) {
          sleep(1);
          continue;
        }
        auto msg =
            std::string{"completion queue: "} + ibv_wc_status_str(wc.status);
        LOG(INFO) << msg;
        throw std::runtime_error(msg);
        continue;
      }

      void *data = (void *)wc.wr_id;
      ibv_mr *recv_mr = nullptr;
      // ibv_mr *recv_mr = (ibv_mr *)wc.wr_id;
      if (wc.opcode & IBV_WC_RECV) {
        recv_mr = (--reged_mem.upper_bound(data))->second;
        if (wc.imm_data != 0) {
          post_recv(qp);
        } else {
          if (!saidGoodBye) sendGoodBye();
          LOG(INFO) << "respond good bye";
          ready2exit = true;
        }
        //          struct connection *conn = (struct connection
        //          *)(uintptr_t)wc.wr_id;
        // LOG(INFO) << "received message";
        //          sub.publish(wc)
        //          std::cout << wc.byte_len << std::endl;
        //          std::cout << wc.imm_data << std::endl;
        //          std::cout << (void *) wc.wr_id << std::endl;

        // if (topology::getInstance().getGpus().size() <= 2) {
        // sub.publish((void *)recv_mr->addr, wc.byte_len);
        sub.publish(data, wc.byte_len);

        // } else {
        //   void *cpumem = BlockManager::get_buffer();
        //   BlockManager::overwrite_bytes(cpumem, recv_mr->addr, wc.byte_len,
        //   0,
        //                                 true);
        //   BlockManager::release_buffer((int *)recv_mr->addr);
        //   sub.publish(cpumem, wc.byte_len);
        // }

        if (recv_mr) {
          // time_block t{"dereg"};
          // eventlogger.log(this, IBV_DEREG_START);
          // linux_run(ibv_dereg_mr(recv_mr));
          // eventlogger.log(this, IBV_DEREG_END);
        }
        // LOG(INFO) << "notification completed";
      } else if (wc.opcode == IBV_WC_SEND) {
        if (data) {
          recv_mr = (--reged_mem.upper_bound(data))->second;
          // time_block t{"dereg"};
          // eventlogger.log(this, IBV_DEREG_START);
          // linux_run(ibv_dereg_mr(recv_mr));
          // eventlogger.log(this, IBV_DEREG_END);

          // {
          //   std::unique_lock<std::mutex> lk{m};
          //   --pending;
          //   cv.notify_all();
          // }
        }
        // LOG(INFO) << "send completed successfully";
      }
    }

    if (ready2exit) {
      ibv_ack_cq_events(cq, i);
      LOG(INFO) << "Bailing out";
      return;
    }
  }
}

int IBHandler::send(ibv_send_wr &wr, ibv_send_wr **save_on_error, bool retry) {
  int ret;
  int i = 0;
  // LOG(INFO) << "send";
  while ((ret = ibv_post_send(qp, &wr, save_on_error)) != 0) {
    if (ret != ENOMEM) return ret;
    if (i++ == 0) LOG(INFO) << "Sleeping";
    // std::this_thread::sleep_for(std::chrono::microseconds{50});
  }

  return 0;
}

int IBHandler::send(ibv_send_wr &wr, bool retry) {
  ibv_send_wr *save_on_error = nullptr;
  return send(wr, &save_on_error, retry);
}

void IBHandler::send_sge(uintptr_t wr_id, ibv_sge *sg_list, size_t sge_cnt,
                         decltype(ibv_send_wr::imm_data) imm) {
  ibv_send_wr wr{/* 0 everything out via value initialization */};

  wr.wr_id = wr_id;  // readable locally on work completion
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.sg_list = sg_list;
  wr.num_sge = sge_cnt;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm;  // Can pass an arbitrary value to receiver, consider for
  // the consumed buffer

  linux_run(send(wr));
  // TODO: save_on_error contains a pointer to the "bad" request, handle
  // more gracefully
}

void IBHandler::sendGoodBye() {
  saidGoodBye = true;
  send_sge(0, nullptr, 0, 0);
}

void IBHandler::send(void *data, size_t bytes,
                     decltype(ibv_send_wr::imm_data) imm) {
  // constexpr size_t ib_slack = 128;
  // {
  //   eventlogger.log(this, IB_LOCK_CONN_START);
  //   std::unique_lock<std::mutex> lk{m};
  //   cv.wait(lk, [this] {
  //     return !active_connections.empty() || pending > ib_slack;
  //   });
  //   ++pending;
  //   eventlogger.log(this, IB_LOCK_CONN_END);
  // }

  // FIXME: Registering per send will probably cause a huge overhead
  // Register memory with IB
  // ibv_mr *send_reg;
  // {
  //   time_block t{"reg"};
  //   eventlogger.log(this, IBV_REG_START);
  //   send_reg = linux_run(ibv_reg_mr(
  //       pd, data, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
  //   eventlogger.log(this, IBV_REG_END);
  // }

  // std::cout << data << std::endl;
  // for (const auto &e : reged_mem) std::cout << e.first << std::endl;
  // std::cout << "=" << reged_mem.upper_bound(data)->first << std::endl;
  // std::cout << (--reged_mem.upper_bound(data))->first << std::endl;
  ibv_mr *send_reg = (--reged_mem.upper_bound(data))->second;

  // Find connection for target node
  // rdma_cm_id *conn = active_connections.at(0);

  ibv_sge sge{/* 0 everything out via value initialization */};

  static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
  // super dangerous cast, do not remove above static_assert!
  sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(data);
  assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
  sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
  sge.lkey = send_reg->lkey;

  send_sge((uintptr_t)data, &sge, 1, imm);
}

void IBHandler::disconnect() {
  // Say good bye, otherwise remote will crash waiting for a recv
  LOG(INFO) << "send good bye";
  sendGoodBye();
  LOG(INFO) << "waiting for confirmation";
  sub.wait();  // FIXME: not really, should wait on something more general...
  LOG(INFO) << "done!";
  // linux_run(rdma_disconnect(active_connections[0]));
}

void *IBHandler::reg(void *mem, size_t bytes) {
  LOG(INFO) << "reg: " << mem << "-" << ((void *)(((char *)mem) + bytes));

  ibv_mr *mr = linux_run(ibv_reg_mr(
      pd, mem, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

  {
    std::unique_lock<std::mutex> lock{m_reg};
    reged_mem.emplace(mem, mr);
  }

  return mem;
}

void IBHandler::unreg(void *mem) {
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
