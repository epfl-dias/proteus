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
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <network/infiniband/private/ib_impl.hpp>
#include <thread>

#include "common/error-handling.hpp"
#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "util/logging.hpp"
#include "util/timing.hpp"

constexpr size_t packNotifications = 256;
constexpr size_t buffnum = 2 * packNotifications;

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
  void *recv_region = BlockManager::get_buffer();
  // } else {
  //   recv_region = BlockManager::h_get_buffer(1);
  // }

  // ibv_mr *recv_mr = linux_run(
  //     ibv_reg_mr(pd, recv_region, buff_size,  // BlockManager::buffer_size,
  //                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

  // std::cout << recv_region << std::endl;
  // for (const auto &e : reged_mem) std::cout << e.first << std::endl;
  ibv_mr *recv_mr =
      (--ibd.getIBImpl().reged_mem.upper_bound(recv_region))->second;
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

const topology::cpunumanode &topology::findLocalCPUNumaNode(
    const ib &ib) const {
  return getCpuNumaNodeById(ib.local_cpu);
}

size_t getIBCnt() {
  int dev_cnt = 0;
  ibv_device **dev_list = linux_run(ibv_get_device_list(&dev_cnt));
  ibv_free_device_list(dev_list);
  return dev_cnt;
}

// static ibv_context *getIBdev(int *dev_cnt, size_t ib_indx) {
//  ibv_device **dev_list = linux_run(ibv_get_device_list(dev_cnt));
//
//  ibv_device *ib_dev = dev_list[ib_indx];
//  LOG(INFO) << ibv_get_device_name(ib_dev) << " " << std::hex
//            << ibv_get_device_guid(ib_dev) << std::dec << " "
//            << (void *)ib_dev->dev_path << " " << (void *)ib_dev->ibdev_path
//            << " " << (void *)ib_dev->dev_name << " " << (void *)ib_dev->name;
//  assert(ib_dev && "No IB devices detected");
//
//  auto context = linux_run(ibv_open_device(ib_dev));
//
//  ibv_free_device_list(dev_list);
//
//  return context;
//}

IBHandler::IBHandler(int cq_backlog, const ib &ibd)
    : sub_named(1),
      pending(0),
      dev_cnt(0),
      ibd(ibd),
      local_cpu(topology::getInstance().findLocalCPUNumaNode(ibd)),
      write_cnt(0),
      actual_writes(0),
      cnts((packet_t *)BlockManager::get_buffer()),
      has_requested_buffers(false) {
  LOG(INFO) << "Local IB address: " << this->ibd.getIBImpl().addr;

  active_connections.emplace(uint64_t{0}, nullptr);

  poller = std::thread(&IBHandler::poll_cq, this);
}

IBHandler::~IBHandler() {
  assert(listener.joinable());
  listener.join();

  assert(poller.joinable());
  poller.join();

  BlockManager::unreg(*this);

  assert(ibd.getIBImpl().reged_mem.size() == 1 &&
         "Should only contain ~nullptr");

  linux_run(ibv_destroy_qp(ibd.getIBImpl().qp));
  linux_run(ibv_destroy_cq(ibd.getIBImpl().cq));
  linux_run(ibv_destroy_comp_channel(ibd.getIBImpl().comp_channel));

  try {
    while (!b.empty()) {
      b.back().release();
      b.pop_back();
    }
    //    while (!sub_named.empty()){
    //      sub_named.back().release();
    //      sub_named.pop_back();
    //    }
    if (!sub.q.empty()) {
      BlockManager::release_buffer(
          sub.wait().data);  // FIXME: not really, should wait on something more
    }
    // general...
  } catch (proteus::internal_error &) {
  }
}

void IBHandler::start() {
  //    run();
  listener = std::thread(&IBHandler::run, this);
}

subscription &IBHandler::register_subscriber() { return sub; }

subscription &IBHandler::create_subscription() {
  sub_named.emplace_back(sub_named.size());
  auto &x = sub_named.back();
  ++subcnts;
  return x;
}

void IBHandler::unregister_subscriber(subscription &) {}

void IBHandler::poll_cq() {
  set_exec_location_on_scope loc{local_cpu};
  LOG(INFO) << "Local CPU numa node: " << local_cpu.id;
  // FIXME: remove
  // ibv_cq *cq;
  // void *ctx;  // used to return consumer-supplied CQ context

  size_t i = 0;
  size_t IBV_WC_RDMA_READ_cnt = 0;
  // size_t IBV_WC_RDMA_WRITE_cnt = 0;

  while (true) {
    // linux_run(ibv_get_cq_event(comp_channel, &cq, &ctx));
    // if (++i == cq_ack_backlog) {
    //   ibv_ack_cq_events(cq, i);
    //   i = 0;
    // }
    // linux_run(ibv_req_notify_cq(cq, 0));

    bool ready2exit = false;

    ibv_wc wc{};
    while (ibv_poll_cq(ibd.getIBImpl().cq, 1, &wc)) {
      // ibv_wc wc_arr[16]{};
      // int cnt = 0;
      // while ((cnt = ibv_poll_cq(cq, 16, wc_arr))) {
      // LOG(INFO) << cnt;
      // for (size_t i = 0; i < cnt && !ready2exit; ++i) {
      //   auto &wc = wc_arr[i];
      eventlogger.log(this, IB_CQ_PROCESSING_EVENT_START);
      if (wc.status != IBV_WC_SUCCESS) {
        if (wc.status == IBV_WC_RNR_RETRY_EXC_ERR) {
          LOG(INFO) << "Retry exceeded";
          sleep(1);
          eventlogger.log(this, IB_CQ_PROCESSING_EVENT_END);
          continue;
        }
        eventlogger.log(this, IB_CQ_PROCESSING_EVENT_END);
        auto msg =
            std::string{"completion queue: "} + ibv_wc_status_str(wc.status);
        LOG(INFO) << msg;
        throw std::runtime_error(msg);
      }

      // ibv_mr *recv_mr = (ibv_mr *)wc.wr_id;
      if (wc.opcode & IBV_WC_RECV) {
        proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
        ibv_mr *recv_mr =
            (--ibd.getIBImpl().reged_mem.upper_bound(data.get()))->second;
        if (wc.imm_data == 1 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
          BlockManager::release_buffer(std::move(data));
          auto databuf = BlockManager::get_buffer();
          for (size_t i = 0; i < buffnum; ++i) {
            auto buff = BlockManager::get_buffer();
            b.emplace_back(buff);
            ibv_mr *send_buff =
                (--ibd.getIBImpl().reged_mem.upper_bound(buff))->second;
            ((buffkey *)databuf)[i] =
                std::make_pair((void *)buff, send_buff->rkey);
          }

          ibv_mr *send_reg =
              (--ibd.getIBImpl().reged_mem.upper_bound(databuf))->second;
          // Find connection for target node
          // rdma_cm_id *conn = active_connections.at(0);

          ibv_sge sge{/* 0 everything out via value initialization */};

          static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
          // super dangerous cast, do not remove above static_assert!
          sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(databuf);
          sge.length =
              static_cast<decltype(ibv_sge::length)>(buffnum * sizeof(buffkey));
          sge.lkey = send_reg->lkey;

          send_sge((uintptr_t)databuf, &sge, 1, 2);
          post_recv(ibd.getIBImpl().qp);
          eventlogger.log(this, IB_CQ_PROCESSING_EVENT_END);
          continue;
        }
        if (wc.imm_data != 0 || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          post_recv(ibd.getIBImpl().qp);
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
        if (wc.imm_data == 2 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
          // buffers.publish(data, wc.byte_len);
          std::unique_lock<std::mutex> lock{pend_buffers_m};

          // auto x = buffers.wait();
          assert(wc.byte_len == buffnum * sizeof(buffkey));
          for (size_t i = 0; i < buffnum; ++i) {
            pend_buffers.emplace_back(((buffkey *)data.get())[i]);
          }
          BlockManager::release_buffer(std::move(data));

          pend_buffers_cv.notify_all();
        } else if (wc.imm_data == 13 &&
                   wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
          auto h =
              (std::pair<decltype(reged_remote_mem)::key_type,
                         decltype(reged_remote_mem)::mapped_type> *)data.get();
          reged_remote_mem.emplace(h->first, h->second);
          BlockManager::release_buffer(std::move(data));
        } else {
          if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            BlockManager::release_buffer(std::move(data));
            size_t buffcnt = wc.byte_len / sizeof(packet_t);
            auto *sizes = (packet_t *)b[wc.imm_data].release();
            for (size_t i = 0; i < buffcnt; ++i) {
              // LOG(INFO) << sizes[i];
              auto size = sizes[i].first;
              if (size == (-1)) {
                b.pop_front();
                continue;
              } else if (((int64_t)size) >= 0) {
                // LOG(INFO) << size;
                assert(140612760240128 != size);
                assert(85771059561805907 != size);
                if (sizes[i].second == 0) {
                  sub.publish(std::move(b.front()), size);
                } else {
                  while (sizes[i].second >= subcnts)
                    ;  // FIXME: should not allow locking here!!!
                  sub_named[sizes[i].second].publish(std::move(b.front()),
                                                     size);
                }
                b.pop_front();
              } else {  // originated from write_to
                // LOG(INFO) << "here " << -size;
                sub.publish(nullptr, -size);
                // b.pop_front();
              }
            }
            BlockManager::release_buffer(sizes);
            b.pop_front();
          } else {
            LOG(INFO) << "outer";
            sub.publish(std::move(data), wc.byte_len);
          }
        }

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
        proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
        if (data) {
          BlockManager::release_buffer(std::move(data));
          // LOG(INFO) << "out "
          //           <<
          //           std::chrono::duration_cast<std::chrono::milliseconds>(
          //                  std::chrono::system_clock::now() - ::start)
          //                  .count();
          // recv_mr = (--reged_mem.upper_bound(data))->second;
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
      } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
        proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
        if (data) {
          {
            // std::lock_guard<std::mutex> lock{write_promises_m};
            for (size_t i = 0; i < 1; ++i) {
              // assert(!write_promises.empty());
              // assert(write_promises.size() > IBV_WC_RDMA_WRITE_cnt);
              std::pair<subscription, proteus::managed_ptr> *p;
              write_promises.pop(p);
              // //.front();
              if (p->second) {
                // auto &p = read_promises.front();
                // LOG(INFO) << "read completed successfull_y";
                // BlockManager::release_buffer(p->second);
                p->first.publish(std::move(p->second), 0);
                //                release_buffer(data);
                // p.first.set_value(p.second);
                // read_promises.pop_front();
              }
              // delete p;
            }
          }
          BlockManager::release_buffer(std::move(data));
        }
      } else if (wc.opcode == IBV_WC_RDMA_READ) {
        std::unique_lock<std::mutex> lock{read_promises_m};
        for (size_t i = 0; i < (size_t)wc.wr_id; ++i) {
          assert(!read_promises.empty());
          auto &p = read_promises[IBV_WC_RDMA_READ_cnt++];  //.front();
          // auto &p = read_promises.front();
          // LOG(INFO) << "read completed successfull_y";
          // BlockManager::release_buffer(p.second);
          std::get<0>(p).publish(std::move(std::get<1>(p)), 0);
          InfiniBandManager::release_buffer(std::move(std::get<2>(p)));
          // p.first.set_value(p.second);
          // read_promises.pop_front();
        }
      } else {
        LOG(FATAL) << "Unknown: " << wc.opcode;
      }
      eventlogger.log(this, IB_CQ_PROCESSING_EVENT_END);
      // }
    }

    if (ready2exit) {
      LOG(INFO) << "Bailing out...";
      ibv_ack_cq_events(ibd.getIBImpl().cq, i);
      LOG(INFO) << "Bailing out";
      return;
    }
  }
}

int IBHandler::send(ibv_send_wr &wr, ibv_send_wr **save_on_error, bool retry) {
  int ret;
  int i = 0;
  while ((ret = ibv_post_send(ibd.getIBImpl().qp, &wr, save_on_error)) != 0) {
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

void IBHandler::flush() {
  std::lock_guard<std::mutex> lock{write_promises_m};
  flush_write();
}

static size_t reads = 0;

void IBHandler::flush_read() {
  // auto buff = BlockManager::get_buffer();

  // auto p = new std::promise<void *>;
  // auto d = new std::pair<decltype(p), void *>;
  // d->first = p;
  // d->second = buff;

  // LOG(INFO) << 0 << " " << (void *)nullptr << " " << (void *)nullptr;
  // auto &p = create_promise(buff);

  // ibv_mr *send_reg = (--reged_mem.upper_bound(buff))->second;

  // ibv_sge sge{/* 0 everything out via value initialization */};

  // static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
  // // super dangerous cast, do not remove above static_assert!
  // sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(buff);
  // assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
  // sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
  // sge.lkey = send_reg->lkey;

  ibv_send_wr wr{/* 0 everything out via value initialization */};

  wr.wr_id = (uintptr_t)reads;  // readable locally on work completion
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = nullptr;
  wr.num_sge = 0;
  wr.send_flags = IBV_SEND_SIGNALED;  //(++reads % 64) ? 0 : IBV_SEND_SIGNALED;
  // assert(!sigoverflow || (c > 2));
  // wr.imm_data = c + 3;  // NOTE: only sent when singaled == true
  // Can pass an arbitrary value to receiver, consider for
  // the consumed buffer
  wr.wr.rdma.remote_addr =
      reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(nullptr);
  wr.wr.rdma.rkey = 0;

  if (wr.send_flags & IBV_SEND_SIGNALED) reads = 0;

  // LOG(INFO) << bytes << " " << buff << " " << data;
  linux_run(send(wr));
}

static void *old_cnts = nullptr;

decltype(IBHandler::read_promises)::value_type &IBHandler::create_promise(
    proteus::managed_ptr buff, proteus::remote_managed_ptr from) {
  std::unique_lock<std::mutex> lock{read_promises_m};
  read_promises.emplace_back(5, std::move(buff), std::move(from));
  return read_promises.back();
}

decltype(IBHandler::write_promises)::value_type IBHandler::create_write_promise(
    void *buff) {
  auto ptr = new std::pair<subscription, proteus::managed_ptr>(5, buff);
  write_promises.push(ptr);
  // write_promises.emplace_back(5, buff);
  return ptr;
}

subscription *IBHandler::read_event() {
  eventlogger.log(this, IB_CREATE_RDMA_READ_START);
  // auto buff = BlockManager::get_buffer();

  // auto p = new std::promise<void *>;
  // auto d = new std::pair<decltype(p), void *>;
  // d->first = p;
  // d->second = buff;

  // LOG(INFO) << bytes << " " << (void *)nullptr << " " << data;
  auto &p = create_promise(nullptr, nullptr);

  ibv_send_wr wr{/* 0 everything out via value initialization */};

  ++reads;

  wr.wr_id = (uintptr_t)reads;  // readable locally on work completion
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = nullptr;
  wr.num_sge = 0;
  wr.send_flags = (reads % 512) ? 0 : IBV_SEND_SIGNALED;
  // assert(!sigoverflow || (c > 2));
  // wr.imm_data = c + 3;  // NOTE: only sent when singaled == true
  // Can pass an arbitrary value to receiver, consider for
  // the consumed buffer
  wr.wr.rdma.remote_addr =
      reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(nullptr);
  wr.wr.rdma.rkey = 0;

  if (wr.send_flags & IBV_SEND_SIGNALED) reads = 0;

  // LOG(INFO) << bytes << " " << buff << " " << data;
  linux_run(send(wr));

  eventlogger.log(this, IB_CREATE_RDMA_READ_END);
  return &std::get<0>(p);
}

subscription *IBHandler::read(proteus::remote_managed_ptr data, size_t bytes) {
  eventlogger.log(this, IB_CREATE_RDMA_READ_START);
  auto buff = proteus::managed_ptr{BlockManager::get_buffer()};

  assert(ibd.getIBImpl().reged_mem.upper_bound(buff.get()) !=
         ibd.getIBImpl().reged_mem.begin());
  ibv_mr *send_reg =
      (--(ibd.getIBImpl().reged_mem.upper_bound(buff.get())))->second;

  ibv_sge sge{/* 0 everything out via value initialization */};

  static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
  // super dangerous cast, do not remove above static_assert!
  sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(buff.get());
  assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
  sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
  sge.lkey = send_reg->lkey;

  ibv_send_wr wr{/* 0 everything out via value initialization */};

  ++reads;

  wr.wr_id = (uintptr_t)reads;  // readable locally on work completion
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = (reads % 512) ? 0 : IBV_SEND_SIGNALED;
  // assert(!sigoverflow || (c > 2));
  // wr.imm_data = c + 3;  // NOTE: only sent when singaled == true
  // Can pass an arbitrary value to receiver, consider for
  // the consumed buffer
  wr.wr.rdma.remote_addr =
      reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(data.get());
  wr.wr.rdma.rkey = (--reged_remote_mem.upper_bound(data.get()))->second;

  if (wr.send_flags & IBV_SEND_SIGNALED) reads = 0;

  // create promise before submitting the work request
  auto &p = create_promise(std::move(buff), std::move(data));

  linux_run(send(wr));

  eventlogger.log(this, IB_CREATE_RDMA_READ_END);
  return &std::get<0>(p);
}

void IBHandler::write_to(proteus::managed_ptr data, size_t bytes,
                         buffkey buff) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  // get_buffer();
  cnts[write_cnt++] = std::make_pair(-bytes, 0);
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  write_to_int(std::move(data), bytes, buff);
}

void IBHandler::write(proteus::managed_ptr data, size_t bytes, size_t sub_id) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  auto buff = get_buffer();
  cnts[write_cnt++] = std::make_pair(bytes, sub_id);
  actual_writes++;
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  write_to_int(std::move(data), bytes, buff);
}

subscription *IBHandler::write_silent(proteus::managed_ptr data, size_t bytes) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  auto buff = get_buffer();
  cnts[write_cnt++] = std::make_pair(-1, 0);
  actual_writes++;
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  return write_to_int(std::move(data), bytes, buff, buff.first);
}

void IBHandler::flush_write() {
  auto buff = get_buffer();
  void *buffpromise = nullptr;
  packet_t *data = cnts;
  BlockManager::release_buffer(old_cnts);
  // old_cnts = cnts;
  size_t bytes = write_cnt * sizeof(packet_t);
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  cnts = (packet_t *)BlockManager::get_buffer();
  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  subscription *ret;
  {
    // std::unique_lock<std::mutex> lock{write_promises_m};
    ibv_mr *send_reg = (--ibd.getIBImpl().reged_mem.upper_bound(data))->second;

    auto p = create_write_promise(buffpromise);

    ibv_sge sge{/* 0 everything out via value initialization */};

    static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
    // super dangerous cast, do not remove above static_assert!
    sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(data);
    assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
    sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
    sge.lkey = send_reg->lkey;

    ibv_send_wr wr{/* 0 everything out via value initialization */};

    wr.wr_id = (uintptr_t)data;  // readable locally on work completion
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = actual_writes;  // Unused

    wr.wr.rdma.remote_addr =
        reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(buff.first);
    wr.wr.rdma.rkey = buff.second;

    linux_run(send(wr));
    // TODO: save_on_error contains a pointer to the "bad" request, handle
    // more gracefully

    ret = &p->first;
    write_cnt = 0;
    actual_writes = 0;
  }
}

subscription *IBHandler::write_to_int(proteus::managed_ptr data, size_t bytes,
                                      buffkey buff, void *buffpromise) {
  bool should_flush = false;
  subscription *ret;
  {
    // std::unique_lock<std::mutex> lock{write_promises_m};
    ibv_mr *send_reg =
        (--ibd.getIBImpl().reged_mem.upper_bound(data.get()))->second;

    auto p = create_write_promise(buffpromise);

    ibv_sge sge{/* 0 everything out via value initialization */};

    static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
    // super dangerous cast, do not remove above static_assert!
    sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(data.get());
    assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
    sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
    sge.lkey = send_reg->lkey;

    ibv_send_wr wr{/* 0 everything out via value initialization */};

    static_assert(sizeof(uintptr_t) == sizeof(proteus::managed_ptr),
                  "Overflow");
    wr.wr_id = reinterpret_cast<uintptr_t>(
        data.release());  // readable locally on work completion
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = actual_writes;  // Unused

    wr.wr.rdma.remote_addr =
        reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(buff.first);
    wr.wr.rdma.rkey = buff.second;

    linux_run(send(wr));
    // TODO: save_on_error contains a pointer to the "bad" request, handle
    // more gracefully

    ret = &p->first;
    should_flush = ((write_cnt % packNotifications) == 0);
  }

  if (should_flush) flush_write();

  return ret;
}

void IBHandler::request_buffers_unsafe() { send_sge(0, nullptr, 0, 1); }

static std::mutex m2;
static std::map<void *, decltype(buffkey::second)> keys;

void IBHandler::release_buffer(proteus::remote_managed_ptr p) {
  if (!p) return;
  // FIXME: Unsafe if the buffer is from a file or MemoryManager
  try {
    // try to get the key, before locking
    auto k = keys.at(p.get());
    // if key exists, this is a remote buffer and we have ownership
    // FIXME: what about shared buffers? We may not have ownership
    std::unique_lock<std::mutex> lock{pend_buffers_m};
    pend_buffers.emplace_front(p.get(), k);
  } catch (std::out_of_range &) {
    // The ptr didn't come from the buffer manager, ignore
  }
  ((void)/* Release and ignore */ p.release());
}

buffkey IBHandler::get_buffer() {
  std::unique_lock<std::mutex> lock{pend_buffers_m};
  if (pend_buffers.size() == ((size_t)(buffnum * 0.1)) ||
      !has_requested_buffers) {
    request_buffers_unsafe();
    has_requested_buffers = true;
  }

  if (pend_buffers.empty()) {
    pend_buffers_cv.wait(lock, [&]() { return !pend_buffers.empty(); });
  }

  auto b = pend_buffers.front();
  pend_buffers.pop_front();
  keys.emplace(b.first, b.second);
  // LOG(INFO) << b.first;
  return b;
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
  //       pd, data, bytes, IBV_ACCESS_LOCAL_WRITE |
  //       IBV_ACCESS_REMOTE_WRITE));
  //   eventlogger.log(this, IBV_REG_END);
  // }

  // std::cout << data << std::endl;
  // for (const auto &e : reged_mem) std::cout << e.first << std::endl;
  // std::cout << "=" << reged_mem.upper_bound(data)->first << std::endl;
  // std::cout << (--reged_mem.upper_bound(data))->first << std::endl;
  ibv_mr *send_reg = (--ibd.getIBImpl().reged_mem.upper_bound(data))->second;

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
  BlockManager::release_buffer(
      sub.wait().data);  // FIXME: not really, should wait on something more
                         // general...
  LOG(INFO) << "done!";
  // linux_run(rdma_disconnect(active_connections[0]));
}

void IBHandler::reg(const void *mem, size_t bytes) {
  ibd.getIBImpl().reg(mem, bytes);
}

buffkey IBHandler::reg2(const void *mem, size_t bytes) {
  reg(mem, bytes);

  auto x =
      std::make_pair(mem, ibd.getIBImpl().reged_mem.find(mem)->second->rkey);
  auto f = (decltype(&x))BlockManager::get_buffer();
  *f = x;
  send(f, sizeof(x), 13);
  return std::make_pair((void *)mem,
                        ibd.getIBImpl().reged_mem.find(mem)->second->rkey);
}

void IBHandler::unreg(const void *mem) { ibd.getIBImpl().unreg(mem); }
