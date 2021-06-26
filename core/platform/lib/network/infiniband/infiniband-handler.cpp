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
#include <glog/logging.h>
#include <malloc.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cassert>
#include <iomanip>
#include <platform/common/error-handling.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/network/infiniband/infiniband-handler.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/logging.hpp>
#include <thread>

#include "private/ib_impl.hpp"

constexpr size_t packNotifications = 256;
constexpr size_t buffnum = 8 * packNotifications;

constexpr unsigned int BUFFRELEASE = 19;

static std::atomic<size_t> avail_remote_buffs = 0;

static std::mutex m2;
static std::map<void *, decltype(buffkey::second)> keys;

static std::mutex m3;
static void *released[128]{nullptr};
static size_t released_cnt = 0;

static std::mutex buffmgmnt;

static std::atomic<size_t> reads = 0;

static size_t invocation_cnts[10]{};

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
  constexpr size_t buff_size = BlockManager::buffer_size;
  auto recv_region = BlockManager::get_buffer();

  ibv_sge sge = sge_for_buff(recv_region.get(), buff_size);

  ibv_recv_wr wr{};
  reinterpret_cast<proteus::managed_ptr &>(wr.wr_id) = std::move(recv_region);
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_recv_wr *bad_wr = nullptr;
  linux_run(ibv_post_recv(qp, &wr, &bad_wr));
}

const topology::cpunumanode &topology::findLocalCPUNumaNode(
    const ib &ib) const {
  return getCpuNumaNodeById(ib.local_cpu);
}

class IBPoller {
  IBHandler &handler;
  bool ready2exit = false;

  void handle_wc_recv(const ibv_wc &wc);
  static void handle_wc_failure(const ibv_wc &wc);

 public:
  explicit IBPoller(IBHandler &handler) : handler(handler) {}
  void run();
};

IBHandler::IBHandler(const ib &ibd)
    : sub_named(),
      pending(0),
      ibd(ibd),
      local_cpu(topology::getInstance().findLocalCPUNumaNode(ibd)),
      write_cnt(0),
      actual_writes(0),
      cnts((packet_t *)BlockManager::get_buffer().release()),
      has_requested_buffers(false) {
  create_subscription();
  LOG(INFO) << "Local IB address: " << this->ibd.getIBImpl().addr;

  active_connections.emplace(uint64_t{0}, nullptr);

  poll_cq = std::make_unique<IBPoller>(*this);
  poller = std::thread(&IBPoller::run, poll_cq.get());
  pthread_setname_np(poller.native_handle(), "poll_cq");

  flusher = std::thread([this]() {
    while (!saidGoodBye) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
      flush_read();
      flush();
    }
  });

  request_buffers_unsafe();
}

IBHandler::~IBHandler() {
  flusher.join();
  assert(poller.joinable());
  poller.join();

  BlockManager::unreg(*this);

  assert(ibd.getIBImpl().reged_mem.size() == 1 &&
         "Should only contain ~nullptr");

  assert(write_promises.empty_unsafe());
  try {
    while (!send_buffers.empty()) {
      BlockManager::release_buffer(std::move(send_buffers.back()));
      send_buffers.pop_back();
    }
    if (!sub.q.empty()) {
      BlockManager::release_buffer(sub.wait().data);
    }
  } catch (proteus::internal_error &) {
  }
}

void IBHandler::start() {}

subscription &IBHandler::register_subscriber() { return sub; }

subscription &IBHandler::getNamedSubscription(size_t subs) {
  return sub_named.at(subs);
}

subscription &IBHandler::create_subscription() { return sub_named.add(); }

void IBHandler::unregister_subscriber(subscription &) {}

void IBPoller::handle_wc_failure(const ibv_wc &wc) {
  if (wc.status == IBV_WC_RNR_RETRY_EXC_ERR) {
    LOG(INFO) << "Retry exceeded";
    sleep(1);
    return;
  }
  auto msg = std::string{"completion queue: "} + ibv_wc_status_str(wc.status);
  LOG(INFO) << msg;
  throw std::runtime_error(msg);
}

namespace proteus {
class unexpected_call : public proteus::internal_error {
 public:
  unexpected_call() : internal_error("Unexpected call") {}
};
}  // namespace proteus

class IBSend {
 public:
  static void make() {}
  static void work_completion_event_on_invoker(const ibv_wc &wc) {
    proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
    BlockManager::release_buffer(std::move(data));
  }

  [[noreturn]] static void work_completion_event_on_passive() {
    throw proteus::unexpected_call();
  }
};

void IBPoller::run() {
  set_exec_location_on_scope loc{handler.local_cpu};

  size_t IBV_WC_RDMA_READ_cnt = 0;

  while (true) {
    ibv_wc wc{};
    while (ibv_poll_cq(handler.ibd.getIBImpl().cq, 1, &wc)) {
      if (wc.status != IBV_WC_SUCCESS) {
        handle_wc_failure(wc);
        continue;
      }

      if (wc.opcode & IBV_WC_RECV) {
        //        event_range2 er{this};
        handle_wc_recv(wc);
      } else if (wc.opcode == IBV_WC_SEND) {
        //        event_range2 er{this};
        IBSend::work_completion_event_on_invoker(wc);
      } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
        proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
        if (data) {
          {
            // std::lock_guard<std::mutex> lock{write_promises_m};
            for (size_t i = 0; i < 1; ++i) {
              // assert(!write_promises.empty());
              // assert(write_promises.size() > IBV_WC_RDMA_WRITE_cnt);
              std::pair<subscription, proteus::managed_ptr> *p;
#ifndef NDEBUG
              bool bp =
#endif
                  handler.write_promises.pop(p);
              assert(bp);
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
        } else
          assert(false);
      } else if (wc.opcode == IBV_WC_RDMA_READ) {
        auto start = IBV_WC_RDMA_READ_cnt;
        IBV_WC_RDMA_READ_cnt += wc.wr_id;
        ThreadPool::getInstance().enqueue(
            [this, cnt = (size_t)wc.wr_id, start]() {
              for (size_t i = 0; i < cnt; ++i) {
                auto &p = handler.read_promises.at(start + i);

                std::get<0>(p).publish(std::move(std::get<1>(p)), 0);
                handler.release_buffer(std::move(std::get<2>(p)));
              }
            });
      } else {
        LOG(FATAL) << "Unknown: " << wc.opcode;
      }
    }

    if (ready2exit) {
      LOG(INFO) << "Bailing out...";
      return;
    }
  }
}

void IBPoller::handle_wc_recv(const ibv_wc &wc) {
  proteus::managed_ptr data{reinterpret_cast<void *>(wc.wr_id)};
  if (wc.imm_data == 0 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    if (!handler.saidGoodBye) handler.sendGoodBye();
    LOG(INFO) << "respond good bye";
    ready2exit = true;
    handler.sub.publish(std::move(data), wc.byte_len);
    for (size_t invocation_cnt : invocation_cnts) {
      LOG(INFO) << invocation_cnt;
    }
    return;
  }

  handler.post_recv(handler.ibd.getIBImpl().qp);
  if (wc.imm_data == 1 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    invocation_cnts[0]++;
    //    handler.post_recv(handler.ibd.getIBImpl().qp);
    ThreadPool::getInstance().enqueue([&, data = std::move(data)]() mutable {
      auto databuf = BlockManager::get_buffer();
      {
        std::lock_guard<std::mutex> lock{buffmgmnt};
        BlockManager::release_buffer(std::move(data));
        for (size_t i = 0; i < buffnum; ++i) {
          auto buff = BlockManager::get_buffer();
          ibv_mr *send_buff =
              (--handler.ibd.getIBImpl().reged_mem.upper_bound(buff.get()))
                  ->second;
          ((buffkey *)databuf.get())[i] =
              std::make_pair(buff.get(), send_buff->rkey);
          handler.send_buffers.emplace_back(std::move(buff));
        }
      }
      ibv_sge sge =
          handler.sge_for_buff(databuf.get(), buffnum * sizeof(buffkey));

      handler.send_sge(std::move(databuf), &sge, 1, 2);

      //    handler.post_recv(handler.ibd.getIBImpl().qp);
    });
    return;
  }
  if (wc.imm_data == BUFFRELEASE && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    invocation_cnts[1]++;
    size_t buffs = wc.byte_len / sizeof(proteus::managed_ptr);
    for (size_t i = 0; i < buffs; ++i) {
      BlockManager::release_buffer(
          std::move(((proteus::managed_ptr *)data.get())[i]));
    }

    BlockManager::release_buffer(std::move(data));
    //        handler.post_recv(handler.ibd.getIBImpl().qp);
    return;
  }

  //    handler.post_recv(handler.ibd.getIBImpl().qp);
  if (wc.imm_data == 2 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    //    handler.post_recv(handler.ibd.getIBImpl().qp);
    invocation_cnts[2]++;
    std::unique_lock<std::mutex> lock{handler.pend_buffers_m};

    avail_remote_buffs += buffnum;
    assert(wc.byte_len == buffnum * sizeof(buffkey));
    for (size_t i = 0; i < buffnum; ++i) {
      handler.pend_buffers.emplace_back(((buffkey *)data.get())[i]);
    }
    BlockManager::release_buffer(std::move(data));

    handler.has_requested_buffers = false;
    handler.pend_buffers_cv.notify_all();
  } else if (wc.imm_data == 13 && wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
    invocation_cnts[3]++;
    //        handler.post_recv(handler.ibd.getIBImpl().qp);
    auto h = (std::pair<decltype(handler.reged_remote_mem)::key_type,
                        decltype(handler.reged_remote_mem)::mapped_type> *)
                 data.get();
    handler.reged_remote_mem.emplace(h->first, h->second);
    BlockManager::release_buffer(std::move(data));
  } else {
    if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      //      handler.post_recv(handler.ibd.getIBImpl().qp);
      //      handler.post_recv(handler.ibd.getIBImpl().qp);
      invocation_cnts[4]++;
      std::lock_guard<std::mutex> lock{buffmgmnt};
      BlockManager::release_buffer(std::move(data));
      size_t buffcnt = wc.byte_len / sizeof(packet_t);
      auto *sizes = (packet_t *)handler.send_buffers[wc.imm_data].release();
      invocation_cnts[8] += buffcnt;
      for (size_t i = 0; i < buffcnt; ++i) {
        // LOG(INFO) << sizes[i];
        auto size = sizes[i].first;
        if (size == (-1)) {
          // silent writes are fulfilling the promise immediately,
          // during invocation of the write and asynchronously to/
          // completion.
          handler.send_buffers.front().release();
          handler.send_buffers.pop_front();
          continue;
        } else if (((int64_t)size) >= 0) {
          if (sizes[i].second == 0) {
            handler.sub.publish(std::move(handler.send_buffers.front()), size);
          } else {
            handler.getNamedSubscription(sizes[i].second)
                .publish(std::move(handler.send_buffers.front()), size);
          }
          handler.send_buffers.pop_front();
        } else {  // originated from write_to
          handler.sub.publish(nullptr, -size);
          LOG(INFO) << "this is leaked: " << handler.send_buffers.front().get();
        }
      }
      BlockManager::release_buffer(sizes);
      assert(sizes == handler.send_buffers.front().get() ||
             !handler.send_buffers.front());
      handler.send_buffers.pop_front();
    } else {
      assert(false);
      handler.post_recv(handler.ibd.getIBImpl().qp);
      LOG(INFO) << "outer";
      handler.sub.publish(std::move(data), wc.byte_len);
      //  handler.post_recv(handler.ibd.getIBImpl().qp);
    }
  }
}

static std::atomic<size_t> send_cnt = 0;

int IBHandler::send(ibv_send_wr &wr, ibv_send_wr **save_on_error, bool retry) {
  int ret;
  int i = 0;
  while ((ret = ibv_post_send(ibd.getIBImpl().qp, &wr, save_on_error)) != 0) {
    if (ret != ENOMEM) return ret;
    if (i++ % 10000 == 0) {
      LOG(INFO) << "Sleeping: " << strerror(ret) << " " << wr.send_flags << " "
                << wr.opcode << " " << send_cnt << " " << wr.imm_data;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }

  if (++send_cnt % 256 == 0 && !(wr.send_flags & IBV_SEND_SIGNALED) &&
      !saidGoodBye) {
    flush_read();
    flush();
  }

  return 0;
}

int IBHandler::send(ibv_send_wr &wr, bool retry) {
  ibv_send_wr *save_on_error = nullptr;
  return send(wr, &save_on_error, retry);
}

void IBHandler::send_sge(proteus::managed_ptr release_upon_send,
                         ibv_sge *sg_list, size_t sge_cnt,
                         decltype(ibv_send_wr::imm_data) imm) {
  ibv_send_wr wr{/* 0 everything out via value initialization */};

  reinterpret_cast<proteus::managed_ptr &>(wr.wr_id) =
      std::move(release_upon_send);  // readable locally on work completion
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.sg_list = sg_list;
  wr.num_sge = sge_cnt;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
  wr.imm_data = imm;  // Can pass an arbitrary value to receiver, consider for
  // the consumed buffer

  linux_run(send(wr));
  // TODO: save_on_error contains a pointer to the "bad" request, handle
  // more gracefully
}

void IBHandler::sendGoodBye() {
  //  flush_read();
  //  flush();
  saidGoodBye = true;
  send_sge(nullptr, nullptr, 0, 0);
}

void IBHandler::flush() {
  std::lock_guard<std::mutex> lock{write_promises_m};
  flush_write();
}

void IBHandler::flush_read() {
  if (reads == 0) return;

  ibv_send_wr wr{/* 0 everything out via value initialization */};

  {
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.wr.rdma.remote_addr =
        reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(nullptr);
    wr.wr.rdma.rkey = 0;

    std::lock_guard<std::mutex> lock{read_promises_m};

    wr.wr_id =
        (uintptr_t)reads.exchange(0);  // readable locally on work completion
  }
  // LOG(INFO) << bytes << " " << buff << " " << data;
  linux_run(send(wr));
}

static void *old_cnts = nullptr;

decltype(IBHandler::read_promises)::value_type &IBHandler::create_promise(
    proteus::managed_ptr buff, proteus::remote_managed_ptr from) {
  assert(!read_promises_m.try_lock());
  auto &pr = read_promises.add();
  std::get<1>(pr) = std::move(buff);
  std::get<2>(pr) = std::move(from);
  return pr;
}

decltype(IBHandler::write_promises)::value_type IBHandler::create_write_promise(
    void *buff) {
  assert(!write_promises_m.try_lock());
  auto ptr = new std::pair<subscription, proteus::managed_ptr>(6, buff);
  LOG_IF(WARNING, saidGoodBye) << "Adding write_promise: " << ptr;
  write_promises.push(ptr);
  // write_promises.emplace_back(5, buff);
  return ptr;
}

subscription *IBHandler::read(proteus::remote_managed_ptr data, size_t bytes) {
  //  eventlogger.log(this, IB_CREATE_RDMA_READ_START);
  auto buff = BlockManager::get_buffer();

  ibv_sge sge = sge_for_buff(buff.get(), bytes);
  ibv_send_wr wr{/* 0 everything out via value initialization */};
  auto &p = [&]() -> auto & {
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    wr.wr.rdma.remote_addr =
        reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(data.get());
    wr.wr.rdma.rkey = (--reged_remote_mem.upper_bound(data.get()))->second;

    std::lock_guard<std::mutex> lock{read_promises_m};

    // FIXME: there is certainly a race condition here, as the
    //  readcnt/promise creation and the send is outside the critical (locked)
    //  section
    auto rid = ++reads;

    wr.wr_id = (uintptr_t)rid;  // readable locally on work completion
    wr.send_flags = ((rid % 64) ? 0 : IBV_SEND_SIGNALED);

    if (wr.send_flags & IBV_SEND_SIGNALED) reads = 0;

    // create promise before submitting the work request
    return create_promise(std::move(buff), std::move(data));
  }
  ();
  linux_run(send(wr));

  //  eventlogger.log(this, IB_CREATE_RDMA_READ_END);
  return &std::get<0>(p);
}

void IBHandler::write_to(proteus::managed_ptr data, size_t bytes,
                         buffkey buff) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  // get_buffer();
  cnts[write_cnt++] = std::make_pair(-bytes, 0);
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  write_to_int(std::move(data), bytes, buff);
}

void IBHandler::write(proteus::managed_ptr data, size_t bytes, size_t sub_id) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  auto buff = get_buffer();
  cnts[write_cnt++] = std::make_pair(bytes, sub_id);
  actual_writes++;
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  write_to_int(std::move(data), bytes, buff);
}

subscription *IBHandler::write_silent(proteus::managed_ptr data, size_t bytes) {
  std::lock_guard<std::mutex> lock{write_promises_m};
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  auto buff = get_buffer();
  cnts[write_cnt++] = std::make_pair(-1, 0);
  actual_writes++;
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  return write_to_int(std::move(data), bytes, buff, buff.first);
}

ibv_sge IBHandler::sge_for_buff(void *data, size_t bytes) {
  ibv_mr *send_reg = (--ibd.getIBImpl().reged_mem.upper_bound(data))->second;
  ibv_sge sge{/* 0 everything out via value initialization */};

  static_assert(sizeof(void *) == sizeof(decltype(ibv_sge::addr)));
  // super dangerous cast, do not remove above static_assert!
  sge.addr = reinterpret_cast<decltype(ibv_sge::addr)>(data);
  assert(bytes <= std::numeric_limits<decltype(ibv_sge::length)>::max());
  sge.length = static_cast<decltype(ibv_sge::length)>(bytes);
  sge.lkey = send_reg->lkey;
  return sge;
}

void IBHandler::flush_write() {
  if (write_cnt == 0) return;

  auto buff = get_buffer();
  void *buffpromise = nullptr;
  packet_t *data = cnts;
  BlockManager::release_buffer(proteus::managed_ptr{old_cnts});
  // old_cnts = cnts;
  size_t bytes = write_cnt * sizeof(packet_t);
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_START);
  cnts = (packet_t *)BlockManager::get_buffer().release();
  //  eventlogger.log(this, IB_RDMA_WAIT_BUFFER_END);

  {
    assert(!write_promises_m.try_lock());
    // std::unique_lock<std::mutex> lock{write_promises_m};
    ibv_sge sge = sge_for_buff(data, bytes);

    create_write_promise(buffpromise);

    ibv_send_wr wr{/* 0 everything out via value initialization */};

    wr.wr_id = (uintptr_t)data;  // readable locally on work completion
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.imm_data = actual_writes;  // Unused

    wr.wr.rdma.remote_addr =
        reinterpret_cast<decltype(wr.wr.rdma.remote_addr)>(buff.first);
    wr.wr.rdma.rkey = buff.second;

    linux_run(send(wr));
    // TODO: save_on_error contains a pointer to the "bad" request, handle
    //  more gracefully

    write_cnt = 0;
    actual_writes = 0;
  }
}

subscription *IBHandler::write_to_int(proteus::managed_ptr data, size_t bytes,
                                      buffkey buff, void *buffpromise) {
  assert(!write_promises_m.try_lock());
  bool should_flush;
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
    // FIXME: THIS IS A LEAK
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
    //  more gracefully

    ret = &p->first;
    should_flush = ((write_cnt % packNotifications) == 0);
  }

  if (should_flush) flush_write();

  return ret;
}

void IBHandler::request_buffers_unsafe() { send_sge(nullptr, nullptr, 0, 1); }

void IBHandler::release_buffer(proteus::remote_managed_ptr p) {
  if (!p) return;
  std::lock_guard<std::mutex> lock{m3};
  // TODO: batch releases

  // TODO: support sharing of remote buffers

  released[released_cnt++] = p.release();
  if (released_cnt == 128) {
    auto databuf = BlockManager::get_buffer();
    size_t relnum = released_cnt;
    for (size_t i = 0; i < relnum; ++i) {
      ((void **)databuf.get())[i] = released[i];
    }

    ThreadPool::getInstance().enqueue(
        [=, this, databuf = std::move(databuf)]() mutable {
          ibv_sge sge = sge_for_buff(databuf.get(), relnum * sizeof(void *));

          send_sge(std::move(databuf), &sge, 1, BUFFRELEASE);
        });
    released_cnt = 0;
  }

  //
  //    // if key exists, this is a remote buffer and we have ownership
  //    // FIXME: what about shared buffers? We may not have ownership
  ////    pend_buffers.emplace_back(p.get(), k);
  ////    pend_buffers_cv.notify_all();
  //    //    LOG(WARNING) << "RELEASE leak: " << ++leak_release;
  //  } catch (std::out_of_range &) {
  //        LOG(WARNING) << "RELEASE leak: " << ++leak_release;
  //    // The ptr didn't come from the buffer manager, ignore
  //  }
  //  ((void)/* Release and ignore */ p.release());
}

size_t get_avail_remote_buffs() { return avail_remote_buffs; }

buffkey IBHandler::get_buffer() {
  std::unique_lock<std::mutex> lock{pend_buffers_m};
  if (pend_buffers.size() == ((size_t)(buffnum * 0.2)) &&
      !has_requested_buffers) {
    if (!has_requested_buffers.exchange(true)) {
      ThreadPool::getInstance().enqueue(&IBHandler::request_buffers_unsafe,
                                        this);
    }
  }

  if (pend_buffers.empty()) {
    pend_buffers_cv.wait(lock, [&]() { return !pend_buffers.empty(); });
  }

  assert(avail_remote_buffs-- == pend_buffers.size());
  auto b = pend_buffers.front();
  pend_buffers.pop_front();
  keys.emplace(b.first, b.second);
  // LOG(INFO) << b.first;
  return b;
}

void IBHandler::send(proteus::managed_ptr data, size_t bytes,
                     decltype(ibv_send_wr::imm_data) imm) {
  ibv_sge sge = sge_for_buff(data.get(), bytes);

  send_sge(std::move(data), &sge, 1, imm);
}

void IBHandler::disconnect() {
  // Say good bye, otherwise remote will crash waiting for a recv
  LOG(INFO) << "send good bye";
  sendGoodBye();
  LOG(INFO) << "waiting for confirmation";
  // FIXME: not really, should wait on something more general...
  BlockManager::release_buffer(sub.wait().data);
  LOG(INFO) << "done!";
}

void IBHandler::reg(const void *mem, size_t bytes) {
  ibd.getIBImpl().reg(mem, bytes);
}

buffkey IBHandler::reg2(const void *mem, size_t bytes) {
  reg(mem, bytes);

  auto x =
      std::make_pair(mem, ibd.getIBImpl().reged_mem.find(mem)->second->rkey);
  auto f = BlockManager::get_buffer();
  *((decltype(&x))f.get()) = x;
  send(std::move(f), sizeof(x), 13);
  return std::make_pair((void *)mem,
                        ibd.getIBImpl().reged_mem.find(mem)->second->rkey);
}

void IBHandler::unreg(const void *mem) { ibd.getIBImpl().unreg(mem); }
