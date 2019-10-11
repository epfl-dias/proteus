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

#include <infiniband/verbs.h>

#include <future>
#include <map>

#include "infiniband-manager.hpp"
#include "topology/topology.hpp"
#include "util/async_containers.hpp"
#include "util/memory-registry.hpp"

struct ib_addr {
  ibv_gid gid;
  uint16_t lid;
  uint32_t psn;
  uint32_t qpn;
};

std::ostream &operator<<(std::ostream &out, const ibv_gid &gid);
std::ostream &operator<<(std::ostream &out, const ib_addr &addr);

typedef std::pair<size_t, size_t> packet_t;

class IBHandler : public MemoryRegistry {
  static_assert(
      std::is_same_v<buffkey::second_type, decltype(ibv_send_wr::wr.rdma.rkey)>,
      "wrong buffkey type");

 protected:
  subscription sub;
  std::deque<subscription> sub_named;
  std::atomic<size_t> subcnts = 1;

  std::map<uint64_t, void *> active_connections;

  std::mutex m;
  std::condition_variable cv;

  std::mutex m_reg;
  std::condition_variable cv_reg;
  std::map<const void *, ibv_mr *> reged_mem;
  std::map<const void *, decltype(ibv_mr::rkey)> reged_remote_mem;

  std::thread listener;

  ibv_pd *pd;
  ibv_comp_channel *comp_channel;
  ibv_cq *cq;

  std::thread poller;
  ibv_context *verbs;

  size_t pending;

  bool saidGoodBye = false;

  int dev_cnt;

  // New attributes
  ibv_context *context;
  const topology::cpunumanode &local_cpu;

  ibv_qp *qp;

  // ibv_pd *pd;
  // ibv_comp_channel *comp_channel;
  // ibv_cq *cq;
  ib_addr addr;

  uint8_t ib_port;
  uint8_t ib_sl;
  uint8_t ib_gidx;

  ib_addr rem_addr;

  std::deque<void *> b;
  std::deque<buffkey> pend_buffers;
  std::mutex pend_buffers_m;
  std::condition_variable pend_buffers_cv;

  // subscription buffers;

  // std::deque<std::pair<std::promise<void *>, void *>> read_promises;
  std::deque<std::pair<subscription, void *>> read_promises;
  std::mutex read_promises_m;

  AsyncQueueSPSC<std::pair<subscription, void *> *> write_promises;
  // std::deque<std::pair<subscription, void *>> write_promises;
  std::mutex write_promises_m;

  size_t write_cnt;
  size_t actual_writes;
  packet_t *cnts;

  bool has_requested_buffers;

 protected:
  void post_recv(ibv_qp *qp);

  void run();

  int send(ibv_send_wr &wr, ibv_send_wr **save_on_error, bool retry = true);
  int send(ibv_send_wr &wr, bool retry = true);

 public:
  explicit IBHandler(int cq_backlog);

  virtual ~IBHandler();

  void start();

  subscription &register_subscriber();
  subscription &create_subscription();

  void unregister_subscriber(subscription &);

  /**
   *
   * Notes: Do not overwrite as its called from constructor
   */
  virtual void reg(const void *mem, size_t bytes) final;
  virtual buffkey reg2(const void *mem, size_t bytes) final;
  virtual void unreg(const void *mem) final;

  void flush();
  void flush_read();

 private:
  void poll_cq();

  void sendGoodBye();
  decltype(read_promises)::value_type &create_promise(void *buff);
  decltype(write_promises)::value_type create_write_promise(void *buff);

  void flush_write(ibv_sge *sge_ptr);
  void flush_write();
  subscription *write_to_int(void *data, size_t bytes, buffkey dst,
                             void *buffpromise = nullptr);

  void send_sge(uintptr_t wr_id, ibv_sge *sg_list, size_t sge_cnt,
                decltype(ibv_send_wr::imm_data) imm);

  void request_buffers_unsafe();

 public:
  void send(void *data, size_t bytes, decltype(ibv_send_wr::imm_data) imm = 5);

  void write(void *data, size_t bytes, size_t sub_id = 0);
  void write_to(void *data, size_t bytes, buffkey dst);
  [[nodiscard]] subscription *write_silent(void *data, size_t bytes);
  [[nodiscard]] subscription *read(void *data, size_t bytes);
  [[nodiscard]] subscription *read_event();

  buffkey get_buffer();

  void disconnect();
};
