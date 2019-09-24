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

#include <map>

#include "infiniband-manager.hpp"
#include "topology/topology.hpp"
#include "util/memory-registry.hpp"

struct ib_addr {
  ibv_gid gid;
  uint16_t lid;
  uint32_t psn;
  uint32_t qpn;
};

std::ostream &operator<<(std::ostream &out, const ibv_gid &gid);
std::ostream &operator<<(std::ostream &out, const ib_addr &addr);

class IBHandler : public MemoryRegistry {
 protected:
  subscription sub;

  std::map<uint64_t, void *> active_connections;

  std::mutex m;
  std::condition_variable cv;

  std::mutex m_reg;
  std::condition_variable cv_reg;
  std::map<void *, ibv_mr *> reged_mem;

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

  subscription buffers;

  size_t write_cnt;
  size_t *cnts;

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

  void unregister_subscriber(subscription &);

  /**
   *
   * Notes: Do not overwrite as its called from constructor
   */
  virtual void *reg(void *mem, size_t bytes) final;
  virtual void unreg(void *mem) final;

  void flush();

 private:
  void poll_cq();

  void sendGoodBye();

  void flush_write(ibv_sge *sge_ptr);
  void flush_write();

  void send_sge(uintptr_t wr_id, ibv_sge *sg_list, size_t sge_cnt,
                decltype(ibv_send_wr::imm_data) imm);

  void request_buffers_unsafe();

 public:
  void send(void *data, size_t bytes, decltype(ibv_send_wr::imm_data) imm = 5);

  void write(void *data, size_t bytes, decltype(ibv_send_wr::imm_data) imm = 7);

  typedef std::pair<void *, decltype(ibv_send_wr::wr.rdma.rkey)> buffkey;
  buffkey get_buffer();

  void disconnect();
};
