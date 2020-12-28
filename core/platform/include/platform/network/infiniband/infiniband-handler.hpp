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
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/async_containers.hpp>
#include <platform/util/memory-registry.hpp>
//
// struct ib_addr {
//  ibv_gid gid;
//  uint16_t lid;
//  uint32_t psn;
//  uint32_t qpn;
//};

std::ostream &operator<<(std::ostream &out, const ibv_gid &gid);
std::ostream &operator<<(std::ostream &out, const ib_addr &addr);

typedef std::pair<size_t, size_t> packet_t;

class IBPoller;

template <typename T>
class threadsafe_autoresizeable_deque {
  static constexpr size_t dqpage = 1024;

  std::atomic<size_t> size = 0;
  std::atomic<int> active = 0;
  std::mutex resize_m;

  std::deque<std::shared_ptr<std::deque<T>>> data[2];

  void resize(size_t i) {
    std::lock_guard<std::mutex> lock{resize_m};
    while (i >= size) {
      auto old_active = active.load();
      auto new_active = 1 - old_active;
      data[new_active].emplace_back(std::make_shared<std::deque<T>>());
      auto &v = *data[new_active].back();
      for (size_t j = 0; j < dqpage; ++j) {
        v.emplace_back(size + j);
      }
      active = new_active;
      size += dqpage;
      data[old_active].emplace_back(data[new_active].back());
    }
  }

 public:
  T &at(size_t i) {
    if (i >= size) resize(i);
    return (*data[active][i / dqpage])[i % dqpage];
  }
};

class IBHandler : public MemoryRegistry {
  static_assert(
      std::is_same_v<buffkey::second_type, decltype(ibv_send_wr::wr.rdma.rkey)>,
      "wrong buffkey type");

 protected:
  friend class IBPoller;
  std::mutex m_sub_named;
  subscription sub;
  threadsafe_autoresizeable_deque<subscription> sub_named;
  std::atomic<size_t> subcnts;

  std::map<uint64_t, void *> active_connections;

  std::mutex m;
  std::condition_variable cv;

  std::mutex m_reg;
  std::condition_variable cv_reg;

  std::map<const void *, decltype(ibv_mr::rkey)> reged_remote_mem;

  std::thread listener;

  const ib &ibd;

  std::thread poller;
  std::thread flusher;

  size_t pending;

  std::atomic<bool> saidGoodBye = false;

  int dev_cnt;

  subscription &getNamedSubscription(size_t subs);
  // New attributes
  //  ibv_context *context;
  const topology::cpunumanode &local_cpu;

  ibv_sge sge_for_buff(void *data, size_t bytes);
  //  ib_addr rem_addr;

  std::deque<proteus::managed_ptr> send_buffers;
  std::deque<buffkey> pend_buffers;
  std::mutex pend_buffers_m;
  std::condition_variable pend_buffers_cv;

  // subscription buffers;

  // std::deque<std::pair<std::promise<void *>, void *>> read_promises;
  std::deque<std::tuple<subscription, proteus::managed_ptr,
                        proteus::remote_managed_ptr>>
      read_promises;
  std::mutex read_promises_m;

  AsyncQueueSPSC<std::pair<subscription, proteus::managed_ptr> *>
      write_promises;
  // std::deque<std::pair<subscription, void *>> write_promises;
  std::mutex write_promises_m;

  size_t write_cnt;
  size_t actual_writes;
  packet_t *cnts;

  std::atomic<bool> has_requested_buffers;

  std::unique_ptr<IBPoller> poll_cq;

 protected:
  void post_recv(ibv_qp *qp);

  int send(ibv_send_wr &wr, ibv_send_wr **save_on_error, bool retry = true);
  int send(ibv_send_wr &wr, bool retry = true);

 public:
  explicit IBHandler(int cq_backlog, const ib &ibd);

  ~IBHandler() override;

  void start();

  subscription &register_subscriber();
  subscription &create_subscription();

  void unregister_subscriber(subscription &);

  /**
   *
   * Notes: Do not overwrite as its called from constructor
   */
  void reg(const void *mem, size_t bytes) final;
  buffkey reg2(const void *mem, size_t bytes);
  void unreg(const void *mem) final;

  void flush();
  void flush_read();

 private:
  void sendGoodBye();
  decltype(read_promises)::value_type &create_promise(
      proteus::managed_ptr buff, proteus::remote_managed_ptr from);
  decltype(write_promises)::value_type create_write_promise(void *buff);

  void flush_write();
  subscription *write_to_int(proteus::managed_ptr data, size_t bytes,
                             buffkey dst, void *buffpromise = nullptr);

  void send_sge(proteus::managed_ptr release_upon_send, ibv_sge *sg_list,
                size_t sge_cnt, decltype(ibv_send_wr::imm_data) imm);

  void request_buffers_unsafe();

 public:
  void send(proteus::managed_ptr data, size_t bytes,
            decltype(ibv_send_wr::imm_data) imm = 5);

  /**
   * Write local data to remote memory based on a subscription.
   *
   * The InfiniBand managers are responsible for providing the remote
   * memory.
   * @p data is passed by value, signaling that ownership of the
   * source data is transferred to the manager, who will deallocate
   * it as soon as it deems appropriate. No writes are allowed to
   * happen on an alias of this pointer.
   *
   * The function may return before write completes.
   *
   * @param data    A proteus manager pointer to the source data
   * @param bytes   Bytes to copy, must be less that BlockManager#block_size
   * @param sub_id  Target subscription
   */
  void write(proteus::managed_ptr data, size_t bytes, size_t sub_id = 0);
  void write_to(proteus::managed_ptr data, size_t bytes, buffkey dst);
  [[nodiscard]] subscription *write_silent(proteus::managed_ptr data,
                                           size_t bytes);
  /**
   * Request remote data read.
   *
   * The function may return before read is complete. Wait on the
   * subscription to block until the results are ready.
   *
   * The ownership of data is handed over to the handler who will
   * release the pointer on will.
   *
   * @param data    Pointer in remote server
   * @param bytes   Number of bytes to read
   * @return        Subscription to block for the results of this operation
   */
  [[nodiscard]] subscription *read(proteus::remote_managed_ptr data,
                                   size_t bytes);
  [[nodiscard]] subscription *read_event();

  buffkey get_buffer();
  void release_buffer(proteus::remote_managed_ptr p);

  void disconnect();
};

size_t getIBCnt();
