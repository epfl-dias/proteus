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

#ifndef INFINIBAND_MANAGER_HPP_
#define INFINIBAND_MANAGER_HPP_

#include <arpa/inet.h>
#include <err.h>
#include <glog/logging.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <platform/common/common.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/network/infiniband/remote-managed-pointer.hpp>
#include <queue>
#include <thread>

namespace errorhanding {
[[noreturn]] inline void failedRDMARun(const char *str, const char *file,
                                       int line, int x) {
  auto msg = std::string{str} + " failed (" + gai_strerror(x) + ")";
  google::LogMessage(file, line, google::GLOG_ERROR).stream() << msg;
  throw std::runtime_error{msg};
}

inline void assertRDMARun(int x, const char *str, const char *file, int line) {
  if (unlikely(x)) failedRDMARun(str, file, line, x);
}

template <typename T>
inline T *assertRDMARun(T *x, const char *str, const char *file, int line) {
  if (unlikely(x == nullptr)) failedRDMARun(str, file, line, x);
  return x;
}
}  // namespace errorhanding

#define rdma_run(x) (errorhanding::assertRDMARun(x, #x, __FILE__, __LINE__))

class IBHandler;

typedef std::pair<void *, uint32_t> buffkey;

constexpr size_t cq_ack_backlog = 1024;

class subscription {
  class value_type {
   public:
    proteus::managed_ptr data;
    size_t size;
#ifndef NDEBUG
    decltype(__builtin_FILE()) file;
    decltype(__builtin_LINE()) line;
#endif

    value_type(proteus::managed_ptr data, size_t size
#ifndef NDEBUG
               ,
               decltype(__builtin_FILE()) file, decltype(__builtin_LINE()) line
#endif
               )
        : data(std::move(data)), size(size), file(file), line(line) {
    }
    value_type(const value_type &) = delete;
    value_type &operator=(const value_type &) = delete;
    value_type &operator=(value_type &&) = delete;
    value_type(value_type &&other) noexcept
        : value_type(std::move(other.data), other.size
#ifndef NDEBUG
                     ,
                     other.file, other.line
#endif
          ) {
      other.data = nullptr;
    }
    ~value_type();

    proteus::managed_ptr release() { return std::move(data); }
  };

  std::mutex m;
  std::condition_variable cv;

 public:
  std::queue<value_type> q;

 public:
  size_t id;
  subscription(size_t id = 0) : id(id) {}

  value_type wait();

  void publish(proteus::managed_ptr data, size_t size
#ifndef NDEBUG
               ,
               decltype(__builtin_FILE()) file = __builtin_FILE(),
               decltype(__builtin_LINE()) line = __builtin_LINE()
#endif
  );
};

class InfiniBandManager {
 public:
  static void init(const std::string &url, uint16_t port = 12345,
                   bool primary = false, bool ipv4 = false);
  static void send(void *data, size_t bytes);

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
  static void write(proteus::managed_ptr data, size_t bytes, size_t sub_id = 0);
  static void write_to(proteus::managed_ptr data, size_t bytes, buffkey b);
  [[nodiscard]] static subscription *write_silent(proteus::managed_ptr data,
                                                  size_t bytes);
  /**
   * Request remote data read.
   *
   * The function may return before read is complete. Wait on the
   * subscription to block until the results are ready.
   *
   * @param data    Pointer in remote server
   * @param bytes   Number of bytes to read
   * @return        Subscription to block for the results of this operation
   */
  [[nodiscard]] static subscription *read(proteus::remote_managed_ptr data,
                                          size_t bytes);
  [[nodiscard]] static subscription *read_event();
  static void flush();
  static void flush_read();
  static buffkey get_buffer();
  static void release_buffer(proteus::remote_managed_ptr p);
  static void disconnectAll();
  static void deinit();

  static subscription &subscribe();
  static void unsubscribe(subscription &);
  static subscription &create_subscription();

  static buffkey reg(const void *mem, size_t bytes);
  static void unreg(const void *mem);

  static uint64_t server_id();

 private:
  static std::vector<IBHandler *> ib;
  static uint64_t srv_id;
};

#endif /* INFINIBAND_MANAGER_HPP_ */
