/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_THREADSAFE_SET_HPP
#define PROTEUS_THREADSAFE_SET_HPP

#include <platform/memory/allocator.hpp>
#include <platform/util/logging.hpp>
/**
 *
 * @tparam N slots in the ring buffer. Prefer powers of two.
 */
template <typename T, size_t N, typename index_t = uint32_t>
class UnsafeRingBuffer {
  class Entry {
   public:
    /**
     * odd  value => free
     * even value => filled
     */
    std::atomic<index_t> marker{index_t(-2)};
    T data;
  };

  //  std::atomic<index_t> head{0};  // only odd values
  std::atomic<index_t> head{0};  // only odd values
  std::atomic<index_t> tail{0};  // only odd values
  std::vector<Entry, proteus::memory::PinnedMemoryAllocator<Entry>> slots;

 public:
  UnsafeRingBuffer() : slots(N) {}

  template <typename... Args>
  void emplace(Args &&...args) noexcept {
    // Assumes single producer
    index_t h = (head += 2) - 2;
    while (slots[(h / 2) % N].marker & 1) std::this_thread::yield();
    slots[(h / 2) % N].data = T{std::forward<Args>(args)...};
    slots[(h / 2) % N].marker = h + 1;
  }

  T pop() noexcept {
    index_t t = (tail += 2) - 2;
    while (slots[(t / 2) % N].marker != t + 1) std::this_thread::yield();
    T tmp = std::move(slots[(t / 2) % N].data);
    slots[(t / 2) % N].marker = t;
    return tmp;
  }

  bool empty_unsafe() const { return head == tail; }
  size_t size_unsafe() const { return (head - tail) / 2; }
};

template <typename T>
struct alignas(128) EntryAQMPMC {
  T first;
  std::atomic<size_t> second;
};

template <typename T>
class alignas(2 * 1024 * 1024)
    /* TODO: also consider 2M, as it gives better perf for some queries */
    AsyncQueueMPMC { /* SPMC */
 private:
  std::atomic<size_t> occ{0};
  size_t k4[15];
  std::atomic<size_t> occrev{0};
  size_t k[5];
  std::vector<
      EntryAQMPMC<T> /*,
      proteus::memory::ExplicitSocketPinnedMemoryAllocator<EntryAQMPMC<T>>*/>
      data;
  size_t k2[5];
  //  std::stack<
  //      T, std::deque<T,
  //      proteus::memory::ExplicitSocketPinnedMemoryAllocator<T>>> data;
  std::atomic<bool> terminating = false;
  size_t k3[19];
  std::mutex m;
  std::condition_variable cv;

 public:
  static constexpr size_t N = 8 * 1024;
  std::atomic<size_t> cnt;
  AsyncQueueMPMC(int socket_id)
      : data(64 * 1024 /*, proteus::memory::ExplicitSocketPinnedMemoryAllocator<T>{socket_id}*/),
        cnt(0) {
    for (size_t i = 0; i < N; ++i) {
      data[i].second = -1;
    }
  }

  [[nodiscard]] bool empty_unsafe() const noexcept { return occ == occrev; }
  [[nodiscard]] auto size_unsafe() const noexcept {
    size_t size = occ - occrev;
    return size > 2 * data.size() ? /* producers wait for jobs */ 0 : size;
  }

  [[nodiscard]] bool empty() noexcept {
    std::lock_guard<std::mutex> lock{m};
    return empty_unsafe();
  }

  template <typename... Args>
  void emplace(Args &&...args) noexcept {
    {
      //      std::lock_guard<std::mutex> lock{m};
      auto locc = occ++;
      while (occrev + N <= locc) std::this_thread::yield();
      //      while (data.at(locc % N).second == locc - N);
      data.at(locc % N).first = T{std::forward<Args>(args)...};
      data.at(locc % N).second = locc;
      //      data.push(std::forward<Args>(args)...);
      //      ++occ;
    }
    //    ++occ;
    //    cv.notify_all();
    ++cnt;
  }

  void push(T t) noexcept { emplace(std::move(t)); }

  void reset() noexcept {
    assert(terminating);
    T x;
    while (pop(x))
      ;
    terminating = false;
    //    occ = 0;
    //    occrev = 0;
    occrev = occ.load();
    cnt = occ.load();
  }

  void close() noexcept {
    terminating = true;
    cv.notify_all();
  }

  bool withPoppedItemDo(const std::function<void(T)> &f) {
    auto tmp = pop();
    if (!tmp.has_value()) return false;
    f(std::move(tmp.value()));
    return true;
  }

  void foreachItemDo(const std::function<void(T)> &f) {
    while (withPoppedItemDo(f))
      ;
  }

  template <typename F, range_log_op = std::invoke_result_t<F>::event_type>
  bool withPoppedItemDo(const F &efactory, const std::function<void(T)> &f) {
    auto tmp = [&]() {
      auto event = efactory();
      return pop();
    }();
    if (!tmp.has_value()) return false;
    f(std::move(tmp.value()));
    return true;
  }

  template <typename F, range_log_op = std::invoke_result_t<F>::event_type>
  void foreachItemDo(const F &efactory, const std::function<void(T)> &f) {
    while (withPoppedItemDo<F>(efactory, f))
      ;
  }

  std::optional<T> pop() noexcept {
    auto reserve = occrev++;
    auto &ref = data.at(reserve % N);

    do {
      if (ref.second == reserve) {
        return {std::move(ref.first)};
      } else {
        std::this_thread::yield();
      }
    } while (!terminating || occ > reserve);
    assert(terminating);
    return {};
  }

  bool pop(T &res) noexcept {
    auto tmp = pop();
    if (tmp) res = std::move(tmp.value());
    return tmp.has_value();
  }

  bool pop2(T &res) noexcept { return pop(res); }
};

template <typename T>
class threadsafe_set { /* SPMC */
 private:
  UnsafeRingBuffer<T, 64 * 1024, uint32_t> buffer;

 public:
  [[nodiscard]] bool empty_unsafe() const noexcept {
    return buffer.empty_unsafe();
  }
  [[nodiscard]] auto size_unsafe() const noexcept {
    return buffer.size_unsafe();
  }

  template <typename... Args>
  void emplace(Args &&...args) noexcept {
    buffer.emplace(std::forward<Args>(args)...);
  }

  T pop() noexcept { return buffer.pop(); }
};

#endif  // PROTEUS_THREADSAFE_SET_HPP
