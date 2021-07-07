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

#include <benchmark/benchmark.h>

#include <platform/util/glog.hpp>

#include "lib/util/datastructures/threadsafe-set.hpp"

class QueueBench : public ::benchmark::Fixture {};
///**
// *
// * @tparam N slots in the ring buffer. Prefer powers of two.
// */
// template <typename T, size_t N, typename index_t = uint32_t>
// class UnsafeRingBuffer {
//  class alignas(64) Entry {
//   public:
//    /**
//     * odd  value => free
//     * even value => filled
//     */
//    std::atomic<index_t> marker;
//    T data;
//  };
//
//  //  std::atomic<index_t> head{0};  // only odd values
//  index_t head{0};               // only odd values
//  std::atomic<index_t> tail{0};  // only odd values
//  std::vector<Entry> slots;
//
// public:
//  UnsafeRingBuffer() : slots(N) {
//    for (size_t i = 0; i < N; ++i) {
//      slots[i].marker = index_t(-2);
//    }
//  }
//
//  template <typename... Args>
//  void emplace(Args &&...args) noexcept {
//    // Assumes single producer
//    index_t h = (head += 2) - 2;
//    while (slots[h % N].marker & 1)
//      ;
//    slots[h % N].data = T{std::forward<Args>(args)...};
//    slots[h % N].marker = h + 1;
//  }
//
//  T pop() noexcept {
//    index_t t = (tail += 2) - 2;
//    while (slots[t % N].marker != t + 1)
//      ;
//    T tmp = std::move(slots[t % N].data);
//    slots[t % N].marker = t;
//    return tmp;
//  }
//
//  bool empty_unsafe() const { return head == tail; }
//  size_t size_unsafe() const { return (head - tail) / 2; }
//};

// template <typename T>
// class threadsafe_set { /* SPMC */
//  private:
//   std::stack<T> data;
//   std::mutex m;
//   std::condition_variable cv;
//
//  public:
//   [[nodiscard]] bool empty_unsafe() const noexcept { return data.empty(); }
//   [[nodiscard]] auto size_unsafe() const noexcept { return data.size(); }
//
//   [[nodiscard]] bool empty() noexcept {
//     std::lock_guard<std::mutex> lock{m};
//     return empty_unsafe();
//   }
//
//   template <typename... Args>
//   void emplace(Args &&...args) noexcept {
//     // FIXME: seems like emplace starves pop!
//     event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END> er(this);
//     {
//       std::lock_guard<std::mutex> lock{m};
//       data.push(std::forward<Args>(args)...);
//     }
//     cv.notify_one();
//   }
//
//   T pop() noexcept {
//     std::unique_lock<std::mutex> lock{m};
//     while (empty_unsafe()) {
//       event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END> er(this);
//       cv.wait(lock, // std::chrono::milliseconds{1},
//                   [this]() { return !empty_unsafe(); });
//       LOG_IF_EVERY_N(INFO, empty_unsafe(), 1000)
//           << "woke up to check, but is empty";
//     }
//     T v = std::move(data.top());
//     data.pop();
//     return std::move(v);
//   }
// };
//
// template <typename T>
// class threadsafe_set { /* SPMC */
// private:
//  UnsafeRingBuffer<T, 1024, uint32_t> buffer;
//
// public:
//  [[nodiscard]] bool empty_unsafe() const noexcept {
//    return buffer.empty_unsafe();
//  }
//  [[nodiscard]] auto size_unsafe() const noexcept {
//    return buffer.size_unsafe();
//  }
//
//  template <typename... Args>
//  void emplace(Args &&...args) noexcept {
//    // FIXME: seems like emplace starves pop!
//    //    event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END> er(this);
//    buffer.emplace(std::forward<Args>(args)...);
//  }
//
//  T pop() noexcept {
//    //    event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END> er(this);
//    return buffer.pop();
//  }
//};
//
// BENCHMARK_F(QueueBench, CreateAndInsertToQueue)(benchmark::State &state) {
//  for (auto _ : state) {
//    threadsafe_set<int> q;
//    int val = 5;
//    q.emplace(val);
//  }
//}

#include <barrier>
#include <olap/common/olap-common.hpp>
#include <stack>
#include <thread>

//
// template <typename T>
// class threadsafe_set2 { /* SPMC */
// private:
//  std::stack<T> data;
//  std::mutex m;
//  std::condition_variable cv;
//
// public:
//  [[nodiscard]] bool empty_unsafe() const noexcept { return data.empty(); }
//  [[nodiscard]] auto size_unsafe() const noexcept { return data.size(); }
//
//  [[nodiscard]] bool empty() noexcept {
//    std::lock_guard<std::mutex> lock{m};
//    return empty_unsafe();
//  }
//
//  template <typename... Args>
//  void emplace(Args &&...args) noexcept {
//    // FIXME: seems like emplace starves pop!
//    //     event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END> er(this);
//    {
//      std::lock_guard<std::mutex> lock{m};
//      data.push(std::forward<Args>(args)...);
//    }
//    cv.notify_one();
//  }
//
//  T pop() noexcept {
//    std::unique_lock<std::mutex> lock{m};
//    while (empty_unsafe()) {
//      //       event_range<log_op::EXCHANGE_CONSUMER_WAIT_FOR_FREE_END>
//      //       er(this);
//      cv.wait(lock,  // std::chrono::milliseconds{1},
//              [this]() { return !empty_unsafe(); });
//      LOG_IF_EVERY_N(INFO, empty_unsafe(), 1000)
//          << "woke up to check, but is empty";
//    }
//    T v = std::move(data.top());
//    data.pop();
//    return std::move(v);
//  }
//};

void exec(AsyncQueueMPMC<size_t> &origin, AsyncQueueMPMC<size_t> *qs,
          size_t c) {
  set_exec_location_on_scope e(topology::getInstance().getCores()[c]);
  while (true) {
    size_t x;
    if (!origin.pop(x)) break;
    qs[x & 1].emplace(x);
  }

  qs[0].close();
  qs[1].close();
}

size_t exec2(AsyncQueueMPMC<size_t> &origin, size_t c) {
  set_exec_location_on_scope e(topology::getInstance().getCores()[c]);

  size_t cnt = 0;

  while (true) {
    size_t x;
    if (!origin.pop(x)) break;
    cnt += 1;
  }

  return cnt;
}

template <typename T>
void BM_MultiThreaded_core(benchmark::State &state) {
  //  set_exec_location_on_scope
  //  eg(topology::getInstance().getCpuNumaNodes()[0]);
  set_exec_location_on_scope eg(topology::getInstance().getCores()[14]);

  auto origin_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>));
  auto firstGroup_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>) * 2);
  auto secondGroup_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>) * 2);

  new (origin_ptr) AsyncQueueMPMC<size_t>(0);
  new (firstGroup_ptr) AsyncQueueMPMC<size_t>(0);
  new (firstGroup_ptr + 1) AsyncQueueMPMC<size_t>(0);
  new (secondGroup_ptr) AsyncQueueMPMC<size_t>(0);
  new (secondGroup_ptr + 1) AsyncQueueMPMC<size_t>(0);

  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};

  for (auto _ : state) {
    std::thread t1{[&]() { exec(*origin_ptr, firstGroup_ptr, 1); }};
    std::thread t2{[&]() { exec(*origin_ptr, secondGroup_ptr, 3); }};
    std::thread t3{[&]() { vecs[0] += exec2(firstGroup_ptr[0], 5); }};
    std::thread t4{[&]() { vecs[1] += exec2(firstGroup_ptr[1], 7); }};
    std::thread t5{[&]() { vecs[2] += exec2(secondGroup_ptr[0], 9); }};
    std::thread t6{[&]() { vecs[3] += exec2(secondGroup_ptr[1], 11); }};

    size_t maxitems = 1024 * 1024;
    totalcnt += maxitems;
    for (size_t i = 0; i < maxitems; ++i) {
      while (origin_ptr->size_unsafe() < 64 * 1024) std::this_thread::yield();
      origin_ptr->emplace(i);
    }
    origin_ptr->close();
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    origin_ptr->reset();
    firstGroup_ptr[0].reset();
    firstGroup_ptr[1].reset();
    secondGroup_ptr[0].reset();
    secondGroup_ptr[1].reset();
  }

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / (vecs[0] + vecs[1]) << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / (vecs[2] + vecs[3]);
  //  LOG_IF(ERROR, vecs[0] + vecs[1] + vecs[2] + vecs[3] != totalcnt)
  //      << "broken..." << vecs[0] + vecs[1] + vecs[2] + vecs[3] << " vs "
  //      << totalcnt;

  state.SetItemsProcessed(totalcnt);
}

void exec_2(AsyncQueueMPMC<size_t> &origin, AsyncQueueMPMC<size_t> *qs,
            size_t c) {
  set_exec_location_on_scope e(
      topology::getInstance().getCores()[c].getLocalCPUNumaNode());
  while (true) {
    size_t x;
    if (!origin.pop(x)) break;
    qs[x & 1].emplace(x);
  }

  qs[0].close();
  qs[1].close();
}

size_t exec2_2(AsyncQueueMPMC<size_t> &origin, size_t c) {
  set_exec_location_on_scope e(
      topology::getInstance().getCores()[c].getLocalCPUNumaNode());

  size_t cnt = 0;

  while (true) {
    size_t x;
    if (!origin.pop(x)) break;
    cnt += 1;
  }

  return cnt;
}

template <typename T>
void BM_MultiThreaded_socket(benchmark::State &state) {
  //  set_exec_location_on_scope
  //  eg(topology::getInstance().getCpuNumaNodes()[1]);
  set_exec_location_on_scope eg(
      topology::getInstance().getCores()[14].getLocalCPUNumaNode());

  auto origin_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>));
  auto firstGroup_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>) * 2);
  auto secondGroup_ptr = (AsyncQueueMPMC<size_t> *)MemoryManager::mallocPinned(
      sizeof(AsyncQueueMPMC<size_t>) * 2);

  new (origin_ptr) AsyncQueueMPMC<size_t>(0);
  new (firstGroup_ptr) AsyncQueueMPMC<size_t>(0);
  new (firstGroup_ptr + 1) AsyncQueueMPMC<size_t>(0);
  new (secondGroup_ptr) AsyncQueueMPMC<size_t>(0);
  new (secondGroup_ptr + 1) AsyncQueueMPMC<size_t>(0);

  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};

  for (auto _ : state) {
    std::thread t1{[&]() { exec_2(*origin_ptr, firstGroup_ptr, 1); }};
    std::thread t2{[&]() { exec_2(*origin_ptr, secondGroup_ptr, 3); }};
    std::thread t3{[&]() { vecs[0] = exec2_2(firstGroup_ptr[0], 5); }};
    std::thread t4{[&]() { vecs[1] = exec2_2(firstGroup_ptr[1], 7); }};
    std::thread t5{[&]() { vecs[2] = exec2_2(secondGroup_ptr[0], 9); }};
    std::thread t6{[&]() { vecs[3] = exec2_2(secondGroup_ptr[1], 11); }};

    size_t maxitems = 1024 * 1024;
    totalcnt = maxitems;
    for (size_t i = 0; i < maxitems; ++i) {
      while (origin_ptr->size_unsafe() < 64 * 1024) std::this_thread::yield();
      origin_ptr->emplace(i);
    }
    origin_ptr->close();
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    origin_ptr->reset();
    firstGroup_ptr[0].reset();
    firstGroup_ptr[1].reset();
    secondGroup_ptr[0].reset();
    secondGroup_ptr[1].reset();
  }

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / (vecs[0] + vecs[1]) << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / (vecs[2] + vecs[3]);
  //  LOG_IF(ERROR, vecs[0] + vecs[1] + vecs[2] + vecs[3] != totalcnt)
  //      << "broken..." << vecs[0] + vecs[1] + vecs[2] + vecs[3] << " vs "
  //      << totalcnt;

  state.SetItemsProcessed(totalcnt);
}

template <typename T>
void BM_MultiThreaded_sym(benchmark::State &state) {
  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};

  for (auto _ : state) {
    size_t maxitems = 1024 * 1024;
    for (size_t i = 0; i < 4; ++i) vecs[i] = 0;
    totalcnt = maxitems;
    for (size_t i = 0; i < maxitems; ++i) {
      auto x = (rand() & 1);
      ++vecs[x * 2 + (i & 1)];
    }
  }

  auto A = vecs[0] + vecs[1];
  auto B = vecs[2] + vecs[3];

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / A << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / B << " "
            << std::abs(A * 1.0 - B) / (A + B);

  state.SetItemsProcessed(totalcnt);
}

void atomiccnt_fighter(std::atomic<size_t> &counter, size_t limit,
                       size_t *vecs) {
  size_t localvec[2]{0, 0};

  while (counter > limit * 2)
    ;

  do {
    auto tmp = counter++;
    if (tmp > limit) break;
    ++localvec[tmp & 1];
  } while (true);

  vecs[0] = localvec[0];
  vecs[1] = localvec[1];
}

void atomiccnt_fighter_core(std::atomic<size_t> &counter, size_t limit,
                            size_t *vecs, size_t core_id) {
  set_exec_location_on_scope eg(topology::getInstance().getCores()[core_id]);
  size_t localvec[2]{0, 0};

  while (counter > limit * 2)
    ;

  do {
    auto tmp = counter++;
    if (tmp > limit) break;
    ++localvec[tmp & 1];
  } while (true);

  vecs[0] = localvec[0];
  vecs[1] = localvec[1];
}

template <typename T>
void BM_MultiThreaded_sym_atomiccntfight_core(benchmark::State &state) {
  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};
  std::atomic<size_t> cnt;

  for (auto _ : state) {
    size_t maxitems = 1024 * 1024;
    cnt = 4 * maxitems;

    std::thread t1{
        [&]() { atomiccnt_fighter_core(cnt, maxitems, vecs + 0, 3); }};
    std::thread t2{
        [&]() { atomiccnt_fighter_core(cnt, maxitems, vecs + 2, 5); }};

    std::this_thread::sleep_for(std::chrono::milliseconds{10});

    cnt = 0;

    t1.join();
    t2.join();
  }

  auto A = vecs[0] + vecs[1];
  auto B = vecs[2] + vecs[3];

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / A << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / B << " "
            << std::abs(A * 1.0 - B) / (A + B);
  //  LOG_IF(ERROR, vecs[0] + vecs[1] + vecs[2] + vecs[3] != totalcnt)
  //      << "broken..." << vecs[0] + vecs[1] + vecs[2] + vecs[3] << " vs "
  //      << totalcnt;

  state.SetItemsProcessed(totalcnt);
}

template <typename T>
void BM_MultiThreaded_sym_atomiccntfight(benchmark::State &state) {
  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};
  std::atomic<size_t> cnt;

  for (auto _ : state) {
    size_t maxitems = 1024 * 1024;
    cnt = 4 * maxitems;

    std::thread t1{[&]() { atomiccnt_fighter(cnt, maxitems, vecs + 0); }};
    std::thread t2{[&]() { atomiccnt_fighter(cnt, maxitems, vecs + 2); }};

    std::this_thread::sleep_for(std::chrono::milliseconds{10});

    cnt = 0;

    t1.join();
    t2.join();
  }

  auto A = vecs[0] + vecs[1];
  auto B = vecs[2] + vecs[3];

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / A << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / B << " "
            << std::abs(A * 1.0 - B) / (A + B);

  state.SetItemsProcessed(totalcnt);
}

void atomiccnt_fighter_socket(std::atomic<size_t> &counter, size_t limit,
                              size_t *vecs, size_t core_id) {
  set_exec_location_on_scope eg(
      topology::getInstance().getCores()[core_id].getLocalCPUNumaNode());
  size_t localvec[2]{0, 0};

  while (counter > limit * 2)
    ;

  do {
    auto tmp = counter++;
    if (tmp > limit) break;
    ++localvec[tmp & 1];
  } while (true);

  vecs[0] = localvec[0];
  vecs[1] = localvec[1];
}

template <typename T>
void BM_MultiThreaded_sym_atomiccntfight_socket(benchmark::State &state) {
  size_t totalcnt = 0;

  size_t vecs[4]{0, 0, 0, 0};
  std::atomic<size_t> cnt;

  for (auto _ : state) {
    size_t maxitems = 1024 * 1024;
    cnt = 4 * maxitems;

    std::thread t1{
        [&]() { atomiccnt_fighter_socket(cnt, maxitems, vecs + 0, 3); }};
    std::thread t2{
        [&]() { atomiccnt_fighter_socket(cnt, maxitems, vecs + 2, 5); }};

    std::this_thread::sleep_for(std::chrono::milliseconds{10});

    cnt = 0;

    t1.join();
    t2.join();
  }

  auto A = vecs[0] + vecs[1];
  auto B = vecs[2] + vecs[3];

  LOG(INFO) << std::setprecision(2) << std::fixed
            << std::abs(vecs[0] * 1.0 - vecs[1]) / A << " "
            << std::abs(vecs[2] * 1.0 - vecs[3]) / B << " "
            << std::abs(A * 1.0 - B) / (A + B);
  //  LOG_IF(ERROR, vecs[0] + vecs[1] + vecs[2] + vecs[3] != totalcnt)
  //      << "broken..." << vecs[0] + vecs[1] + vecs[2] + vecs[3] << " vs "
  //      << totalcnt;

  state.SetItemsProcessed(totalcnt);
}

BENCHMARK_TEMPLATE(BM_MultiThreaded_core, threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_MultiThreaded_socket, threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_MultiThreaded_sym, threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_MultiThreaded_sym_atomiccntfight, threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_MultiThreaded_sym_atomiccntfight_core,
                   threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_MultiThreaded_sym_atomiccntfight_socket,
                   threadsafe_set<size_t>)
    //    ->RangeMultiplier(1 << 10)
    //    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMillisecond)
    //    ->MinTime(10 /* seconds */)
    ->UseRealTime();
// BENCHMARK_TEMPLATE(BM_MultiThreaded, threadsafe_set2<size_t>)
//    //    ->RangeMultiplier(1 << 10)
//    ->Range(1 << 10, 1 << 20)
//    ->Threads(benchmark::CPUInfo::Get().num_cpus / 2)
//    ->Threads(benchmark::CPUInfo::Get().num_cpus)
//    ->Unit(benchmark::kMillisecond)
////    ->MinTime(10 /* seconds */)
//    ->UseRealTime();

int main(int argc, char **argv) {
  auto ctx = proteus::olap(0.02, 0.001);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
