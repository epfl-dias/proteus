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
#ifndef ROUTER_HPP_
#define ROUTER_HPP_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

#include "codegen/util/parallel-context.hpp"
#include "operators/operators.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/device-manager.hpp"
#include "util/async_containers.hpp"
#include "util/logging.hpp"

class Router;

extern "C" {
void *acquireBuffer(int target, Router *xch);
void *try_acquireBuffer(int target, Router *xch);
void releaseBuffer(int target, Router *xch, void *buff);
void freeBuffer(int target, Router *xch, void *buff);
}

class Router : public UnaryOperator {
 public:
  Router(Operator *const child, ParallelContext *const context,
         DegreeOfParallelism numOfParents,
         const vector<RecordAttribute *> &wantedFields, int slack,
         std::optional<expression_t> hash = std::nullopt,
         bool numa_local = true, bool rand_local_cpu = false,
         DeviceType targets = DeviceType::GPU)
      : UnaryOperator(child),
        wantedFields(wantedFields),
        slack(slack),
        fanout(numOfParents.dop),
        producers(child->getDOP().dop),
        remaining_producers(producers),
        context(context),
        hashExpr(std::move(hash)),
        numa_local(numa_local),
        rand_local_cpu(rand_local_cpu),
        need_cnt(false) {
    assert(
        (!hashExpr || !numa_local) &&
        "Just to make it more clear that hash has precedence over numa_local");

    free_pool = new std::stack<void *>[fanout];
    free_pool_mutex = new std::mutex[fanout];
    free_pool_cv = new std::condition_variable[fanout];

    // ready_pool          = new std::queue<void *>     [fanout];
    // ready_pool_mutex    = new std::mutex             [fanout];
    // ready_pool_cv       = new std::condition_variable[fanout];

    ready_fifo = new AsyncQueueMPSC<void *>[fanout];

    auto &devmanager = DeviceManager::getInstance();

    if (targets == DeviceType::CPU) {
      for (size_t i = 0; i < fanout; ++i) {
        target_processors.emplace_back(devmanager.getAvailableCPUCore(this, i));
        // target_processors.emplace_back(vec[i % vec.size()]);
      }
    } else {
      assert(topology::getInstance().getGpuCount() > 0 &&
             "Are you using an outdated plan?");

      for (size_t i = 0; i < fanout; ++i) {
        target_processors.emplace_back(devmanager.getAvailableGPU(this, i));
        // target_processors.emplace_back(vec[i % vec.size()]);
      }
    }
  }

  virtual ~Router() { LOG(INFO) << "Collapsing Router operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const { return wantedFields; }
  virtual DegreeOfParallelism getDOP() const {
    return DegreeOfParallelism{fanout};
  }

 protected:
  virtual void generate_catch();

  void fire(int target, PipelineGen *pipGen);

 private:
  void *acquireBuffer(int target, bool polling = false);
  void releaseBuffer(int target, void *buff);
  void freeBuffer(int target, void *buff);
  bool get_ready(int target, void *&buff);

  friend void *acquireBuffer(int target, Router *xch);
  friend void *try_acquireBuffer(int target, Router *xch);
  friend void releaseBuffer(int target, Router *xch, void *buff);
  friend void freeBuffer(int target, Router *xch, void *buff);

  inline void setProducers(int producers) {
    this->producers = producers;
    remaining_producers = producers;
  }

 protected:
  void open(Pipeline *pip);
  void close(Pipeline *pip);

  const vector<RecordAttribute *> wantedFields;

  const int slack;
  const size_t fanout;
  int producers;
  std::atomic<int> remaining_producers;
  ParallelContext *const context;

  llvm::Type *params_type;
  PipelineGen *catch_pip;

  std::vector<std::thread> firers;

  // std::queue<void *>            * ready_pool;
  // std::mutex                    * ready_pool_mutex;
  // std::condition_variable       * ready_pool_cv;

  AsyncQueueMPSC<void *> *ready_fifo;

  std::stack<void *> *free_pool;
  std::mutex *free_pool_mutex;
  std::condition_variable *free_pool_cv;

  std::mutex init_mutex;

  std::vector<exec_location> target_processors;

  std::optional<expression_t> hashExpr;
  bool numa_local;
  bool rand_local_cpu;

  bool need_cnt;
};

#endif /* ROUTER_HPP_ */
