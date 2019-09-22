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
#ifndef EXCHANGE_HPP_
#define EXCHANGE_HPP_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

#include "codegen/util/parallel-context.hpp"
#include "operators/operators.hpp"
#include "topology/affinity_manager.hpp"
#include "util/async_containers.hpp"
#include "util/logging.hpp"

class Exchange;

extern "C" {
void *acquireBuffer(int target, Exchange *xch);
void *try_acquireBuffer(int target, Exchange *xch);
void releaseBuffer(int target, Exchange *xch, void *buff);
void freeBuffer(int target, Exchange *xch, void *buff);
}

class Exchange : public UnaryOperator {
 public:
  Exchange(Operator *const child, ParallelContext *const context,
           int numOfParents, const vector<RecordAttribute *> &wantedFields,
           int slack, std::optional<expression_t> hash = std::nullopt,
           bool numa_local = true, bool rand_local_cpu = false,
           int producers = 1, bool cpu_targets = false, int numa_socket_id = -1)
      : UnaryOperator(child),
        context(context),
        numOfParents(numOfParents),
        wantedFields(wantedFields),
        slack(slack),
        hashExpr(std::move(hash)),
        numa_local(numa_local),
        rand_local_cpu(rand_local_cpu),
        producers(producers),
        remaining_producers(producers),
        need_cnt(false) {
    assert(
        (!hash || !numa_local) &&
        "Just to make it more clear that hash has precedence over numa_local");

    free_pool = new std::stack<void *>[numOfParents];
    free_pool_mutex = new std::mutex[numOfParents];
    free_pool_cv = new std::condition_variable[numOfParents];

    // ready_pool          = new std::queue<void *>     [numOfParents];
    // ready_pool_mutex    = new std::mutex             [numOfParents];
    // ready_pool_cv       = new std::condition_variable[numOfParents];

    ready_fifo = new AsyncQueueMPSC<void *>[numOfParents];

    if (cpu_targets) {
      const auto &vec = topology::getInstance().getCpuNumaNodes();
      if (numa_socket_id >= 0 && numa_socket_id < vec.size()) {
        const auto &numaSocket = vec[numa_socket_id];
        for (int i = 0; i < numOfParents; ++i) {
          target_processors.emplace_back(numaSocket);
        }

      } else {
        for (int i = 0; i < numOfParents; ++i) {
          target_processors.emplace_back(vec[i % vec.size()]);
        }
      }

    } else {
      assert(topology::getInstance().getGpuCount() > 0 &&
             "Are you using an outdated plan?");
      const auto &vec = topology::getInstance().getGpus();

      for (int i = 0; i < numOfParents; ++i) {
        target_processors.emplace_back(vec[i % vec.size()]);
      }
    }
  }

  virtual ~Exchange() { LOG(INFO) << "Collapsing Exchange operator"; }

  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const { return wantedFields; }

 protected:
  virtual void generate_catch();

  void fire(int target, PipelineGen *pipGen);

 private:
  void *acquireBuffer(int target, bool polling = false);
  void releaseBuffer(int target, void *buff);
  void freeBuffer(int target, void *buff);
  bool get_ready(int target, void *&buff);

  friend void *acquireBuffer(int target, Exchange *xch);
  friend void *try_acquireBuffer(int target, Exchange *xch);
  friend void releaseBuffer(int target, Exchange *xch, void *buff);
  friend void freeBuffer(int target, Exchange *xch, void *buff);

 protected:
  void open(Pipeline *pip);
  void close(Pipeline *pip);

  const vector<RecordAttribute *> wantedFields;

  const int slack;
  const int numOfParents;
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

#endif /* EXCHANGE_HPP_ */
