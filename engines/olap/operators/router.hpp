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

#include "engines/olap/util/parallel-context.hpp"
#include "operators/operators.hpp"
#include "routing/routing-policy.hpp"
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
         std::optional<expression_t> hash, RoutingPolicy policy_type,
         DeviceType targets, std::unique_ptr<Affinitizer> aff = nullptr)
      : UnaryOperator(child),
        wantedFields(wantedFields),
        slack(slack),
        fanout(numOfParents.dop),
        producers(child->getDOP().dop),
        remaining_producers(producers),
        context(context),
        hashExpr(std::move(hash)),
        need_cnt(false),
        targets(targets),
        policy_type(policy_type),
        aff(aff ? std::move(aff) : getDefaultAffinitizer(targets)) {
    assert((policy_type == RoutingPolicy::HASH_BASED) == hash.has_value() &&
           "hash should only contain a value for hash-based routing policies");

    free_pool = new std::stack<void *>[fanout];
    free_pool_mutex = new std::mutex[fanout];
    free_pool_cv = new std::condition_variable[fanout];

    // ready_pool          = new std::queue<void *>     [fanout];
    // ready_pool_mutex    = new std::mutex             [fanout];
    // ready_pool_cv       = new std::condition_variable[fanout];

    ready_fifo = new AsyncQueueMPSC<void *>[fanout];
  }

  virtual ~Router() { LOG(INFO) << "Collapsing Router operator"; }

  virtual void produce();
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  virtual void consume(Context *const context,
                       const OperatorState &childState) {
    ParallelContext *ctx = dynamic_cast<ParallelContext *>(context);
    if (!ctx) {
      string error_msg =
          "[DeviceCross: ] Operator only supports code "
          "generation using the ParallelContext";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    consume(ctx, childState);
  }
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const { return wantedFields; }
  virtual DegreeOfParallelism getDOP() const {
    return DegreeOfParallelism{fanout};
  }

 protected:
  virtual void generate_catch();

  void fire(int target, PipelineGen *pipGen);

  virtual std::unique_ptr<routing::RoutingPolicy> getPolicy() const;

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

  std::optional<expression_t> hashExpr;

  bool need_cnt;

  const DeviceType targets;
  const RoutingPolicy policy_type;

  std::unique_ptr<Affinitizer> aff;
};

#endif /* ROUTER_HPP_ */
