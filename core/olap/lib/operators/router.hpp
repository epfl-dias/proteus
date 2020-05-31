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
#include <threadpool/threadvector.hpp>
#include <utility>

#include "olap/routing/routing-policy.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/device-manager.hpp"
#include "util/async_containers.hpp"
#include "util/logging.hpp"

class Router;

extern "C" {
[[nodiscard]] void *acquireBuffer(int target, Router *xch);
[[nodiscard]] void *try_acquireBuffer(int target, Router *xch);
void releaseBuffer(int target, Router *xch, void *buff);
void freeBuffer(int target, Router *xch, void *buff);
}

class Router : public experimental::UnaryOperator {
 public:
  Router(Operator *const child, DegreeOfParallelism numOfParents,
         vector<RecordAttribute *> wantedFields, int slack,
         std::optional<expression_t> hash, RoutingPolicy policy_type,
         std::unique_ptr<Affinitizer> aff)
      : UnaryOperator(child),
        wantedFields(std::move(wantedFields)),
        slack(slack),
        fanout(numOfParents.dop),
        producers(child->getDOP().dop),
        remaining_producers(producers),
        hashExpr(std::move(hash)),
        need_cnt(false),
        free_pool(nullptr),
        policy_type(policy_type),
        aff(std::move(aff)) {
    assert((policy_type == RoutingPolicy::HASH_BASED) == hash.has_value() &&
           "hash should only contain a value for hash-based routing policies");
    assert(this->aff && "Affinitizer should be non-null");
  }

  Router(Operator *const child, DegreeOfParallelism numOfParents,
         vector<RecordAttribute *> wantedFields, int slack,
         std::optional<expression_t> hash, RoutingPolicy policy_type,
         DeviceType targets)
      : Router(child, numOfParents, wantedFields, slack, hash, policy_type,
               getDefaultAffinitizer(targets)) {}

  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override { return wantedFields; }
  [[nodiscard]] DegreeOfParallelism getDOP() const override {
    return DegreeOfParallelism{fanout};
  }

 protected:
  virtual void generate_catch(ParallelContext *context);

  void fire(int target, PipelineGen *pipGen);

  virtual std::unique_ptr<routing::RoutingPolicy> getPolicy() const;

 protected:
  [[nodiscard]] void *acquireBuffer(int target, bool polling = false);
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

  virtual void spawnWorker(size_t i);

  threadvector firers;

  const size_t fanout;
  int producers;

  PipelineGen *catch_pip;

  std::atomic<int> remaining_producers;
  const int slack;
  size_t buf_size;

  AsyncQueueMPSC<void *> *ready_fifo;

  std::unique_ptr<Affinitizer> aff;
  std::mutex init_mutex;

 private:
  const vector<RecordAttribute *> wantedFields;

  llvm::Type *params_type;

  std::stack<void *> *free_pool;
  std::mutex *free_pool_mutex;
  std::condition_variable *free_pool_cv;

  std::optional<expression_t> hashExpr;

  bool need_cnt;

  const RoutingPolicy policy_type;
};

#endif /* ROUTER_HPP_ */
