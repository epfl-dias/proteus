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
#ifndef ROUTER_SCALEOUT_HPP_
#define ROUTER_SCALEOUT_HPP_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <utility>

#include "network/infiniband/infiniband-manager.hpp"
#include "olap/util/parallel-context.hpp"
#include "router.hpp"
#include "topology/affinity_manager.hpp"
#include "util/async_containers.hpp"
#include "util/logging.hpp"

class RouterScaleOut : public Router {
  size_t cnt = 0;

 public:
  RouterScaleOut(Operator *const child, DegreeOfParallelism numOfParents,
                 const std::vector<RecordAttribute *> &wantedFields, int slack,
                 std::optional<expression_t> hash, RoutingPolicy policy_type,
                 DeviceType targets, int producers)
      : Router(child, numOfParents, wantedFields, slack, std::move(hash),
               policy_type, targets),
        sub(nullptr) {
    setProducers(producers);
  }

  DegreeOfParallelism getDOP() const override { return DegreeOfParallelism{1}; }
  DegreeOfParallelism getDOPServers() const override {
    return Router::getDOP();
  }

  // virtual void produce();
  // virtual void consume(Context *const context, const OperatorState
  // &childState);

  // protected:
  //  virtual void generate_catch();

  //  void fire(int target, PipelineGen *pipGen);

  void fire(int target, PipelineGen *pipGen) override;

 protected:
  proteus::managed_ptr acquireBuffer(int target, bool polling) override;
  void releaseBuffer(int target, proteus::managed_ptr buff) override;
  void freeBuffer(int target, proteus::managed_ptr buff) override;
  bool get_ready(int target, proteus::managed_ptr &buff) override;

  std::atomic<size_t> closed = 0;
  subscription *sub;
  bool strmclosed = false;

 protected:
  // void open(Pipeline *pip);
  void open(Pipeline *pip) override;
  void close(Pipeline *pip) override;
  // void open(Pipeline *pip);
  virtual void fire_close(Pipeline *pip);
};

#endif /* ROUTER_SCALEOUT_HPP_ */
