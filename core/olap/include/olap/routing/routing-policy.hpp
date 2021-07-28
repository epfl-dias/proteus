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

#ifndef ROUTING_POLICY_HPP_
#define ROUTING_POLICY_HPP_

#include <olap/expressions/expressions.hpp>
#include <olap/routing/routing-policy-types.hpp>
#include <olap/util/parallel-context.hpp>
#include <platform/topology/device-types.hpp>
#include <platform/topology/topology.hpp>
#include <utility>

#include "affinitizers.hpp"

struct routing_target {
  llvm::Value *target;
  bool may_retry;
};

namespace routing {
class RoutingPolicy {
 public:
  virtual ~RoutingPolicy() = default;
  virtual routing_target evaluate(ParallelContext *context,
                                  const OperatorState &childState,
                                  ProteusValueMemory retrycnt) = 0;
};

class Random : public RoutingPolicy {
  size_t fanout;

 public:
  explicit Random(size_t fanout) : fanout(fanout) {}
  routing_target evaluate(ParallelContext *context,
                          const OperatorState &childState,
                          ProteusValueMemory retrycnt) override;
};

class HashBased : public RoutingPolicy {
  size_t fanout;
  expression_t e;

 public:
  HashBased(size_t fanout, expression_t e) : fanout(fanout), e(std::move(e)) {}
  routing_target evaluate(ParallelContext *context,
                          const OperatorState &childState,
                          ProteusValueMemory retrycnt) override;
};

class Local : public RoutingPolicy {
  size_t fanout;
  RecordAttribute wantedField;
  const AffinityPolicy *aff;  // use pointer to satisfy lifetime requirements

 public:
  Local(size_t fanout, const std::vector<RecordAttribute *> &wantedFields,
        const AffinityPolicy *aff);
  routing_target evaluate(ParallelContext *context,
                          const OperatorState &childState,
                          ProteusValueMemory retrycnt) override;
};

class LocalServer : public HashBased {
 public:
  LocalServer(size_t fanout);
};

class PreferLocal : public RoutingPolicy {
  Local priority;
  Random alternative;

 public:
  PreferLocal(size_t fanout, const std::vector<RecordAttribute *> &wantedFields,
              const AffinityPolicy *aff);
  routing_target evaluate(ParallelContext *context,
                          const OperatorState &childState,
                          ProteusValueMemory retrycnt) override;
};

class PreferLocalServer : public RoutingPolicy {
  LocalServer priority;
  Random alternative;

 public:
  PreferLocalServer(size_t fanout);
  routing_target evaluate(ParallelContext *context,
                          const OperatorState &childState,
                          ProteusValueMemory retrycnt) override;
};

}  // namespace routing

#endif /* ROUTING_POLICY_HPP_ */
