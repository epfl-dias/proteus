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

#include "expressions/expressions.hpp"
#include "topology/device-types.hpp"
#include "util/parallel-context.hpp"

struct routing_target {
  llvm::Value *target;
  bool may_retry;
};

enum class RoutingPolicy { RANDOM, LOCAL, HASH_BASED };

class Affinitizer {
 public:
  virtual ~Affinitizer() = default;
  virtual size_t getAvailableCUIndex(size_t i) const = 0;
  virtual size_t size() const = 0;
  virtual size_t getLocalCUIndex(void *) const = 0;
};

class AffinityPolicy {
 protected:
  std::vector<std::vector<size_t>> indexes;
  const Affinitizer *aff;

 public:
  AffinityPolicy(size_t fanout, const Affinitizer *aff);
  size_t getIndexOfRandLocalCU(void *ptr) const;
};

namespace routing {
class RoutingPolicy {
 public:
  virtual ~RoutingPolicy() = default;
  virtual routing_target evaluate(ParallelContext *const context,
                                  const OperatorState &childState) = 0;
};

class Random : public RoutingPolicy {
  size_t fanout;

 public:
  Random(size_t fanout) : fanout(fanout) {}
  routing_target evaluate(ParallelContext *const context,
                          const OperatorState &childState) override;
};

class HashBased : public RoutingPolicy {
  size_t fanout;
  expression_t e;

 public:
  HashBased(size_t fanout, expression_t e) : fanout(fanout), e(e) {}
  routing_target evaluate(ParallelContext *const context,
                          const OperatorState &childState) override;
};

class Local : public RoutingPolicy {
  size_t fanout;
  RecordAttribute wantedField;
  const AffinityPolicy *aff;  // use pointer to satisfy lifetime requirements

 public:
  Local(size_t fanout, DeviceType dev,
        const std::vector<RecordAttribute *> &wantedFields);
  routing_target evaluate(ParallelContext *const context,
                          const OperatorState &childState) override;
};

}  // namespace routing

#endif /* ROUTING_POLICY_HPP_ */
