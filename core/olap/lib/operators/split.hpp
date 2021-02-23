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
#ifndef SPLIT_HPP_
#define SPLIT_HPP_

#include "lib/operators/router/router.hpp"

class Split : public Router {
 public:
  Split(Operator *const child, size_t numOfParents,
        const vector<RecordAttribute *> &wantedFields, int slack,
        std::optional<expression_t> hash = std::nullopt,
        RoutingPolicy policy_type = RoutingPolicy::LOCAL)
      : Router(child, DegreeOfParallelism{numOfParents}, wantedFields, slack,
               hash, policy_type, getDefaultAffinitizer(DeviceType::CPU)),
        produce_calls(0) {
    producers = 1;  // Set so that it does not get overwritten by Routers' cnstr
  }

  ~Split() override { LOG(INFO) << "Collapsing Split operator"; }

  void produce_(ParallelContext *context) override;

  void setParent(Operator *parent) override {
    UnaryOperator::setParent(parent);

    this->parent.emplace_back(parent);
  }

  DegreeOfParallelism getDOP() const override { return getChild()->getDOP(); }

 protected:
  void spawnWorker(size_t i, const void *session) override;

 private:
  size_t produce_calls;
  std::vector<PipelineGen *> catch_pip;
  std::vector<Operator *> parent;
};

#endif /* SPLIT_HPP_ */
