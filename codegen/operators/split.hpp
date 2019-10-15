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

#include "operators/router.hpp"

class Split : public Router {
 public:
  Split(Operator *const child, ParallelContext *const context,
        size_t numOfParents, const vector<RecordAttribute *> &wantedFields,
        int slack, std::optional<expression_t> hash = std::nullopt,
        RoutingPolicy policy_type = RoutingPolicy::LOCAL)
      : Router(child, context, DegreeOfParallelism{numOfParents}, wantedFields,
               slack, hash, policy_type, DeviceType::CPU),
        produce_calls(0) {
    producers = 1;  // Set so that it does not get overwritten by Routers' cnstr
  }

  virtual ~Split() { LOG(INFO) << "Collapsing Split operator"; }

  virtual void produce();

  virtual void setParent(Operator *parent) {
    UnaryOperator::setParent(parent);

    this->parent.emplace_back(parent);
  }

  virtual DegreeOfParallelism getDOP() const { return getChild()->getDOP(); }

 protected:
  void open(Pipeline *pip);

 private:
  size_t produce_calls;
  std::vector<PipelineGen *> catch_pip;
  std::vector<Operator *> parent;
};

#endif /* SPLIT_HPP_ */
