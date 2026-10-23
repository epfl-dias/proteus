/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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
#ifndef UNIONALL_HPP_
#define UNIONALL_HPP_

#include "lib/operators/router/router.hpp"
#include "olap/util/parallel-context.hpp"

class UnionAll : public Router {
 public:
  UnionAll(std::vector<Operator *> &children,
           const std::vector<RecordAttribute *> &wantedFields,
           DegreeOfParallelism fanout = DegreeOfParallelism{1})
      : Router(children[0], fanout, wantedFields, 8, std::nullopt,
               RoutingPolicy::RANDOM, getDefaultAffinitizer(DeviceType::CPU)),
        children(children) {
    CHECK_GT(children.size(), 0)
        << "UnionAll operator must have at least one child";
    setChild(nullptr);
    producers = children.size() * children[0]->getDOP().dop;
    remaining_producers = producers;
  }

  ~UnionAll() override { LOG(INFO) << "Collapsing UnionAll operator"; }

  void produce_(OlapParallelContext *context) override;

  DegreeOfParallelism getDOPServers() const override {
    auto dop = children[0]->getDOPServers();
#ifdef NDEBUG
    for (const auto &op : children) {
      assert(dop == op->getDOPServers());
    }
#endif
    return dop;
  }

  DeviceType getDeviceType() const override {
    return children[0]->getDeviceType();
  }

  [[nodiscard]] bool isPacked() const override {
    auto x = children[0]->isPacked();
#ifndef NDEBUG
    for (const auto &c : children) {
      CHECK_EQ(x, c->isPacked())
          << "UnionAll operator children must have the same packing";
    }
#endif
    return x;
  }

  std::vector<Operator *> getChildren() const { return children; }

 private:
  std::vector<Operator *> children;
};

#endif /* UNIONALL_HPP_ */
