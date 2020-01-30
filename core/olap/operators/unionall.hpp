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
#ifndef UNIONALL_HPP_
#define UNIONALL_HPP_

#include "operators/router.hpp"
#include "util/parallel-context.hpp"

class UnionAll : public Router {
 public:
  UnionAll(vector<Operator *> &children,
           const vector<RecordAttribute *> &wantedFields)
      : Router(children[0], DegreeOfParallelism{1}, wantedFields, 8,
               std::nullopt, RoutingPolicy::RANDOM, DeviceType::CPU,
               getDefaultAffinitizer(DeviceType::CPU)),
        children(children) {
    setChild(nullptr);
    producers = children.size();
  }

  virtual ~UnionAll() { LOG(INFO) << "Collapsing UnionAll operator"; }

  virtual void produce_(ParallelContext *context);
  //     virtual void consume(   Context * const context, const
  //     OperatorState& childState); virtual void consume(ParallelContext *
  //     const context, const OperatorState& childState); virtual bool
  //     isFiltering() const {return false;}

  // protected:
  //     virtual void generate_catch();

  virtual DegreeOfParallelism getDOP() const {
    auto dop = children[0]->getDOP();
#ifdef NDEBUG
    for (const auto &op : children) {
      assert(dop == op->getDOP());
    }
#endif
    return dop;
  }

 private:
  vector<Operator *> children;
};

#endif /* UNIONALL_HPP_ */
