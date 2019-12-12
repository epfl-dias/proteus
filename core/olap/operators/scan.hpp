/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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
#ifndef SCAN_HPP_
#define SCAN_HPP_

#include "operators/operators.hpp"

class Scan : public UnaryOperator {
 public:
  Scan(Context *const context, Plugin &pg)
      : UnaryOperator(nullptr), context(context), pg(pg) {}
  //    Scan(Context* const context, Plugin& pg, Operator* parent) :
  //            UnaryOperator(nullptr), context(context), pg(pg) {
  //        this->setParent(parent);
  //    }
  virtual ~Scan() { LOG(INFO) << "Collapsing scan operator"; }
  virtual Operator *const getChild() const final {
    throw runtime_error(string("Scan operator has no children"));
  }

  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

  virtual RecordType getRowType() const;

  virtual DeviceType getDeviceType() const { return DeviceType::CPU; }
  virtual DegreeOfParallelism getDOP() const { return DegreeOfParallelism{1}; }

 private:
  Context *const __attribute__((unused)) context;
  Plugin &pg;
};

#endif /* SCAN_HPP_ */
