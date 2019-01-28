/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Labaratory (DIAS)
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

class Scan : public UnaryRawOperator {
 public:
  Scan(RawContext *const context, Plugin &pg)
      : UnaryRawOperator(NULL), context(context), pg(pg) {}
  //    Scan(RawContext* const context, Plugin& pg, RawOperator* parent) :
  //            UnaryRawOperator(NULL), context(context), pg(pg) {
  //        this->setParent(parent);
  //    }
  virtual ~Scan() { LOG(INFO) << "Collapsing scan operator"; }
  RawOperator *const getChild() const {
    throw runtime_error(string("Scan operator has no children"));
  }

  virtual void produce();
  virtual void consume(RawContext *const context,
                       const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

 private:
  RawContext *const __attribute__((unused)) context;
  Plugin &pg;
};

#endif /* SCAN_HPP_ */
