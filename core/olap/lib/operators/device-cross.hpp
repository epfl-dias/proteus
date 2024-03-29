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

#ifndef DEVICE_CROSS_HPP_
#define DEVICE_CROSS_HPP_

#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class DeviceCross : public UnaryOperator {
 protected:
  DeviceCross(Operator *const child) : UnaryOperator(child) {}

 public:
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState) = 0;

  void consume(Context *const context,
               const OperatorState &childState) override {
    ParallelContext *ctx = dynamic_cast<ParallelContext *>(context);
    if (!ctx) {
      string error_msg =
          "[DeviceCross: ] Operator only supports code "
          "generation using the ParallelContext";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    consume(ctx, childState);
  }

  bool isFiltering() const override { return false; }
};

#endif /* DEVICE_CROSS_HPP_ */
