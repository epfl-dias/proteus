/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
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

#ifndef DEVICE_CROSS_HPP_
#define DEVICE_CROSS_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class DeviceCross : public UnaryRawOperator {
 protected:
  DeviceCross(RawOperator *const child) : UnaryRawOperator(child) {}

 public:
  virtual void consume(GpuRawContext *const context,
                       const OperatorState &childState) = 0;

  virtual void consume(RawContext *const context,
                       const OperatorState &childState) {
    GpuRawContext *ctx = dynamic_cast<GpuRawContext *>(context);
    if (!ctx) {
      string error_msg =
          "[DeviceCross: ] Operator only supports code "
          "generation using the GpuRawContext";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    consume(ctx, childState);
  }

  virtual bool isFiltering() const { return false; }
};

#endif /* DEVICE_CROSS_HPP_ */