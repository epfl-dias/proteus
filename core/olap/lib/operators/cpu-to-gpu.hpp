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
#ifndef CPU_TO_GPU_HPP_
#define CPU_TO_GPU_HPP_

#include "device-cross.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class CpuToGpu : public DeviceCross {
 public:
  CpuToGpu(Operator *const child, const vector<RecordAttribute *> &wantedFields)
      : DeviceCross(child), wantedFields(wantedFields) {}

  virtual ~CpuToGpu() { LOG(INFO) << "Collapsing CpuToGpu operator"; }

  virtual void produce_(ParallelContext *context);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);

  virtual void generateGpuSide(ParallelContext *context);

  virtual RecordType getRowType() const { return wantedFields; }

  virtual DeviceType getDeviceType() const {
    assert(getChild()->getDeviceType() == DeviceType::CPU);
    return DeviceType::GPU;
  }

 private:
  const vector<RecordAttribute *> wantedFields;

  PipelineGen *gpu_pip;
  StateVar childVar_id;
  StateVar strmVar_id;
};

#endif /* CPU_TO_GPU_HPP_ */
