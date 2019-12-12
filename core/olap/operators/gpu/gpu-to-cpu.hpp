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
#ifndef GPU_TO_CPU_HPP_
#define GPU_TO_CPU_HPP_

#include "common/gpu/gpu-common.hpp"
#include "operators/device-cross.hpp"
#include "operators/operators.hpp"
#include "util/parallel-context.hpp"

class GpuToCpu : public DeviceCross {
 public:
  GpuToCpu(Operator *const child, ParallelContext *const context,
           const vector<RecordAttribute *> &wantedFields, size_t size,
           gran_t granularity = gran_t::GRID)
      : DeviceCross(child),
        context(context),
        wantedFields(wantedFields),
        size(size),
        granularity(granularity) {}

  virtual ~GpuToCpu() { LOG(INFO) << "Collapsing GpuToCpu operator"; }

  virtual void produce_(ParallelContext *context);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);

  virtual RecordType getRowType() const { return wantedFields; }

  virtual DeviceType getDeviceType() const {
    assert(getChild()->getDeviceType() == DeviceType::GPU);
    return DeviceType::CPU;
  }

 private:
  void generate_catch();

  void open(Pipeline *pip);
  void close(Pipeline *pip);

  const vector<RecordAttribute *> wantedFields;

  ParallelContext *const context;

  PipelineGen *cpu_pip;

  llvm::Type *params_type;

  StateVar lockVar_id;
  StateVar lastVar_id;
  StateVar flagsVar_id;
  StateVar storeVar_id;
  StateVar threadVar_id;
  StateVar eofVar_id;

  StateVar flagsVar_id_catch;
  StateVar storeVar_id_catch;
  StateVar eofVar_id_catch;

  size_t size;

  gran_t granularity;
};

#endif /* GPU_TO_CPU_HPP_ */
