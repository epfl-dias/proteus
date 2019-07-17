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

#include "codegen/util/parallel-context.hpp"
#include "operators/device-cross.hpp"
#include "operators/operators.hpp"

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

  virtual void produce();
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);

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

  size_t lockVar_id;
  size_t lastVar_id;
  size_t flagsVar_id;
  size_t storeVar_id;
  size_t threadVar_id;
  size_t eofVar_id;

  size_t flagsVar_id_catch;
  size_t storeVar_id_catch;
  size_t eofVar_id_catch;

  size_t size;

  gran_t granularity;
};

#endif /* GPU_TO_CPU_HPP_ */
