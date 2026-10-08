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
#ifndef CPU_TO_GPU_HPP_
#define CPU_TO_GPU_HPP_

#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "device-cross.hpp"
#include "olap/util/parallel-context.hpp"
#include "operators.hpp"

class CpuToGpu : public DeviceCross {
 public:
  CpuToGpu(Operator *const child,
           const std::vector<RecordAttribute *> &wantedFields)
      : DeviceCross(child), wantedFields(wantedFields) {}

  ~CpuToGpu() override;

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *const context,
               const OperatorState &childState) override;

  virtual void generateGpuSide(OlapParallelContext *context);

  void enableStreamingGridWindow() { streamingGridWindow = true; }

  RecordType getRowType() const override { return wantedFields; }

  DeviceType getDeviceType() const override {
    assert(getChild()->getDeviceType() == DeviceType::CPU);
    return DeviceType::GPU;
  }

 private:
  struct BatchDeviceBufferCacheEntry {
    void **batch_descs_gpu{nullptr};
    size_t capacity_blocks{0};
    int device_id{-1};
  };

  const std::vector<RecordAttribute *> wantedFields;
  bool streamingGridWindow{false};

  PipelineGen *gpu_pip;
  StateVar childVar_id;
  StateVar strmVar_id;

  // Batch launch state (used when FLAGS_in_gpu && FLAGS_one_kernel)
  StateVar batchBufVar_id;       // i8*  : base of CPU descriptor buffer
  StateVar batchCountVar_id;     // i64* : pointer to accumulated block count
  StateVar subStateCacheVar_id;  // i8** : pointer to cached sub_state pointer

  mutable std::mutex batchDeviceBuffersMutex_;
  std::unordered_map<const void *, BatchDeviceBufferCacheEntry>
      batchDeviceBuffers_;

  [[nodiscard]] size_t batchDescriptorStride() const;
  void ensureBatchDeviceBuffer(const void *pipeline_key, size_t needed_blocks);
  [[nodiscard]] BatchDeviceBufferCacheEntry getBatchDeviceBuffer(
      const void *pipeline_key) const;
};

#endif /* CPU_TO_GPU_HPP_ */
