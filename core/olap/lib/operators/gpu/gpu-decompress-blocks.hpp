/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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
#ifndef GPU_DECOMPRESS_BLOCKS_HPP_
#define GPU_DECOMPRESS_BLOCKS_HPP_

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <platform/common/gpu/gpu-common.hpp>

#include "lib/operators/operators.hpp"
#include "olap/util/parallel-context.hpp"

class NvmePlugin;

class GpuDecompressBlocks : public experimental::UnaryOperator {
 public:
  GpuDecompressBlocks(Operator *const child,
                      std::vector<expression_t> wantedFields,
                      gran_t granularity = gran_t::GRID)
      : UnaryOperator(child),
        wantedFields(std::move(wantedFields)) {
    assert(this->getChild()->getDeviceType() == DeviceType::GPU &&
           "GpuDecompressBlocks only supports GPU pipelines");
    assert(granularity != gran_t::BLOCK &&
           "BLOCK granularity is not supported yet");
    std::lock_guard<std::mutex> lock(live_instances_mtx_);
    live_instances_.insert(this);
  }

  void produce_(OlapParallelContext *context) override;
  void consume(OlapParallelContext *context,
               const OperatorState &childState) override;
  [[nodiscard]] bool isFiltering() const override { return false; }

  [[nodiscard]] RecordType getRowType() const override {
    return this->getChild()->getRowType();
  }

  [[nodiscard]] bool isPacked() const override {
    assert(this->getChild()->isPacked());
    return true;
  }
  ~GpuDecompressBlocks() override;
  static void releaseAllRuntimeBuffers();

 private:
  void releaseRuntimeBuffers();
  void ensureRuntimeBuffersAllocated(NvmePlugin *nvme_pg);
  void decompressRuntimeBuffers();
  void open(Pipeline *pip);
  void close(Pipeline *pip);

  const std::vector<expression_t> wantedFields;
  NvmePlugin *nvme_pg_{nullptr};
  std::string rel_name_;
  bool runtime_buffers_allocated_{false};
  bool runtime_buffers_decompressed_{false};
  std::vector<size_t> part_block_counts_;
  std::vector<std::vector<void *>> decompressed_part_bases_;
  std::vector<std::vector<size_t>> decompressed_part_bytes_;

  static std::mutex live_instances_mtx_;
  static std::unordered_set<GpuDecompressBlocks *> live_instances_;
};

#endif /* GPU_DECOMPRESS_BLOCKS_HPP_ */
