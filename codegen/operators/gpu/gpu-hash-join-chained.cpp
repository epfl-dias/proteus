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

#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "codegen/memory/memory-manager.hpp"
#include "expressions/expressions-hasher.hpp"
#include "operators/gpu/gmonoids.hpp"
#include "topology/topology.hpp"

void GpuHashJoinChained::open_build(Pipeline *pip) {
  std::vector<void *> next_w_values;

  uint32_t *head = (uint32_t *)MemoryManager::mallocGpu(
      sizeof(uint32_t) * (1 << hash_bits) + sizeof(int32_t));
  int32_t *cnt = (int32_t *)(head + (1 << hash_bits));

  cudaStream_t strm = createNonBlockingStream();
  gpu_run(cudaMemsetAsync(head, -1, sizeof(uint32_t) * (1 << hash_bits), strm));
  gpu_run(cudaMemsetAsync(cnt, 0, sizeof(int32_t), strm));

  for (const auto &w : build_packet_widths) {
    next_w_values.emplace_back(
        MemoryManager::mallocGpu((w / 8) * maxBuildInputSize));
  }

  pip->setStateVar(head_param_id, head);
  pip->setStateVar(cnt_param_id, cnt);

  for (size_t i = 0; i < build_packet_widths.size(); ++i) {
    pip->setStateVar(out_param_ids[i], next_w_values[i]);
  }

  next_w_values.emplace_back(head);
  confs[pip->getGroup()] = next_w_values;

  syncAndDestroyStream(strm);
}

void GpuHashJoinChained::close_build(Pipeline *pip) {
  int32_t h_cnt = -1;
  gpu_run(cudaMemcpy(&h_cnt, pip->getStateVar<int32_t *>(cnt_param_id),
                     sizeof(int32_t), cudaMemcpyDefault));
  assert(((size_t)h_cnt) <= maxBuildInputSize &&
         "Build input sized exceeded given parameter");
}

void GpuHashJoinChained::close_probe(Pipeline *pip) {
  for (const auto &p : confs[pip->getGroup()]) MemoryManager::freeGpu(p);
}
