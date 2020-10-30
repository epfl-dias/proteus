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

#include "mem-move-local-to.hpp"

#include <future>
#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>
#include <topology/affinity_manager.hpp>

MemMoveLocalTo::MemMoveConf *MemMoveLocalTo::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  return new (pmmc) MemMoveConf;
}

void MemMoveLocalTo::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                            bool is_noop) {
  if (!is_noop) gpu_run(cudaEventRecord(buff->event, strm));

  tran.push(buff);
}

buff_pair MemMoveLocalTo::MemMoveConf::push(void *src, size_t bytes,
                                            int target_device, uint64_t) {
  const auto *d2 = topology::getInstance().getGpuAddressed(src);
  int dev = d2 ? d2->id : -1;

  if (dev == target_device || bytes <= 0 || dev < 0 ||
      numa_node_of_gpu(dev) == numa_node_of_gpu(target_device))
    return buff_pair::not_moved(src);  // block already in correct device

  set_exec_location_on_scope d(*d2);

  assert(bytes <=
         BlockManager::block_size);  // FIMXE: buffer manager should be able
  // to provide blocks of arbitary size
  char *buff =
      (char *)BlockManager::get_buffer_numa(numa_node_of_gpu(target_device));

  BlockManager::overwrite_bytes(buff, src, bytes, strm, false);
  // buffer-manager<int32_t>::release_buffer ((int32_t *) src );

  return buff_pair{buff, src};
}

bool MemMoveLocalTo::MemMoveConf::getPropagated(MemMoveDevice::workunit **ret) {
  return tran.pop(*ret);
}
