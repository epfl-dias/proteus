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

#include "operators/mem-move-scaleout.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>

#include "network/infiniband/infiniband-manager.hpp"
#include "threadpool/threadpool.hpp"
#include "util/logging.hpp"

MemMoveScaleOut::MemMoveConf *MemMoveScaleOut::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  return new (pmmc) MemMoveScaleOut::MemMoveConf;
}

void MemMoveScaleOut::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                             bool is_noop) {
  if (!is_noop) {
    // null
  }

  tran.push(buff);

  ++cnt;
  if (cnt % (slack / 2) == 0) InfiniBandManager::flush_read();
}

buff_pair MemMoveScaleOut::MemMoveConf::push(void *src, size_t bytes,
                                             int target_device) {
  auto x = InfiniBandManager::read(src, bytes);
  return buff_pair{x, src};
}

void *MemMoveScaleOut::MemMoveConf::pull(void *buff) {
  return ((subscription *)buff)->wait().data;
}

bool MemMoveScaleOut::MemMoveConf::getPropagated(
    MemMoveDevice::workunit **ret) {
  if (!tran.pop(*ret)) return false;
  return true;
}

void MemMoveScaleOut::close(Pipeline *pip) {
  LOG(INFO) << "closing...";
  InfiniBandManager::flush_read();
  MemMoveDevice::close(pip);
}
