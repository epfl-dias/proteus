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

#include "mem-move-scaleout.hpp"

#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>
#include <network/infiniband/infiniband-manager.hpp>
#include <threadpool/threadpool.hpp>
#include <util/logging.hpp>

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
                                             int target_device,
                                             uint64_t srcServer) {
  if (srcServer == InfiniBandManager::server_id()) {
    return {new std::pair(src, false), nullptr};
  }

  //  BlockManager::share_host_buffer((int32_t *)src);
  auto x = InfiniBandManager::read(
      proteus::remote_managed_ptr{proteus::managed_ptr{src}, srcServer}, bytes);
  //  InfiniBandManager::flush_read();
  return buff_pair{new std::pair(x, true),
                   reinterpret_cast<void *>(uintptr_t{1})};
}

void *MemMoveScaleOut::MemMoveConf::pull(void *buff) {
  // LOG(INFO) << buff;
  // return buff;
  auto ptr = ((std::pair<void *, bool> *)buff);
  auto p = *ptr;
  delete ptr;
  if (p.second) {
    return ((subscription *)p.first)->wait().release().release();
  } else {
    return p.first;
  }
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

ProteusValueMemory MemMoveScaleOut::getServerId(
    ParallelContext *, const OperatorState &childState) const {
  RecordAttribute srcServer{wantedFields[0]->getRelationName(), "srcServer",
                            new Int64Type()};  // FIXME: OID type for blocks ?
  return childState[srcServer];
}
