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

#include "mem-broadcast-scaleout.hpp"

#include <platform/memory/block-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/util/timing.hpp>

buff_pair MemBroadcastScaleOut::MemBroadcastConf::pushBroadcast(
    void *src, size_t bytes, int target_device, bool disable_noop) {
  if (target_device == InfiniBandManager::server_id()) {
    return buff_pair{new std::pair(src, false),
                     nullptr};  // block already in correct device
  }

  // BlockManager::share_host_buffer((int32_t *)src);
  auto x = InfiniBandManager::write_silent(proteus::managed_ptr{src}, bytes);
  return buff_pair{new std::pair(x, true), nullptr};
}

void MemBroadcastScaleOut::MemBroadcastConf::propagateBroadcast(
    MemMoveDevice::workunit *buff, int target_device) {
  tran.push(buff);

  if (target_device == InfiniBandManager::server_id()) return;
  assert(slack);
  ++cnt;
  if (cnt % (slack / 2) == 0) InfiniBandManager::flush();
}

void *MemBroadcastScaleOut::MemBroadcastConf::pull(void *buff) {
  auto ptr = (std::pair<void *, bool> *)buff;
  auto p = *ptr;
  delete ptr;
  if (p.second) {
    auto x = ((subscription *)p.first)->wait();
    return x.release().release();
  } else {
    return p.first;
  }
}

bool MemBroadcastScaleOut::MemBroadcastConf::getPropagated(
    MemMoveDevice::workunit **ret) {
  if (!tran.pop(*ret)) return false;
  return true;
}

MemBroadcastScaleOut::MemBroadcastConf *MemBroadcastScaleOut::createMoveConf()
    const {
  return new MemBroadcastScaleOut::MemBroadcastConf;
}
