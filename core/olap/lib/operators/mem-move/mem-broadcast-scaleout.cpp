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

static std::atomic<size_t> cnt = 0;

buff_pair MemBroadcastScaleOut::MemBroadcastConf::pushBroadcast(
    const proteus::managed_ptr &src, size_t bytes, int target_device,
    bool disable_noop) {
  auto newbuff = BlockManager::share_host_buffer(src);
  if (target_device == InfiniBandManager::server_id()) {
    // block already in correct device
    return buff_pair{proteus::managed_ptr{
                         new std::pair<void *, bool>(newbuff.release(), false)},
                     nullptr};
  }
  //  LOG_EVERY_N(WARNING, 100) << ++::cnt;

  auto x = InfiniBandManager::write_silent(std::move(newbuff), bytes);
  auto y = new std::pair<void *, bool>(x, true);
  return buff_pair{proteus::managed_ptr{y}, nullptr};
}

void MemBroadcastScaleOut::MemBroadcastConf::propagateBroadcast(
    MemMoveDevice::workunit *buff, int target_device) {
  tran.push(buff);

  if (target_device == InfiniBandManager::server_id()) return;
  assert(slack);
  ++cnt;
  if (cnt % (slack / 2) == 0) InfiniBandManager::flush();
}

proteus::managed_ptr MemBroadcastScaleOut::MemBroadcastConf::pull(
    proteus::managed_ptr buff) {
  auto ptr = (std::pair<void *, bool> *)buff.release();
  auto p = *ptr;
  delete ptr;
  if (p.second) {
    auto x = ((subscription *)p.first)->wait();
    return x.release();
  } else {
    return proteus::managed_ptr{p.first};
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
