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

#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/logging.hpp>
#include <platform/util/timing.hpp>

MemMoveScaleOut::MemMoveConf *MemMoveScaleOut::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  return new (pmmc) MemMoveScaleOut::MemMoveConf;
}

void MemMoveScaleOut::genReleaseOldBuffer(ParallelContext *context,
                                          llvm::Value *src) const {
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());
  //  src->getType()->dump();
  //  context->log(context->getBuilder()->CreatePtrToInt(context->getBuilder()->CreateBitCast(src,
  //  charPtrType), llvm::Type::getInt64Ty(context->getLLVMContext())));
  context->gen_call(release_buffer,
                    {context->getBuilder()->CreateBitCast(src, charPtrType)});
}

void MemMoveScaleOut::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                             bool is_noop) {
  if (!is_noop) {
    // null
  }

  tran.push(buff);

  ++cnt;
  if (cnt % (slack / (slack / 2)) == 0) InfiniBandManager::flush_read();
}

buff_pair MemMoveScaleOut::MemMoveConf::push(proteus::managed_ptr src,
                                             size_t bytes, int target_device,
                                             uint64_t srcServer) {
  if (srcServer == InfiniBandManager::server_id()) {
    //    BlockManager::share_host_buffer((int32_t *)src);
    //    BlockManager::share_host_buffer((int32_t *)src);
    auto x_mem =
        MemoryManager::mallocPinned(sizeof(std::pair(std::move(src), false)));
    auto x = new (x_mem) std::pair(std::move(src), false);
    return buff_pair::not_moved(proteus::managed_ptr{x});
  }

  auto x = InfiniBandManager::read(
      proteus::remote_managed_ptr{std::move(src), srcServer}, bytes);
  ++cnt;
  if (cnt % (slack / (slack / 2)) == 0) InfiniBandManager::flush_read();
  auto x_mem = MemoryManager::mallocPinned(sizeof(std::pair(x, false)));
  return buff_pair::not_moved(
      proteus::managed_ptr{new (x_mem) std::pair(x, true)});
}

proteus::managed_ptr MemMoveScaleOut::MemMoveConf::pull(
    proteus::managed_ptr buff) {
  //  event_range<log_op::MEMMOVE_CONSUME_WAIT_START> er(this);
  // LOG(INFO) << buff;
  // return buff;
  auto ptr = ((std::pair<void *, bool> *)buff.release());
  auto p = *ptr;
  MemoryManager::freePinned(ptr);
  //  delete ptr;
  if (p.second) {
    InfiniBandManager::flush_read();
    return ((subscription *)p.first)->wait().release();
  } else {
    return proteus::managed_ptr{p.first};
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
