/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "hash-join-chained-morsel.hpp"

#include <platform/memory/memory-manager.hpp>
#include <platform/util/timing.hpp>

#include "lib/util/jit/pipeline.hpp"

llvm::Value *HashJoinChainedMorsel::nextIndex(ParallelContext *context) {
  // TODO: consider using just the object id as the index, instead of the atomic
  //  index
  auto *out_cnt = context->getStateVar(cnt_param_id);
  out_cnt->setName(opLabel + "_cnt_ptr");

  auto v = context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::Add, out_cnt,
      llvm::ConstantInt::get(out_cnt->getType()->getPointerElementType(), 1),
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(
          context->getSizeOf(out_cnt->getType()->getPointerElementType())),
#endif
      llvm::AtomicOrdering::SequentiallyConsistent);

  v->setName("index");
  return v;
}

llvm::Value *HashJoinChainedMorsel::replaceHead(ParallelContext *context,
                                                llvm::Value *h_ptr,
                                                llvm::Value *index) {
  auto *old_head = context->getBuilder()->CreateAtomicRMW(
      llvm::AtomicRMWInst::BinOp::Xchg, h_ptr, index,
#if LLVM_VERSION_MAJOR >= 13
      llvm::Align(context->getSizeOf(index)),
#endif
      llvm::AtomicOrdering::SequentiallyConsistent);
  old_head->setName("old_head");
  return old_head;
}

void HashJoinChainedMorsel::open_build(Pipeline *pip) {
  {
    std::lock_guard<std::mutex> lock(init_lock);
    if (workerCnt++ == 0) {
      std::vector<void *> next_w_values;

      auto *head = (uint32_t *)MemoryManager::mallocPinned(
          sizeof(uint32_t) * (1 << hash_bits) + sizeof(int32_t));
      auto *cnt = (int32_t *)(head + (1 << hash_bits));

      // cudaStream_t strm;
      // gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
      memset(head, -1, sizeof(uint32_t) * (1 << hash_bits));
      memset(cnt, 0, sizeof(int32_t));

      for (const auto &w : build_packet_widths) {
        next_w_values.emplace_back(
            MemoryManager::mallocPinned((w / 8) * maxBuildInputSize));
      }

      next_w_values.emplace_back(head);
      confs[0] = next_w_values;
    }
  }

  auto *head = (uint32_t *)confs[0].back();
  auto *cnt = (int32_t *)(head + (1 << hash_bits));

  pip->setStateVar(head_param_id, head);
  pip->setStateVar(cnt_param_id, cnt);

  for (size_t i = 0; i < build_packet_widths.size(); ++i) {
    pip->setStateVar(out_param_ids[i], confs[0][i]);
  }
}

void HashJoinChainedMorsel::open_probe(Pipeline *pip) {
  std::vector<void *> next_w_values = confs[0];
  auto *head = (uint32_t *)next_w_values.back();

  pip->setStateVar(probe_head_param_id, head);

  for (size_t i = 0; i < build_packet_widths.size(); ++i) {
    pip->setStateVar(in_param_ids[i], next_w_values[i]);
  }
}

void HashJoinChainedMorsel::close_build(Pipeline *pip) {}

void HashJoinChainedMorsel::close_probe(Pipeline *pip) {
  std::lock_guard<std::mutex> lock(init_lock);
  if (--workerCnt == 0) {
    auto *head = (uint32_t *)confs[0].back();
    auto *cnt = (int32_t *)(head + (1 << hash_bits));

    int32_t h_cnt = *cnt;
    LOG(INFO) << h_cnt << " " << maxBuildInputSize;
    LOG_IF(INFO, h_cnt < 0.5 * maxBuildInputSize || h_cnt > maxBuildInputSize)
        << "Actual build "
           "input size: "
        << h_cnt << " (capacity: " << maxBuildInputSize << ")";
    assert(((size_t)h_cnt) <= maxBuildInputSize &&
           "Build input sized exceeded given parameter");

    for (const auto &p : confs[0]) MemoryManager::freePinned(p);
  }
}
