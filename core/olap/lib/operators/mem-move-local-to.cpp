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

#include "lib/util/catalog.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include <future>

#include "lib/util/jit/pipeline.hpp"
#include "memory/block-manager.hpp"
#include "threadpool/threadpool.hpp"
#include "topology/affinity_manager.hpp"

struct buff_pair {
  char *new_buff;
  char *old_buff;
};

using namespace llvm;

extern "C" {
buff_pair make_mem_move_local_to(char *src, size_t bytes, int target_device,
                                 MemMoveLocalTo::MemMoveConf *mmc) {
  const auto *d2 = topology::getInstance().getGpuAddressed(src);
  int dev = d2 ? d2->id : -1;

  if (dev == target_device || bytes <= 0 || dev < 0 ||
      numa_node_of_gpu(dev) == numa_node_of_gpu(target_device))
    return buff_pair{src, src};  // block already in correct device

  set_exec_location_on_scope d(*d2);

  assert(bytes <=
         BlockManager::block_size);  // FIMXE: buffer manager should be able
                                     // to provide blocks of arbitary size
  char *buff = (char *)buffer_manager<int32_t>::get_buffer_numa(
      numa_node_of_gpu(target_device));

  buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm, false);
  // buffer-manager<int32_t>::release_buffer ((int32_t *) src );

  return buff_pair{buff, src};
}
}

void MemMoveLocalTo::produce_(ParallelContext *context) {
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = Type::getInt32Ty(context->getLLVMContext());
  auto charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

  auto pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  auto oidType = pg->getOIDType()->getLLVMType(llvmContext);

  std::vector<llvm::Type *> tr_types;
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    tr_types.push_back(wantedFields[i]->getLLVMType(llvmContext));
    tr_types.push_back(wantedFields[i]->getLLVMType(
        llvmContext));  // old buffer, to be released
  }
  tr_types.push_back(oidType);  // cnt
  tr_types.push_back(oidType);  // oid

  data_type = StructType::get(llvmContext, tr_types);

  RecordAttribute tupleCnt =
      RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?
  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType());

  // Generate catch code
  int p = context->appendParameter(PointerType::get(data_type, 0), true, true);
  context->setGlobalFunction();

  auto Builder = context->getBuilder();
  auto entryBB = Builder->GetInsertBlock();
  auto F = entryBB->getParent();

  auto mainBB = BasicBlock::Create(llvmContext, "main", F);

  auto endBB = BasicBlock::Create(llvmContext, "end", F);
  context->setEndingBlock(endBB);

  Builder->SetInsertPoint(entryBB);

  auto params = Builder->CreateLoad(context->getArgument(p));

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  auto release = context->getFunction("release_buffer");

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    ProteusValueMemory mem_valWrapper;

    mem_valWrapper.mem = context->CreateEntryBlockAlloca(
        F, wantedFields[i]->getAttrName() + "_ptr",
        wantedFields[i]->getOriginalType()->getLLVMType(llvmContext));
    mem_valWrapper.isNull =
        context->createFalse();  // FIMXE: should we alse transfer this
                                 // information ?

    auto param = Builder->CreateExtractValue(params, 2 * i);

    auto src = Builder->CreateExtractValue(params, 2 * i + 1);

    auto relBB = BasicBlock::Create(llvmContext, "rel", F);
    auto merBB = BasicBlock::Create(llvmContext, "mer", F);

    auto do_rel = Builder->CreateICmpEQ(param, src);
    Builder->CreateCondBr(do_rel, merBB, relBB);

    Builder->SetInsertPoint(relBB);

    Builder->CreateCall(release, Builder->CreateBitCast(src, charPtrType));

    Builder->CreateBr(merBB);

    Builder->SetInsertPoint(merBB);

    Builder->CreateStore(param, mem_valWrapper.mem);

    variableBindings[*(wantedFields[i])] = mem_valWrapper;
  }

  ProteusValueMemory mem_cntWrapper;
  mem_cntWrapper.mem = context->CreateEntryBlockAlloca(F, "activeCnt", oidType);
  mem_cntWrapper.isNull =
      context
          ->createFalse();  // FIMXE: should we alse transfer this information ?

  Value *cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size());
  Builder->CreateStore(cnt, mem_cntWrapper.mem);

  variableBindings[tupleCnt] = mem_cntWrapper;

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = context->CreateEntryBlockAlloca(F, activeLoop, oidType);
  mem_oidWrapper.isNull =
      context
          ->createFalse();  // FIMXE: should we alse transfer this information ?

  Value *oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);
  Builder->CreateStore(oid, mem_oidWrapper.mem);

  variableBindings[tupleIdentifier] = mem_oidWrapper;

  context->setCurrentEntryBlock(Builder->GetInsertBlock());

  Builder->SetInsertPoint(mainBB);

  OperatorState state{*this, variableBindings};
  getParent()->consume(context, state);

  Builder->CreateBr(endBB);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());
  // Insert an explicit fall through from the current (entry) block to the
  // CondBB.
  Builder->CreateBr(mainBB);

  Builder->SetInsertPoint(context->getEndingBlock());
  // Builder->CreateRetVoid();

  context->popPipeline();

  catch_pip = context->removeLatestPipeline();

  // push new pipeline for the throw part
  context->pushPipeline();

  device_id_var = context->appendStateVar(int32_type);
  // cu_stream_var       = context->appendStateVar(charPtrType);
  memmvconf_var = context->appendStateVar(charPtrType);

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

void MemMoveLocalTo::consume(Context *const context,
                             const OperatorState &childState) {
  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();

  auto charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

  auto workunit_type = StructType::get(
      llvmContext, std::vector<Type *>{charPtrType, charPtrType});

  map<RecordAttribute, ProteusValueMemory> old_bindings{
      childState.getBindings()};

  // Find block size
  auto pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt =
      RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?

  auto it = old_bindings.find(tupleCnt);
  assert(it != old_bindings.end());

  ProteusValueMemory mem_cntWrapper = it->second;

  auto make_mem_move = context->getFunction("make_mem_move_local_to");

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto device_id = ((ParallelContext *)context)->getStateVar(device_id_var);
  // Value * cu_stream       = ((ParallelContext *)
  // context)->getStateVar(cu_stream_var);

  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(mem_cntWrapper.mem);

  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType());
  it = old_bindings.find(tupleIdentifier);
  assert(it != old_bindings.end());
  ProteusValueMemory mem_oidWrapper = it->second;
  auto oid = Builder->CreateLoad(mem_oidWrapper.mem);

  auto memmv = ((ParallelContext *)context)->getStateVar(memmvconf_var);

  std::vector<llvm::Value *> pushed;
  llvm::Value *is_noop = context->createTrue();
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute block_attr(*(wantedFields[i]), true);

    auto it = old_bindings.find(block_attr);
    assert(it != old_bindings.end());
    ProteusValueMemory mem_valWrapper = it->second;

    auto mv = Builder->CreateBitCast(Builder->CreateLoad(mem_valWrapper.mem),
                                     charPtrType);

    auto mv_block_type = mem_valWrapper.mem->getType()
                             ->getPointerElementType()
                             ->getPointerElementType();

    llvm::Value *size = ConstantInt::get(
        llvmContext, APInt(64, context->getSizeOf(mv_block_type)));
    auto Nloc = Builder->CreateZExtOrBitCast(N, size->getType());
    size = Builder->CreateMul(size, Nloc);

    vector<llvm::Value *> mv_args{mv, size, device_id, memmv};

    // Do actual mem move
    auto moved_buffpair = Builder->CreateCall(make_mem_move, mv_args);
    auto moved = Builder->CreateExtractValue(moved_buffpair, 0);
    auto to_release = Builder->CreateExtractValue(moved_buffpair, 1);

    pushed.push_back(Builder->CreateBitCast(
        moved, mem_valWrapper.mem->getType()->getPointerElementType()));
    pushed.push_back(Builder->CreateBitCast(
        to_release, mem_valWrapper.mem->getType()->getPointerElementType()));

    is_noop =
        Builder->CreateAnd(is_noop, Builder->CreateICmpEQ(moved, to_release));
  }
  pushed.push_back(N);
  pushed.push_back(oid);

  llvm::Value *d = UndefValue::get(data_type);
  for (size_t i = 0; i < pushed.size(); ++i) {
    d = Builder->CreateInsertValue(d, pushed[i], i);
  }

  auto acquire = context->getFunction("mem_move_local_to_acquireWorkUnit");

  auto workunit_ptr8 = Builder->CreateCall(acquire, memmv);
  auto workunit_ptr = Builder->CreateBitCast(
      workunit_ptr8, PointerType::getUnqual(workunit_type));

  auto workunit_dat = Builder->CreateLoad(workunit_ptr);
  auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
  d_ptr = Builder->CreateBitCast(d_ptr, PointerType::getUnqual(data_type));
  Builder->CreateStore(d, d_ptr);

  auto propagate = context->getFunction("mem_move_local_to_propagateWorkUnit");
  Builder->CreateCall(propagate,
                      std::vector<Value *>{memmv, workunit_ptr8, is_noop});
}

void MemMoveLocalTo::open(Pipeline *pip) {
  int device = topology::getInstance().getActiveGpu().id;

  // set_device_on_scope d(1-device);

  cudaStream_t strm = createNonBlockingStream();

  cudaStream_t strm2 = createNonBlockingStream();

  MemMoveConf *mmc = new MemMoveConf;
#ifndef NCUDA
  mmc->strm = strm;
  mmc->strm2 = strm2;
#endif
  mmc->slack = slack;
  mmc->next_e = 0;
  mmc->events = new cudaEvent_t[slack];
  mmc->old_buffs = new void *[slack];

  workunit *wu = new workunit[slack];
  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);
  // void * data_buff = malloc(data_size * slack);
  for (size_t i = 0; i < slack; ++i) {
    wu[i].data =
        malloc(data_size);  //((void *) (((char *) data_buff) + i * data_size));
    gpu_run(cudaEventCreateWithFlags(
        &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));

    mmc->idle.push(wu + i);

    gpu_run(cudaEventCreateWithFlags(
        mmc->events + i, cudaEventDisableTiming | cudaEventBlockingSync));
    mmc->old_buffs[i] = nullptr;
  }

  mmc->worker = ThreadPool::getInstance().enqueue(
      &MemMoveLocalTo::catcher, this, mmc, pip->getGroup(), exec_location{});

  pip->setStateVar<int>(device_id_var, device);

  // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
  pip->setStateVar<void *>(memmvconf_var, mmc);
}

void MemMoveLocalTo::close(Pipeline *pip) {
  // int device = topology::getInstance().getActiveGpu().id;
  // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
  MemMoveConf *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

  nvtxRangePushA("MemMoveLocal_running");
  nvtxRangePushA("MemMoveLocal_running2");

  nvtxRangePop();
  mmc->tran.close();
  nvtxRangePop();
  mmc->worker.get();

  gpu_run(cudaStreamSynchronize(mmc->strm));
  gpu_run(cudaStreamDestroy(mmc->strm));
  gpu_run(cudaStreamSynchronize(mmc->strm2));
  gpu_run(cudaStreamDestroy(mmc->strm2));

  nvtxRangePushA("MemMoveLocalTo::release");
  workunit *start_wu;
  // void     * start_wu_data;
  for (size_t i = 0; i < slack; ++i) {
    workunit *wu = mmc->idle.pop_unsafe();

    if (mmc->old_buffs[i])
      buffer_manager<int32_t>::release_buffer((int32_t *)mmc->old_buffs[i]);

    gpu_run(cudaEventDestroy(wu->event));
    gpu_run(cudaEventDestroy(mmc->events[i]));
    free(wu->data);

    if (i == 0 || wu < start_wu) start_wu = wu;
    // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
  }
  nvtxRangePop();
  // free(start_wu_data);
  delete[] start_wu;
  delete[] mmc->events;
  delete[] mmc->old_buffs;

  mmc->idle.close();

  // delete mmc->worker;
  delete mmc;
}

extern "C" {
MemMoveLocalTo::workunit *mem_move_local_to_acquireWorkUnit(
    MemMoveLocalTo::MemMoveConf *mmc) {
  MemMoveLocalTo::workunit *ret = nullptr;
#ifndef NDEBUG
  bool popres =
#endif
      mmc->idle.pop(ret);
  assert(popres);
  return ret;
}

void mem_move_local_to_propagateWorkUnit(MemMoveLocalTo::MemMoveConf *mmc,
                                         MemMoveLocalTo::workunit *buff,
                                         bool is_noop) {
  if (!is_noop) gpu_run(cudaEventRecord(buff->event, mmc->strm));

  mmc->tran.push(buff);
}

bool mem_move_local_to_acquirePendingWorkUnit(MemMoveLocalTo::MemMoveConf *mmc,
                                              MemMoveLocalTo::workunit **ret) {
  return mmc->tran.pop(*ret);
}

void mem_move_local_to_releaseWorkUnit(MemMoveLocalTo::MemMoveConf *mmc,
                                       MemMoveLocalTo::workunit *buff) {
  mmc->idle.push(buff);
}
}

void MemMoveLocalTo::catcher(MemMoveConf *mmc, int group_id,
                             const exec_location &target_dev) {
  set_exec_location_on_scope d(target_dev);

  auto pip = catch_pip->getPipeline(group_id);
  nvtxRangePushA("MemMoveLocalTo::catch");

  nvtxRangePushA("MemMoveLocalTo::catch_open");
  pip->open();
  nvtxRangePop();

  {
    do {
      MemMoveLocalTo::workunit *p = nullptr;
      if (!mem_move_local_to_acquirePendingWorkUnit(mmc, &p)) break;

      gpu_run(cudaEventSynchronize(p->event));
      nvtxRangePushA("MemMoveLocalTo::catch_cons");
      pip->consume(0, p->data);
      nvtxRangePop();

      mem_move_local_to_releaseWorkUnit(
          mmc, p);  // FIXME: move this inside the generated code
    } while (true);
  }

  nvtxRangePushA("MemMoveLocalTo::catch_close");
  pip->close();
  nvtxRangePop();
  nvtxRangePop();
}
