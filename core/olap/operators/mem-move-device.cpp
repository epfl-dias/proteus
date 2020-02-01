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

#include "operators/mem-move-device.hpp"

#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "threadpool/threadpool.hpp"
#include "util/catalog.hpp"
#include "util/jit/pipeline.hpp"
#include "util/logging.hpp"

buff_pair buff_pair::not_moved(void *buff) { return {buff, buff}; }

buff_pair MemMoveDevice::MemMoveConf::push(void *src, size_t bytes,
                                           int target_device) {
  const auto *d = topology::getInstance().getGpuAddressed(src);
  int dev = d ? d->id : -1;

  if (dev == target_device) {
    return buff_pair::not_moved(src);  // block in correct device
  }

  assert(bytes <=
         BlockManager::block_size);  // FIMXE: buffer manager should be able
                                     // to provide blocks of arbitary size
  char *buff = (char *)BlockManager::h_get_buffer(target_device);

  if (bytes > 0) {
    BlockManager::overwrite_bytes(buff, src, bytes, strm, false);
  }

  return buff_pair{buff, src};
}

extern "C" {
buff_pair make_mem_move_device(char *src, size_t bytes, int target_device,
                               MemMoveDevice::MemMoveConf *mmc) {
  return mmc->push(src, bytes, target_device);
}

void *MemMoveConf_pull(void *buff, MemMoveDevice::MemMoveConf *mmc) {
  return mmc->pull(buff);
}
}

void MemMoveDevice::produce_(ParallelContext *context) {
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = llvm::Type::getInt32Ty(context->getLLVMContext());
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

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

  data_type = llvm::StructType::get(llvmContext, tr_types);

  RecordAttribute tupleCnt =
      RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?
  RecordAttribute tupleIdentifier = RecordAttribute(
      wantedFields[0]->getRelationName(), activeLoop, pg->getOIDType());

  // Generate catch code
  int p = context->appendParameter(llvm::PointerType::get(data_type, 0), true,
                                   true);
  context->setGlobalFunction();

  auto *Builder = context->getBuilder();
  auto *entryBB = Builder->GetInsertBlock();
  auto *F = entryBB->getParent();

  auto *mainBB = llvm::BasicBlock::Create(llvmContext, "main", F);

  auto *endBB = llvm::BasicBlock::Create(llvmContext, "end", F);
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

    auto relBB = llvm::BasicBlock::Create(llvmContext, "rel", F);
    auto merBB = llvm::BasicBlock::Create(llvmContext, "mer", F);

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

  auto cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size());
  Builder->CreateStore(cnt, mem_cntWrapper.mem);

  variableBindings[tupleCnt] = mem_cntWrapper;

  ProteusValueMemory mem_oidWrapper;
  mem_oidWrapper.mem = context->CreateEntryBlockAlloca(F, activeLoop, oidType);
  mem_oidWrapper.isNull =
      context
          ->createFalse();  // FIMXE: should we alse transfer this information ?

  auto oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);
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

  ((ParallelContext *)context)->registerOpen(this, [this](Pipeline *pip) {
    this->open(pip);
  });
  ((ParallelContext *)context)->registerClose(this, [this](Pipeline *pip) {
    this->close(pip);
  });

  getChild()->produce(context);
}

void MemMoveDevice::consume(Context *const context,
                            const OperatorState &childState) {
  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();
  auto F = insBB->getParent();

  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto workunit_type = llvm::StructType::get(
      llvmContext, std::vector<llvm::Type *>{charPtrType, charPtrType});

  map<RecordAttribute, ProteusValueMemory> old_bindings{
      childState.getBindings()};

  // Find block size
  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt =
      RecordAttribute(wantedFields[0]->getRelationName(), "activeCnt",
                      pg->getOIDType());  // FIXME: OID type for blocks ?

  auto it = old_bindings.find(tupleCnt);
  assert(it != old_bindings.end());

  ProteusValueMemory mem_cntWrapper = it->second;

  auto make_mem_move = context->getFunction("make_mem_move_device");

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto device_id = ((ParallelContext *)context)->getStateVar(device_id_var);
  // Value * cu_stream       = ((ParallelContext *)
  // context)->getStateVar(cu_stream_var);

  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(mem_cntWrapper.mem);

  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};
  it = old_bindings.find(tupleIdentifier);
  assert(it != old_bindings.end());
  ProteusValueMemory mem_oidWrapper = it->second;
  llvm::Value *oid = Builder->CreateLoad(mem_oidWrapper.mem);

  llvm::Value *memmv = ((ParallelContext *)context)->getStateVar(memmvconf_var);

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

    llvm::Value *size = llvm::ConstantInt::get(
        llvmContext, llvm::APInt(64, context->getSizeOf(mv_block_type)));
    auto Nloc = Builder->CreateZExtOrBitCast(N, size->getType());
    size = Builder->CreateMul(size, Nloc);
    llvm::Value *moved = mv;
    llvm::Value *to_release = mv;
    if (do_transfer[i]) {
      vector<llvm::Value *> mv_args{mv, size, device_id, memmv};

      // Do actual mem move
      auto moved_buffpair = Builder->CreateCall(make_mem_move, mv_args);
      moved = Builder->CreateExtractValue(moved_buffpair, 0);
      to_release = Builder->CreateExtractValue(moved_buffpair, 1);
    } else {
      LOG(INFO) << "Lazy: " << wantedFields[i]->getRelationName() << "."
                << wantedFields[i]->getAttrName();
    }

    pushed.push_back(Builder->CreateBitCast(
        moved, mem_valWrapper.mem->getType()->getPointerElementType()));
    pushed.push_back(Builder->CreateBitCast(
        to_release, mem_valWrapper.mem->getType()->getPointerElementType()));

    is_noop =
        Builder->CreateAnd(is_noop, Builder->CreateICmpEQ(moved, to_release));
  }
  pushed.push_back(N);
  pushed.push_back(oid);

  llvm::Value *d = llvm::UndefValue::get(data_type);
  for (size_t i = 0; i < pushed.size(); ++i) {
    d = Builder->CreateInsertValue(d, pushed[i], i);
  }

  auto acquire = context->getFunction("acquireWorkUnit");

  auto workunit_ptr8 = Builder->CreateCall(acquire, memmv);
  auto workunit_ptr = Builder->CreateBitCast(
      workunit_ptr8, llvm::PointerType::getUnqual(workunit_type));

  auto workunit_dat = Builder->CreateLoad(workunit_ptr);
  auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
  d_ptr =
      Builder->CreateBitCast(d_ptr, llvm::PointerType::getUnqual(data_type));
  Builder->CreateStore(d, d_ptr);

  auto propagate = context->getFunction("propagateWorkUnit");
  Builder->CreateCall(
      propagate, std::vector<llvm::Value *>{memmv, workunit_ptr8, is_noop});
}

MemMoveDevice::MemMoveConf *MemMoveDevice::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  return new (pmmc) MemMoveConf;
}

void MemMoveDevice::destroyMoveConf(MemMoveDevice::MemMoveConf *mmc) const {
  mmc->~MemMoveConf();
  MemoryManager::freePinned(mmc);
}

void MemMoveDevice::open(Pipeline *pip) {
  workunit *wu =
      (workunit *)MemoryManager::mallocPinned(sizeof(workunit) * slack);

  // nvtxRangePushA("memmove::open");
  cudaStream_t strm = createNonBlockingStream();

  // cudaStream_t strm2;
  // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

  eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

  MemMoveConf *mmc = createMoveConf();

  eventlogger.log(this, log_op::MEMMOVE_OPEN_END);
#ifndef NCUDA
  mmc->strm = strm;
  // mmc->strm2          = strm2;
#endif
  mmc->slack = slack;
  mmc->next_e = 0;
  // mmc->events         = new cudaEvent_t[slack];
  // mmc->old_buffs      = new void      *[slack];
  mmc->data_buffs = MemoryManager::mallocPinned(data_size * slack);
  char *data_buff = (char *)mmc->data_buffs;
  eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
  for (size_t i = 0; i < slack; ++i) {
    wu[i].data = ((void *)(data_buff + i * data_size));
    // // gpu_run(cudaEventCreateWithFlags(&(wu[i].event),
    // cudaEventDisableTiming));//  | cudaEventBlockingSync));
    //         gpu_run(cudaEventCreate(&(wu[i].event)));
    //         gpu_run(cudaStreamCreate(&(wu[i].strm)));

    mmc->idle.push(wu + i);

    // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming
    // | cudaEventBlockingSync)); gpu_run(cudaEventCreate(mmc->events + i));
    // mmc->old_buffs[i] = nullptr;
  }
  eventlogger.log(this, log_op::MEMMOVE_OPEN_END);
  // nvtxRangePushA("memmove::open2");
  for (size_t i = 0; i < slack; ++i) {
    gpu_run(cudaEventCreateWithFlags(
        &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
    // // gpu_run(cudaEventCreateWithFlags(&(wu[i].event),
    // cudaEventDisableTiming));//  | cudaEventBlockingSync));
    //         gpu_run(cudaEventCreate(&(wu[i].event)));
    //         gpu_run(cudaStreamCreate(&(wu[i].strm)));

    // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming
    // | cudaEventBlockingSync)); gpu_run(cudaEventCreate(mmc->events + i));
    // mmc->old_buffs[i] = nullptr;
  }
  // nvtxRangePop();

  eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
  mmc->worker = ThreadPool::getInstance().enqueue(
      &MemMoveDevice::catcher, this, mmc, pip->getGroup(), exec_location{});
  // mmc->worker = new thread(&MemMoveDevice::catcher, this, mmc,
  // pip->getGroup(), exec_location{});
  eventlogger.log(this, log_op::MEMMOVE_OPEN_END);

  int device = -1;
  if (!to_cpu) device = topology::getInstance().getActiveGpu().id;
  pip->setStateVar<int>(device_id_var, device);

  // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
  pip->setStateVar<void *>(memmvconf_var, mmc);
  // nvtxRangePop();
}

void MemMoveDevice::close(Pipeline *pip) {
  eventlogger.log(this, log_op::MEMMOVE_CLOSE_START);
  // int device = get_device();
  // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
  MemMoveConf *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

  mmc->tran.close();

  nvtxRangePop();
  mmc->worker.get();
  // mmc->worker->join();

  eventlogger.log(this, log_op::MEMMOVE_CLOSE_END);

  eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_START);
  // gpu_run(cudaStreamSynchronize(g_strm));

  // int32_t h_s;
  // gpu_run(cudaMemcpy(&h_s, s, sizeof(int32_t), cudaMemcpyDefault));
  // std::cout << "rrr" << h_s << std::endl;

  // MemoryManager::freeGpu(s);
  syncAndDestroyStream(mmc->strm);
  // gpu_run(cudaStreamSynchronize(mmc->strm2));
  // gpu_run(cudaStreamDestroy    (mmc->strm2));

  nvtxRangePushA("MemMoveDev_running2");
  nvtxRangePushA("MemMoveDev_running");

  nvtxRangePushA("MemMoveDev_release");
  workunit *start_wu;
  // void     * start_wu_data;
  for (size_t i = 0; i < slack; ++i) {
    workunit *wu = mmc->idle.pop_unsafe();

    // if (mmc->old_buffs[i]) buffer-manager<int32_t>::release_buffer((int32_t
    // *) mmc->old_buffs[i]);

    gpu_run(cudaEventDestroy(wu->event));
    // gpu_run(cudaEventDestroy(mmc->events[i]));
    // free(wu->data);

    if (i == 0 || wu < start_wu) start_wu = wu;
    // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
  }
  nvtxRangePop();
  nvtxRangePop();

  MemoryManager::freePinned(mmc->data_buffs);
  // assert(mmc->tran.empty_unsafe());
  // assert(mmc->idle.empty_unsafe());
  // free(start_wu_data);
  // delete[] start_wu;
  MemoryManager::freePinned(start_wu);
  // delete[] mmc->events   ;
  // delete[] mmc->old_buffs;

  mmc->idle.close();  // false);

  // delete mmc->worker;
  // delete mmc;
  destroyMoveConf(mmc);
  eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_END);
}

void MemMoveDevice::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                           bool is_noop) {
  if (!is_noop) gpu_run(cudaEventRecord(buff->event, strm));

  tran.push(buff);
}

MemMoveDevice::workunit *MemMoveDevice::MemMoveConf::acquire() {
  MemMoveDevice::workunit *ret = nullptr;
#ifndef NDEBUG
  bool popres =
#endif
      idle.pop(ret);
  assert(popres);
  return ret;
}

bool MemMoveDevice::MemMoveConf::getPropagated(MemMoveDevice::workunit **ret) {
  if (!tran.pop(*ret)) return false;
  gpu_run(cudaEventSynchronize((*ret)->event));
  return true;
}

void MemMoveDevice::MemMoveConf::release(MemMoveDevice::workunit *buff) {
  idle.push(buff);
}

extern "C" {
MemMoveDevice::workunit *acquireWorkUnit(MemMoveDevice::MemMoveConf *mmc) {
  return mmc->acquire();
}

void propagateWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                       MemMoveDevice::workunit *buff, bool is_noop) {
  mmc->propagate(buff, is_noop);
}

bool acquirePendingWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                            MemMoveDevice::workunit **ret) {
  return mmc->getPropagated(ret);
}

void releaseWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                     MemMoveDevice::workunit *buff) {
  mmc->release(buff);
}
}

void MemMoveDevice::catcher(MemMoveConf *mmc, int group_id,
                            exec_location target_dev) {
  // std::cout << target_dev. << std::endl;
  set_exec_location_on_scope d(target_dev);
  std::this_thread::yield();

  nvtxRangePushA("memmove::catch");

  auto pip = catch_pip->getPipeline(group_id);

  nvtxRangePushA("memmove::catch_open");
  pip->open();
  nvtxRangePop();

  {
    do {
      MemMoveDevice::workunit *p = nullptr;
      if (!mmc->getPropagated(&p)) break;
      for (size_t i = 0; i < wantedFields.size(); ++i) {
        ((void **)(p->data))[i * 2] =
            MemMoveConf_pull(((void **)(p->data))[i * 2], mmc);
      }

      nvtxRangePushA("memmove::catch_cons");
      pip->consume(0, p->data);
      nvtxRangePop();

      mmc->release(p);  // FIXME: move this inside the generated code
    } while (true);
  }

  nvtxRangePushA("memmove::catch_close");
  pip->close();
  nvtxRangePop();

  nvtxRangePop();
}
