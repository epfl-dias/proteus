/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include "codegen/memory/buffer-manager.cuh"
#include "codegen/memory/memory-manager.hpp"
#include "threadpool/threadpool.hpp"

struct buff_pair {
  char *new_buff;
  char *old_buff;
};

extern "C" {
buff_pair make_mem_move_device(char *src, size_t bytes, int target_device,
                               MemMoveDevice::MemMoveConf *mmc) {
  const auto *d = topology::getInstance().getGpuAddressed(src);
  int dev = d ? d->id : -1;

  if (dev == target_device)
    return buff_pair{src, src};  // block already in correct device

  // set_device_on_scope d(dev);

  // if (dev >= 0) set_affinity_local_to_gpu(dev);

  assert(bytes <= sizeof(int32_t) *
                      h_vector_size);  // FIMXE: buffer manager should be able
                                       // to provide blocks of arbitary size
  // std::cout << "MemMoveTarget: " << target_device << std::endl;
  char *buff = (char *)buffer_manager<int32_t>::h_get_buffer(target_device);

  // int numa_target = numa_node_of_gpu(target_device);
  // if (dev >= 0 && (numa_node_of_gpu(dev) != numa_target)){
  //     set_device_on_scope d(dev);

  //     if (dev >= 0) set_affinity_local_to_gpu(dev);

  //     size_t curr_e  = mmc->next_e;
  //     cudaEvent_t e  = mmc->events   [curr_e];
  //     void * old_ptr = mmc->old_buffs[curr_e];
  //     // mmc->old_buffs[curr_e] = NULL;

  //     if (old_ptr) buffer-manager<int32_t>::release_buffer((int32_t *)
  //     old_ptr); //FIXME: cannot release it yet!
  //     gpu_run(cudaEventSynchronize(e));
  //     mmc->next_e = (curr_e + 1) % mmc->slack;

  //     char * interbuff = (char *)
  //     buffer-manager<int32_t>::get_buffer_numa(numa_target);
  //     mmc->old_buffs[curr_e] = src;

  //     buffer-manager<int32_t>::overwrite_bytes(interbuff, src, bytes,
  //     mmc->strm2, false); gpu_run(cudaEventRecord(e, mmc->strm2)); src =
  //     interbuff;

  //     gpu_run(cudaStreamWaitEvent(mmc->strm, e, 0));
  // }

  if (bytes > 0)
    buffer_manager<int32_t>::overwrite_bytes(buff, src, bytes, mmc->strm,
                                             false);
  // assert(bytes == sizeof(int32_t) * h_vector_size);
  // std::cout << bytes << " " << sizeof(int32_t) * h_vector_size << std::endl;
  // cudaStream_t strm;
  // gpu_run(cudaStreamCreate(&(wu->strm)));
  // gpu_run(cudaMemcpyAsync(buff, src, bytes, cudaMemcpyDefault, wu->strm));
  // gpu_run(cudaMemcpyAsync(buff2, buff, bytes, cudaMemcpyDefault, wu->strm));
  // std::cout << "alloc" << (void *) buff2 << std::endl;

  // gpu_run(cudaMemcpy(buff, src, bytes, cudaMemcpyDefault));
  // buffer-manager<int32_t>::overwrite_bytes(buff, src, bytes, wu->strm,
  // false); buffer-manager<int32_t>::overwrite_bytes(buff2, buff, bytes,
  // wu->strm, false); gpu_run(cudaStreamSynchronize(mmc->strm));
  // gpu_run(cudaStreamSynchronize(wu->strm));
  // buffer-manager<int32_t>::release_buffer ((int32_t *) src );

  return buff_pair{buff, src};
}
}

void MemMoveDevice::produce() {
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

  getChild()->produce();
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

void MemMoveDevice::open(Pipeline *pip) {
  std::cout << "MemMoveDevice:open" << std::endl;
  workunit *wu =
      (workunit *)MemoryManager::mallocPinned(sizeof(workunit) * slack);

  // nvtxRangePushA("memmove::open");
  cudaStream_t strm;
  gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

  // cudaStream_t strm2;
  // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

  eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  MemMoveConf *mmc = new (pmmc) MemMoveConf;

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
    // mmc->old_buffs[i] = NULL;
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
    // mmc->old_buffs[i] = NULL;
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
  std::cout << "MemMoveDevice:close" << std::endl;
  // int device = get_device();
  // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
  MemMoveConf *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

  mmc->tran.close();
  std::cout << "MemMoveDevice:close3" << std::endl;

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
  std::cout << "MemMoveDevice:close4" << std::endl;

  gpu_run(cudaStreamSynchronize(mmc->strm));
  std::cout << "MemMoveDevice:close2" << std::endl;
  gpu_run(cudaStreamDestroy(mmc->strm));
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
  std::cout << "MemMoveDevice:close1" << std::endl;

  // delete mmc->worker;
  // delete mmc;
  mmc->~MemMoveConf();
  MemoryManager::freePinned(mmc);
  eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_END);
}

extern "C" {
MemMoveDevice::workunit *acquireWorkUnit(MemMoveDevice::MemMoveConf *mmc) {
  MemMoveDevice::workunit *ret = nullptr;
#ifndef NDEBUG
  bool popres =
#endif
      mmc->idle.pop(ret);
  assert(popres);
  return ret;
}

void propagateWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                       MemMoveDevice::workunit *buff, bool is_noop) {
  // if (!is_noop)
  // gpu_run(cudaEventRecord(buff->event, mmc->strm));
  // gpu_run(cudaEventDestroy(buff->event));
  // gpu_run(cudaEventCreate(&(buff->event)));
  // gpu_run(cudaEventRecord(buff->event, mmc->strm));
  // gpu_run(cudaStreamSynchronize(mmc->strm));
  // std::cout << (void *) buff->event << " " << (void *) mmc->strm <<
  // std::endl;
  // std::cout << "rec" << (void *) buff->event << std::endl;

  // gpu_run(cudaEventSynchronize(buff->event));
  // gpu_run(cudaEventDestroy(buff->event));
  // gpu_run(cudaEventRecord(buff->event, mmc->strm));
  // gpu_run(cudaEventSynchronize(buff->event));
  // gpu_run(cudaEventSynchronize(buff->event));
  // gpu_run(cudaEventRecord(buff->event, mmc->strm));
  // gpu_run(cudaStreamWaitEvent(buff->strm, buff->event, 0));
  // std::cout << "asdasdasD" << __rdtsc() << " " << (void *) buff->event <<
  // std::endl; gpu_run(cudaStreamSynchronize(buff->strm));
  // gpu_run(cudaStreamSynchronize(buff->strm));
  // gpu_run(cudaStreamSynchronize(buff->strm));
  // gpu_run(cudaStreamDestroy(buff->strm));
  if (!is_noop) gpu_run(cudaEventRecord(buff->event, mmc->strm));

  mmc->tran.push(buff);
}

bool acquirePendingWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                            MemMoveDevice::workunit **ret) {
  if (!mmc->tran.pop(*ret)) return false;
  gpu_run(cudaEventSynchronize((*ret)->event));
  // gpu_run(cudaStreamSynchronize((*ret)->strm));
  // gpu_run(cudaStreamDestroy((*ret)->strm));
  // gpu_run(cudaStreamSynchronize((*ret)->strm));
  // gpu_run(cudaEventSynchronize((*ret)->event));
  // gpu_run(cudaEventDestroy((*ret)->event));
  // gpu_run(cudaStreamSynchronize((*ret)->strm));
  // gpu_run(cudaStreamSynchronize((*ret)->strm));
  // gpu_run(cudaEventRecord((*ret)->event, mmc->strm));
  // gpu_run(cudaStreamWaitEvent((*ret)->strm, (*ret)->event, 0));
  // gpu_run(cudaStreamSynchronize((*ret)->strm));
  // std::cout << "asdasdasD" << __rdtsc() << " " << (void *) (*ret)->event <<
  // std::endl; gpu_run(cudaStreamSynchronize(mmc->strm)); std::cout <<
  // "asdasdasD" << __rdtsc() << " " << (void *) (*ret)->event << std::endl;
  return true;
}

void releaseWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                     MemMoveDevice::workunit *buff) {
  mmc->idle.push(buff);
}
}

void MemMoveDevice::catcher(MemMoveConf *mmc, int group_id,
                            exec_location target_dev) {
  // std::cout << target_dev. << std::endl;
  set_exec_location_on_scope d(target_dev);
  std::this_thread::yield();

  nvtxRangePushA("memmove::catch");

  Pipeline *pip = catch_pip->getPipeline(group_id);

  nvtxRangePushA("memmove::catch_open");
  pip->open();
  nvtxRangePop();

  {
    do {
      MemMoveDevice::workunit *p = nullptr;
      // eventlogger.log(this, log_op::MEMMOVE_CONSUME_WAIT_START);
      if (!acquirePendingWorkUnit(mmc, &p)) break;
      // eventlogger.log(this, log_op::MEMMOVE_CONSUME_WAIT_END  );
      // ++cnt;
      // std::cout << (void *) p->event << " " << (void *) mmc->strm <<
      // std::endl; gpu_run(cudaStreamSynchronize(mmc->strm));
      nvtxRangePushA("memmove::catch_cons");
      // N += ((int64_t *) p->data)[2];
      // std::cout << *((void **) p->data) << " " << get_device(*((void **)
      // p->data)) << " Started.............................." << std::endl;
      // size_t x = ((int64_t *) p->data)[2];
      // int32_t k = 0;
      // for (size_t i = 0 ; i < x ; ++i){
      //     k += ((int32_t **) p->data)[1][i];
      // }
      // sum2 += k;
      // std::cout << "s" << ((int32_t **) p->data)[0] << " " << k << std::endl;
      eventlogger.log(this, log_op::MEMMOVE_CONSUME_START);
      pip->consume(0, p->data);
      eventlogger.log(this, log_op::MEMMOVE_CONSUME_END);
      // // size_t x = ((int64_t *) p->data)[2];
      // for (size_t i = 0 ; i < x ; ++i){
      //     sum += ((int32_t **) p->data)[1][i];
      // }
      // std::cout << *((void **) p->data) << " " << get_device(*((void **)
      // p->data)) << " Finished............................." << std::endl;
      nvtxRangePop();

      releaseWorkUnit(mmc, p);  // FIXME: move this inside the generated code
    } while (true);
  }

  nvtxRangePushA("memmove::catch_close");
  pip->close();
  nvtxRangePop();

  nvtxRangePop();
}
