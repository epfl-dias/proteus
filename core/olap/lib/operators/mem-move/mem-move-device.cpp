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

#include "mem-move-device.hpp"

#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/logging.hpp>
#include <platform/util/timing.hpp>

#include "lib/util/catalog.hpp"
#include "lib/util/jit/pipeline.hpp"

buff_pair buff_pair::not_moved(proteus::managed_ptr buff) {
  return {std::move(buff), nullptr};
}

bool buff_pair::moved() const { return !old_buff; }

proteus::managed_ptr MemMoveDevice::MemMoveConf::force_push(
    const proteus::managed_ptr &src, size_t bytes, int target_device,
    uint64_t srcServer, cudaStream_t movestrm) {
  // FIXME: buffer manager should be able to provide blocks of arbitrary size
  assert(bytes <= BlockManager::block_size);
  auto buff = BlockManager::h_get_buffer(target_device);

  if (bytes > 0) {
    BlockManager::overwrite_bytes(buff.get(), src.get(), bytes, movestrm,
                                  false);
  }

  return buff;
}

buff_pair MemMoveDevice::MemMoveConf::push(proteus::managed_ptr src,
                                           size_t bytes, int target_device,
                                           uint64_t srcServer) {
  assert(srcServer == 0);
  const auto *d = topology::getInstance().getGpuAddressed(src.get());
  int dev = d ? static_cast<int>(d->id) : -1;

  if (dev == target_device) {
    return buff_pair::not_moved(std::move(src));  // block in correct device
  }

  auto buff = force_push(src, bytes, target_device, srcServer, strm);
  return buff_pair{std::move(buff), std::move(src)};
}

extern "C" {
pb make_mem_move_device(char *src, size_t bytes, int target_device,
                        uint64_t srcServer,
                        MemMoveDevice::MemMoveConf *mmc) noexcept {
  auto x =
      mmc->push(proteus::managed_ptr{src}, bytes, target_device, srcServer);
  return {x.new_buff.release(), x.old_buff.release()};
}

MemMoveDevice::workunit *acquireWorkUnit(
    MemMoveDevice::MemMoveConf *mmc) noexcept {
  return mmc->acquire();
}

void propagateWorkUnit(MemMoveDevice::MemMoveConf *mmc,
                       MemMoveDevice::workunit *buff, bool is_noop) noexcept {
  mmc->propagate(buff, is_noop);
}
}

void MemMoveDevice::genReleaseOldBuffer(ParallelContext *context,
                                        llvm::Value *src) const {
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());
  context->gen_call(release_buffer,
                    {context->getBuilder()->CreateBitCast(src, charPtrType)});
}

void MemMoveDevice::produce_(ParallelContext *context) {
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = llvm::Type::getInt32Ty(context->getLLVMContext());
  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  auto oidType = pg->getOIDType()->getLLVMType(llvmContext);

  std::vector<llvm::Type *> tr_types;
  for (auto wantedField : wantedFields) {
    tr_types.push_back(wantedField->getLLVMType(llvmContext));
    tr_types.push_back(
        wantedField->getLLVMType(llvmContext));  // old buffer, to be released
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

  auto Builder = context->getBuilder();
  auto entryBB = Builder->GetInsertBlock();
  auto F = entryBB->getParent();

  auto mainBB = llvm::BasicBlock::Create(llvmContext, "main", F);

  auto endBB = llvm::BasicBlock::Create(llvmContext, "end", F);
  context->setEndingBlock(endBB);

  Builder->SetInsertPoint(entryBB);

  auto params = Builder->CreateLoad(
      context->getArgument(p)->getType()->getPointerElementType(),
      context->getArgument(p));

  map<RecordAttribute, ProteusValueMemory> variableBindings;

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    auto param = Builder->CreateExtractValue(params, 2 * i);

    auto src = Builder->CreateExtractValue(params, 2 * i + 1);

    genReleaseOldBuffer(context, src);

    variableBindings[*(wantedFields[i])] =
        context->toMem(param, context->createFalse());
  }

  auto cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size());

  variableBindings[tupleCnt] = context->toMem(cnt, context->createFalse());

  auto oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);

  variableBindings[tupleIdentifier] =
      context->toMem(oid, context->createFalse());

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

  context->popPipeline();

  catch_pip = context->removeLatestPipeline();

  // push new pipeline for the throw part
  context->pushPipeline();

  device_id_var = context->appendStateVar(int32_type);
  memmvconf_var = context->appendStateVar(charPtrType);

  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  getChild()->produce(context);
}

ProteusValueMemory MemMoveDevice::getServerId(
    ParallelContext *context, const OperatorState &childState) const {
  return context->toMem(context->createInt64(0), context->createFalse());
}

void MemMoveDevice::consume(ParallelContext *context,
                            const OperatorState &childState) {
  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();

  auto charPtrType = llvm::Type::getInt8PtrTy(context->getLLVMContext());

  auto workunit_type = llvm::StructType::get(
      llvmContext, std::vector<llvm::Type *>{charPtrType, charPtrType});

  // Find block size
  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper = childState[tupleCnt];

  ProteusValueMemory mem_srcServer = getServerId(context, childState);

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  auto device_id = ((ParallelContext *)context)->getStateVar(device_id_var);

  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(
      mem_cntWrapper.mem->getType()->getPointerElementType(),
      mem_cntWrapper.mem);
  auto srcS = Builder->CreateLoad(
      mem_srcServer.mem->getType()->getPointerElementType(), mem_srcServer.mem);

  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};

  ProteusValueMemory mem_oidWrapper = childState[tupleIdentifier];
  llvm::Value *oid = Builder->CreateLoad(
      mem_oidWrapper.mem->getType()->getPointerElementType(),
      mem_oidWrapper.mem);

  llvm::Value *memmv = ((ParallelContext *)context)->getStateVar(memmvconf_var);

  std::vector<llvm::Value *> pushed;
  llvm::Value *is_noop = context->createTrue();
  for (size_t i = 0; i < wantedFields.size(); ++i) {
    RecordAttribute block_attr(*(wantedFields[i]), true);

    ProteusValueMemory mem_valWrapper = childState[block_attr];

    auto mv = Builder->CreateBitCast(
        Builder->CreateLoad(
            mem_valWrapper.mem->getType()->getPointerElementType(),
            mem_valWrapper.mem),
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
      // Do actual mem move
      auto moved_buffpair = context->gen_call(
          make_mem_move_device, {mv, size, device_id, srcS, memmv});
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

  // acquire and set workunit values
  auto workunit_ptr8 = context->gen_call(acquireWorkUnit, {memmv});
  auto workunit_ptr = Builder->CreateBitCast(
      workunit_ptr8, llvm::PointerType::getUnqual(workunit_type));

  auto workunit_dat = Builder->CreateLoad(
      workunit_ptr->getType()->getPointerElementType(), workunit_ptr);
  auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
  d_ptr =
      Builder->CreateBitCast(d_ptr, llvm::PointerType::getUnqual(data_type));
  Builder->CreateStore(d, d_ptr);

  // finally propagate the workunit
  context->gen_call(propagateWorkUnit, {memmv, workunit_ptr8, is_noop});
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
  auto *wu = (workunit *)MemoryManager::mallocPinned(sizeof(workunit) * slack);

  // nvtxRangePushA("memmove::open");
  cudaStream_t strm = createNonBlockingStream();

  eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

  MemMoveConf *mmc = createMoveConf();

#ifndef NCUDA
  mmc->strm = strm;
#endif
  mmc->slack = slack;
  mmc->data_buffs = MemoryManager::mallocPinned(data_size * slack);
  char *data_buff = (char *)mmc->data_buffs;
  for (size_t i = 0; i < slack; ++i) {
    wu[i].data = ((void *)(data_buff + i * data_size));
    mmc->idle.push(wu + i);
  }
  // nvtxRangePushA("memmove::open2");
  for (size_t i = 0; i < slack; ++i) {
    gpu_run(cudaEventCreateWithFlags(
        &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
  }
  // nvtxRangePop();

  mmc->worker = ThreadPool::getInstance().enqueue(
      &MemMoveDevice::catcher, this, mmc, pip->getGroup(), exec_location{},
      pip->getSession());
  eventlogger.log(this, log_op::MEMMOVE_OPEN_END);

  pip->setStateVar<int>(device_id_var, getTargetDevice());

  pip->setStateVar<void *>(memmvconf_var, mmc);
  // nvtxRangePop();
}

int MemMoveDevice::getTargetDevice() const {
  if (to_cpu) return -1;
  return topology::getInstance().getActiveGpu().id;
}

void MemMoveDevice::close(Pipeline *pip) {
  eventlogger.log(this, log_op::MEMMOVE_CLOSE_START);
  auto *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

  mmc->tran.close();

  nvtxRangePop();
  mmc->worker.get();

  eventlogger.log(this, log_op::MEMMOVE_CLOSE_END);

  eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_START);
  syncAndDestroyStream(mmc->strm);

  nvtxRangePushA("MemMoveDev_running2");
  nvtxRangePushA("MemMoveDev_running");

  nvtxRangePushA("MemMoveDev_release");
  workunit *start_wu = nullptr;
  for (size_t i = 0; i < slack; ++i) {
    workunit *wu = mmc->idle.pop_unsafe();
    gpu_run(cudaEventDestroy(wu->event));
    if (i == 0 || wu < start_wu) start_wu = wu;
  }
  nvtxRangePop();
  nvtxRangePop();

  MemoryManager::freePinned(mmc->data_buffs);
  MemoryManager::freePinned(start_wu);

  mmc->idle.close();

  destroyMoveConf(mmc);
  eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_END);
}

void MemMoveDevice::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                           bool is_noop) {
  if (!is_noop) gpu_run(cudaEventRecord(buff->event, strm));

  tran.push(buff);
}

MemMoveDevice::workunit *MemMoveDevice::MemMoveConf::acquire() {
  // time_block t{"pop: "};
  // LOG(INFO) << "pop";
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
  // LOG(INFO) << "pushed";
  // time_block t{"pushed: "};
  idle.push(buff);
}

void MemMoveDevice::catcher(MemMoveConf *mmc, int group_id,
                            exec_location target_dev, const void *session) {
  set_exec_location_on_scope d(target_dev);
  std::this_thread::yield();

  nvtxRangePushA("memmove::catch");

  auto pip = catch_pip->getPipeline(group_id);

  nvtxRangePushA("memmove::catch_open");
  pip->open(session);
  nvtxRangePop();

  {
    do {
      MemMoveDevice::workunit *p = nullptr;
      if (!mmc->getPropagated(&p)) break;
      for (size_t i = 0; i < wantedFields.size(); ++i) {
        ((proteus::managed_ptr *)(p->data))[i * 2] =
            mmc->pull(std::move(((proteus::managed_ptr *)(p->data))[i * 2]));
      }

      nvtxRangePushA("memmove::catch_cons");
      pip->consume(0, p->data);
      nvtxRangePop();

      mmc->release(p);
    } while (true);
  }

  event_range<range_log_op::MEMMOVE_OPEN> er{this, catch_pip, pip->getGroup()};
  nvtxRangePushA("memmove::catch_close");
  pip->close();
  nvtxRangePop();

  nvtxRangePop();
}
