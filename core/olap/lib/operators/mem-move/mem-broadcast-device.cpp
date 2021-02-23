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

#include "mem-broadcast-device.hpp"

#include <lib/util/catalog.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/threadpool/threadpool.hpp>

#include "lib/util/jit/pipeline.hpp"

using namespace llvm;

void MemBroadcastDevice::MemBroadcastConf::prepareNextBatch() {
  //  if (!to_cpu) return;
  for (size_t i = 0; i < 16; ++i) {
    //    BlockManager::release_buffer(targetbuffer[i]);
    // FIXME: can be much much simpler and optimal if codegen'ed
    targetbuffer[i] = nullptr;
  }
}

buff_pair MemBroadcastDevice::MemBroadcastConf::pushBroadcast(
    const proteus::managed_ptr &src, size_t bytes, int target_device,
    bool disable_noop) {
  const auto &topo = topology::getInstance();

  // FIXME: CPU handling is a little bit broken
  auto target_device_offset =
      to_cpu
          ? target_device % (int)topology::getInstance().getCpuNumaNodeCount()
          : target_device;

  // FIXME: That's a hack until we share GPU buffers correctly
  auto force_share =
      dynamic_cast<const topology::gpunode *>(
          &topology::getInstance().getNumaAddressed(src.get())) !=
      nullptr;  // overwrites always_share!

  auto on_target_numa = [&]() {
    auto &src_numa = topology::getInstance().getNumaAddressed(src.get());
    auto &tgt_numa =
        to_cpu
            ? static_cast<const topology::numanode &>(
                  topology::getInstance().getCpuNumaNodes().at(
                      target_device_offset))
            : static_cast<const topology::numanode &>(
                  topology::getInstance().getGpus().at(target_device_offset));

    return src_numa == tgt_numa;
  };

  if (!force_share && (always_share || on_target_numa())) {
    // just share buffer, no movement
    targetbuffer[target_device_offset] = src.get();
  } else {
    if (!targetbuffer[target_device_offset] || force_share) {
      auto buff = force_push(src, bytes, (to_cpu) ? -1 : target_device_offset,
                             0, strm.at(target_device_offset));
      targetbuffer[target_device_offset] = buff.get();
      // bypass normal path, to avoid weird ownership manipulations
      return buff_pair::not_moved(std::move(buff));
    }
  }

  // This is also weird though... essentially targetbuffer is a currently
  // non-owning cache, for which we know the pointers are still active
  // somewhere, as we are still creating the transfer descriptions.
  proteus::managed_ptr tb{targetbuffer[target_device_offset]};
  auto ret = BlockManager::share_host_buffer(tb);
  tb.release();
  return {std::move(ret), nullptr};
}

extern "C" {
void step_mmc_mem_move_broadcast_device(
    MemBroadcastDevice::MemBroadcastConf *mmc) noexcept {
  mmc->prepareNextBatch();
}

pb make_mem_move_broadcast_device(char *src, size_t bytes, int target_device,
                                  MemBroadcastDevice::MemBroadcastConf *mmc,
                                  bool disable_noop) noexcept {
  auto tmp =
      mmc->pushBroadcast(reinterpret_cast<const proteus::managed_ptr &>(src),
                         bytes, target_device, disable_noop);
  return {tmp.new_buff.release(), tmp.old_buff.release()};
}

void propagateWorkUnitBroadcast(MemBroadcastDevice::MemBroadcastConf *mmc,
                                MemBroadcastDevice::workunit *buff,
                                int target_device) noexcept {
  mmc->propagateBroadcast(buff, target_device);
}
}

void MemBroadcastDevice::produce_(ParallelContext *context) {
  auto &llvmContext = context->getLLVMContext();
  auto int32_type = Type::getInt32Ty(context->getLLVMContext());
  auto charPtrType = Type::getInt8PtrTy(context->getLLVMContext());

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  auto oidType = pg->getOIDType()->getLLVMType(llvmContext);

  std::vector<llvm::Type *> tr_types;
  for (auto wantedField : wantedFields) {
    tr_types.push_back(wantedField->getLLVMType(llvmContext));
    tr_types.push_back(
        wantedField->getLLVMType(llvmContext));  // old buffer, to be released
  }
  tr_types.push_back(int32_type);
  tr_types.push_back(oidType);  // cnt
  tr_types.push_back(oidType);  // oid

  data_type = StructType::get(llvmContext, tr_types);

  RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?
  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};
  RecordAttribute tupleTarget{wantedFields[0]->getRelationName(),
                              "__broadcastTarget", new IntType()};

  // Generate catch code
  auto p = context->appendParameter(PointerType::get(data_type, 0), true, true);
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
    auto param = Builder->CreateExtractValue(params, 2 * i);

    auto src = Builder->CreateExtractValue(params, 2 * i + 1);

    Builder->CreateCall(release, Builder->CreateBitCast(src, charPtrType));

    // FIMXE: should we alse transfer this information ?
    variableBindings[*(wantedFields[i])] =
        context->toMem(param, context->createFalse());
  }

  auto target = Builder->CreateExtractValue(params, 2 * wantedFields.size());
  variableBindings[tupleTarget] =
      context->toMem(target, context->createFalse());

  // FIMXE: should we alse transfer this information ?
  auto cnt = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 1);
  variableBindings[tupleCnt] = context->toMem(cnt, context->createFalse());

  // FIMXE: should we alse transfer this information ?
  auto oid = Builder->CreateExtractValue(params, 2 * wantedFields.size() + 2);
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

void MemBroadcastDevice::consume(ParallelContext *context,
                                 const OperatorState &childState) {
  // Prepare
  auto &llvmContext = context->getLLVMContext();
  auto Builder = context->getBuilder();
  auto insBB = Builder->GetInsertBlock();

  auto charPtrType = Type::getInt8PtrTy(llvmContext);

  // auto cpp_bool_type = Type::getInt8Ty(llvmContext);
  static_assert(sizeof(bool) == 1, "Fix type above");

  auto workunit_type = StructType::get(
      llvmContext, std::vector<llvm::Type *>{charPtrType, charPtrType});

  // Find block size
  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0]->getRelationName());
  RecordAttribute tupleCnt{wantedFields[0]->getRelationName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?

  ProteusValueMemory mem_cntWrapper = childState[tupleCnt];

  Builder->SetInsertPoint(context->getCurrentEntryBlock());

  // auto device_id = ((ParallelContext *)context)->getStateVar(device_id_var);
  // auto  cu_stream       = ((ParallelContext *)
  // context)->getStateVar(cu_stream_var);

  Builder->SetInsertPoint(insBB);
  auto N = Builder->CreateLoad(mem_cntWrapper.mem);

  RecordAttribute tupleIdentifier{wantedFields[0]->getRelationName(),
                                  activeLoop, pg->getOIDType()};

  ProteusValueMemory mem_oidWrapper = childState[tupleIdentifier];
  auto oid = Builder->CreateLoad(mem_oidWrapper.mem);

  auto memmv = context->getStateVar(memmvconf_var);

  std::vector<std::vector<Value *>> pushed{targets.size()};

  auto null_ptr =
      llvm::ConstantPointerNull::get((llvm::PointerType *)charPtrType);
  for (auto wantedField : wantedFields) {
    RecordAttribute block_attr(*wantedField, true);

    ProteusValueMemory mem_valWrapper = childState[block_attr];

    auto block = Builder->CreateLoad(mem_valWrapper.mem);
    auto mv = Builder->CreateBitCast(block, charPtrType);

    auto block_type = mem_valWrapper.mem->getType()->getPointerElementType();
    auto mv_block_type = block_type->getPointerElementType();

    llvm::Value *size = llvm::ConstantInt::get(
        llvmContext, APInt(64, context->getSizeOf(mv_block_type)));
    auto Nloc = Builder->CreateZExtOrBitCast(N, size->getType());
    size = Builder->CreateMul(size, Nloc);

    context->gen_call(step_mmc_mem_move_broadcast_device, {memmv});

    llvm::Value *any_noop = context->createFalse();
    for (size_t t_i = 0; t_i < targets.size(); ++t_i) {
      auto target_id = context->createInt32(targets[t_i]);

      // Do actual mem move
      auto moved_buffpair =
          context->gen_call(make_mem_move_broadcast_device,
                            {mv, size, target_id, memmv, any_noop});
      auto moved = Builder->CreateExtractValue(moved_buffpair, 0);
      auto to_release = Builder->CreateExtractValue(moved_buffpair, 1);

      pushed[t_i].push_back(Builder->CreateBitCast(moved, block_type));

      any_noop = Builder->CreateOr(any_noop,
                                   Builder->CreateICmpEQ(to_release, null_ptr));

      auto null_block =
          ConstantPointerNull::get((llvm::PointerType *)block_type);

      if (t_i == targets.size() - 1) {
        //        auto rel = Builder->CreateSelect(any_noop, null_block, block);

        pushed[t_i].push_back(block);
      } else {
        pushed[t_i].push_back(null_block);
      }
    }
  }

  for (size_t t_i = 0; t_i < targets.size(); ++t_i) {
    pushed[t_i].push_back(context->createInt32(t_i));
    pushed[t_i].push_back(N);
    pushed[t_i].push_back(oid);
  }

  for (size_t t_i = 0; t_i < targets.size(); ++t_i) {
    llvm::Value *d = UndefValue::get(data_type);
    for (size_t i = 0; i < pushed[t_i].size(); ++i) {
      d = Builder->CreateInsertValue(d, pushed[t_i][i], i);
    }

    auto workunit_ptr8 = context->gen_call(acquireWorkUnit, {memmv});
    auto workunit_ptr = Builder->CreateBitCast(
        workunit_ptr8, PointerType::getUnqual(workunit_type));

    auto workunit_dat = Builder->CreateLoad(workunit_ptr);
    auto d_ptr = Builder->CreateExtractValue(workunit_dat, 0);
    d_ptr = Builder->CreateBitCast(d_ptr, PointerType::getUnqual(data_type));
    Builder->CreateStore(d, d_ptr);

    auto target_id = context->createInt32(targets[t_i]);
    context->gen_call(propagateWorkUnitBroadcast,
                      {memmv, workunit_ptr8, target_id});
  }
}

MemBroadcastDevice::MemBroadcastConf *MemBroadcastDevice::createMoveConf()
    const {
  return new MemBroadcastDevice::MemBroadcastConf;
}

void MemBroadcastDevice::open(Pipeline *pip) {
  nvtxRangePushA("memmove::open");
  // cudaStream_t strm2;
  // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

  MemBroadcastConf *mmc = createMoveConf();

#ifndef NCUDA
  if (!always_share) {
    for (const auto &t : targets) mmc->strm[t] = createNonBlockingStream();
  } else {
    mmc->strm[0] = createNonBlockingStream();
  }
#endif

  mmc->num_of_targets = targets.size();
  mmc->to_cpu = to_cpu;
  mmc->always_share = always_share;
  mmc->slack = slack;
  // mmc->next_e         = 0;
  // // mmc->events         = new cudaEvent_t[slack];
  // mmc->old_buffs      = new void      *[slack];

  auto wu = new workunit[slack];
  size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);
  // void * data_buff = malloc(data_size * slack);
  nvtxRangePushA("memmove::open2");
  for (size_t i = 0; i < slack; ++i) {
    wu[i].data =
        malloc(data_size);  //((void *) (((char *) data_buff) + i * data_size));
    gpu_run(cudaEventCreateWithFlags(
        &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
    // // gpu_run(cudaEventCreateWithFlags(&(wu[i].event),
    // cudaEventDisableTiming));//  | cudaEventBlockingSync));
    //         gpu_run(cudaEventCreate(&(wu[i].event)));
    //         gpu_run(cudaStreamCreate(&(wu[i].strm)));

    mmc->idle.push(wu + i);

    // gpu_run(cudaEventCreateWithFlags(mmc->events + i, cudaEventDisableTiming
    // | cudaEventBlockingSync)); gpu_run(cudaEventCreate(mmc->events + i));
    // mmc->old_buffs[i] = nullptr;
  }
  nvtxRangePop();

  mmc->worker = ThreadPool::getInstance().enqueue(
      &MemBroadcastDevice::catcher, this, mmc, pip->getGroup(), exec_location{},
      pip->getSession());

  int device = -1;
  if (!to_cpu) device = topology::getInstance().getActiveGpu().id;
  pip->setStateVar<int>(device_id_var, device);

  // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
  pip->setStateVar<void *>(memmvconf_var, mmc);
  nvtxRangePop();
}

void MemBroadcastDevice::close(Pipeline *pip) {
  // int device = get_device();
  // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
  auto mmc = pip->getStateVar<MemBroadcastConf *>(memmvconf_var);
  pip->setStateVar<void *>(memmvconf_var, nullptr);

  mmc->tran.close();

  nvtxRangePop();
  mmc->worker.get();

  // gpu_run(cudaStreamSynchronize(g_strm));

  // int32_t h_s;
  // gpu_run(cudaMemcpy(&h_s, s, sizeof(int32_t), cudaMemcpyDefault));
  // std::cout << "rrr" << h_s << std::endl;

  // MemoryManager::freeGpu(s);

  if (!always_share) {
    for (const auto &t : targets) syncAndDestroyStream(mmc->strm[t]);
  } else {
    syncAndDestroyStream(mmc->strm[0]);
  }

  // gpu_run(cudaStreamSynchronize(mmc->strm2));
  // gpu_run(cudaStreamDestroy    (mmc->strm2));

  nvtxRangePushA("MemMoveDev_running2");
  nvtxRangePushA("MemMoveDev_running");

  nvtxRangePushA("MemMoveDev_release");
  workunit *start_wu = nullptr;
  // void     * start_wu_data;
  for (size_t i = 0; i < slack; ++i) {
    workunit *wu = mmc->idle.pop_unsafe();

    // if (mmc->old_buffs[i]) buffer-manager<int32_t>::release_buffer((int32_t
    // *) mmc->old_buffs[i]);

    gpu_run(cudaEventDestroy(wu->event));
    // gpu_run(cudaEventDestroy(mmc->events[i]));
    free(wu->data);

    if (i == 0 || wu < start_wu) start_wu = wu;
    // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
  }
  nvtxRangePop();
  nvtxRangePop();
  // assert(mmc->tran.empty_unsafe());
  // assert(mmc->idle.empty_unsafe());
  // free(start_wu_data);
  delete[] start_wu;
  // delete[] mmc->events   ;
  // delete[] mmc->old_buffs;

  mmc->idle.close();

  // delete mmc->worker;
  delete mmc;
}

void MemBroadcastDevice::MemBroadcastConf::propagateBroadcast(
    MemMoveDevice::workunit *buff, int target_device) {
  gpu_run(cudaEventRecord(buff->event, strm[target_device]));

  tran.push(buff);
}
