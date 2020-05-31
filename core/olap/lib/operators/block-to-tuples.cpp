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
#include "block-to-tuples.hpp"

#include "lib/util/catalog.hpp"
#include "lib/util/jit/pipeline.hpp"
#include "memory/buffer-manager.cuh"
#include "memory/memory-manager.hpp"
#include "olap/plugins/plugins.hpp"
#include "util/logging.hpp"

using namespace llvm;

void BlockToTuples::produce_(ParallelContext *context) {
  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  for (const auto &wantedField : wantedFields) {
    old_buffs.push_back(context->appendStateVar(llvm::PointerType::getUnqual(
        RecordAttribute{wantedField.getRegisteredAs(), true}.getLLVMType(
            context->getLLVMContext()))));
  }

  getChild()->produce(context);
}

void BlockToTuples::nextEntry(llvm::Value *mem_itemCtr,
                              ParallelContext *context) {
  // Prepare
  auto Builder = context->getBuilder();

  // Increment and store back

  Value *val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);

  Value *inc;
  if (gpu && granularity == gran_t::GRID) {
    inc = Builder->CreateIntCast(context->threadNum(),
                                 val_curr_itemCtr->getType(), false);
  } else {
    inc = ConstantInt::get((IntegerType *)val_curr_itemCtr->getType(), 1);
  }

  Value *val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, inc);
  Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

void BlockToTuples::consume(ParallelContext *context,
                            const OperatorState &childState) {
  // Prepare
  LLVMContext &llvmContext = context->getLLVMContext();
  IRBuilder<> *Builder = context->getBuilder();

  // TODO move into context
  expressions::ProteusValueExpression tid{
      new Int64Type(), {context->threadId(), context->createFalse()}};

  auto is_leader = eq(tid, INT64_C(0));

  // FIXME: do we need that?
  // Value *activemask;
  // if (dynamic_cast<GpuPipelineGen *>(context->getCurrentPipeline())) {
  //   activemask = gpu_intrinsic::activemask(context);
  // }

  // FIXME: assumes thread 0 gets to execute block2tuples
  context->gen_if(is_leader, childState)([&]() {
    Type *charPtrType = Type::getInt8PtrTy(llvmContext);
    for (size_t i = 0; i < wantedFields.size(); ++i) {
      RecordAttribute attr{wantedFields[i].getRegisteredAs(), true};
      Value *arg = Builder->CreateLoad(childState[attr].mem);
      Value *old = Builder->CreateLoad(context->getStateVar(old_buffs[i]));
      old = Builder->CreateBitCast(old, charPtrType);

      Function *f = context->getFunction(
          "release_buffers");  // FIXME: Assumes grid launch +
                               // Assumes 1 block per kernel!
      Builder->CreateCall(f, {old});

      Builder->CreateStore(arg, context->getStateVar(old_buffs[i]));
    }
  });

  // FIXME: do we need that?
  // if (dynamic_cast<GpuPipelineGen *>(context->getCurrentPipeline())) {
  //   auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  //   Builder->CreateCall(warpsync, {activemask});
  // }

  Plugin *pg =
      Catalog::getInstance().getPlugin(wantedFields[0].getRegisteredRelName());

  RecordAttribute tupleCnt{wantedFields[0].getRegisteredRelName(), "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?
  Value *cnt = Builder->CreateLoad(childState[tupleCnt].mem, "cnt");

  auto mem_itemCtr = context->CreateEntryBlockAlloca("i_ptr", cnt->getType());
  Builder->CreateStore(
      Builder->CreateIntCast(context->threadId(), cnt->getType(), false),
      mem_itemCtr);

  Value *lhs;

  /**
   * Equivalent:
   * while(itemCtr < size)
   */
  context->gen_while([&]() {
    lhs = Builder->CreateLoad(mem_itemCtr, "i");

    return ProteusValue{Builder->CreateICmpSLT(lhs, cnt),
                        context->createFalse()};
  })([&](BranchInst *loop_cond) {
    MDNode *LoopID;

    {
      MDString *vec_st =
          MDString::get(llvmContext, "llvm.loop.vectorize.enable");
      Type *int1Type = Type::getInt1Ty(llvmContext);
      Metadata *one = ConstantAsMetadata::get(ConstantInt::get(int1Type, 1));
      llvm::Metadata *vec_en[] = {vec_st, one};
      MDNode *vectorize_enable = MDNode::get(llvmContext, vec_en);

      MDString *itr_st =
          MDString::get(llvmContext, "llvm.loop.interleave.count");
      Type *int32Type = Type::getInt32Ty(llvmContext);
      Metadata *count = ConstantAsMetadata::get(ConstantInt::get(int32Type, 4));
      llvm::Metadata *itr_en[] = {itr_st, count};
      MDNode *interleave_count = MDNode::get(llvmContext, itr_en);

      llvm::Metadata *Args[] = {nullptr, vectorize_enable, interleave_count};
      LoopID = MDNode::get(llvmContext, Args);
      LoopID->replaceOperandWith(0, LoopID);

      loop_cond->setMetadata(LLVMContext::MD_loop, LoopID);
    }

    // Get the 'oid' of each record and pass it along.
    // More general/lazy plugins will only perform this action,
    // instead of eagerly 'converting' fields
    // FIXME This action corresponds to materializing the oid. Do we want this?
    RecordAttribute tupleIdentifier{wantedFields[0].getRegisteredRelName(),
                                    activeLoop, pg->getOIDType()};

    ProteusValueMemory mem_posWrapper{mem_itemCtr, context->createFalse()};

    std::map<RecordAttribute, ProteusValueMemory> variableBindings;
    variableBindings[tupleIdentifier] = mem_posWrapper;
    for (const auto &w : wantedFields) {
      RecordAttribute tIdentifier{w.getRegisteredRelName(), activeLoop,
                                  pg->getOIDType()};
      variableBindings[tIdentifier] = mem_posWrapper;
    }

    // Actual Work (Loop through attributes etc.)
    for (const auto &field : wantedFields) {
      RecordAttribute attr{field.getRegisteredAs(), true};

      Value *arg = Builder->CreateLoad(childState[attr].mem);

      auto bufVarStr = field.getRegisteredRelName();
      auto currBufVar = bufVarStr + "." + attr.getAttrName();

      Value *ptr = Builder->CreateGEP(arg, lhs);

      // Function    * pfetch =
      // Intrinsic::getDeclaration(Builder->GetInsertBlock()->getParent()->getParent(),
      // Intrinsic::prefetch);

      // Instruction * ins = Builder->CreateCall(pfetch, std::vector<Value*>{
      //                     Builder->CreateBitCast(ptr, charPtrType),
      //                     context->createInt32(0),
      //                     context->createInt32(3),
      //                     context->createInt32(1)}
      //                     );
      // {
      //     ins->setMetadata("llvm.mem.parallel_loop_access", LoopID);
      // }

      if (!pg->isLazy()) {
        // If not lazy, load the data now
        Instruction *parsed =
            Builder->CreateLoad(ptr);  // TODO : use CreateAlignedLoad
        {
          parsed->setMetadata(LLVMContext::MD_mem_parallel_loop_access, LoopID);
        }
        ptr = parsed;
      }

      variableBindings[field.getRegisteredAs()] =
          context->toMem(ptr, context->createFalse());
    }

    // Triggering parent
    OperatorState state{*this, variableBindings};
    getParent()->consume(context, state);

    nextEntry(mem_itemCtr, context);
  });
}

void BlockToTuples::open(Pipeline *pip) {
  eventlogger.log(this, log_op::BLOCK2TUPLES_OPEN_START);

  void **buffs;

  if (gpu) {
    buffs =
        (void **)MemoryManager::mallocGpu(sizeof(void *) * wantedFields.size());
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    gpu_run(
        cudaMemsetAsync(buffs, 0, sizeof(void *) * wantedFields.size(), strm));
    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));
  } else {
    buffs = (void **)MemoryManager::mallocPinned(sizeof(void *) *
                                                 wantedFields.size());
    memset(buffs, 0, sizeof(void *) * wantedFields.size());
  }

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    pip->setStateVar<void *>(old_buffs[i], buffs + i);
  }
  eventlogger.log(this, log_op::BLOCK2TUPLES_OPEN_END);
}

void BlockToTuples::close(Pipeline *pip) {
  void **h_buffs;
  void **buffs = pip->getStateVar<void **>(old_buffs[0]);

  if (gpu) {
    h_buffs = (void **)malloc(sizeof(void *) * wantedFields.size());
    cudaStream_t strm;
    gpu_run(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    gpu_run(cudaMemcpyAsync(h_buffs, buffs,
                            sizeof(void *) * wantedFields.size(),
                            cudaMemcpyDefault, strm));
    gpu_run(cudaStreamSynchronize(strm));
    gpu_run(cudaStreamDestroy(strm));
  } else {
    h_buffs = buffs;
  }

  for (size_t i = 0; i < wantedFields.size(); ++i) {
    buffer_manager<int32_t>::release_buffer((int32_t *)h_buffs[i]);
  }

  if (gpu)
    MemoryManager::freeGpu(buffs);
  else
    MemoryManager::freePinned(buffs);

  if (gpu) free(h_buffs);
}
