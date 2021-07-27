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

#include <olap/plugins/plugins.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/util/logging.hpp>

#include "lib/util/catalog.hpp"
#include "lib/util/jit/pipeline.hpp"

using namespace llvm;

void BlockToTuples::produce_(ParallelContext *context) {
  context->registerOpen(this, [this](Pipeline *pip) { this->open(pip); });
  context->registerClose(this, [this](Pipeline *pip) { this->close(pip); });

  for (const auto &wantedField : wantedFields) {
    if (dynamic_cast<const BlockType *>(wantedField.getExpressionType())) {
      old_buffs.push_back(context->appendStateVar(llvm::PointerType::getUnqual(
          RecordAttribute{wantedField.getRegisteredAs()}.getLLVMType(
              context->getLLVMContext()))));
    }
  }

  assert(!old_buffs.empty() &&
         "There should be at least one BlockType'd argument");
  getChild()->produce(context);
}

ProteusBareValue BlockToTuples::step(ParallelContext *context,
                                     llvm::IntegerType *type) {
  auto Builder = context->getBuilder();
  Value *inc;
  if (gpu && granularity == gran_t::GRID) {
    inc = Builder->CreateIntCast(context->threadNum(), type, false);
  } else {
    inc = ConstantInt::get(type, 1);
  }
  return {inc};
}

void BlockToTuples::nextEntry(llvm::Value *mem_itemCtr,
                              ParallelContext *context) {
  // Prepare
  auto Builder = context->getBuilder();

  // Increment and store back

  Value *val_curr_itemCtr = Builder->CreateLoad(mem_itemCtr);

  Value *inc = step(context, (IntegerType *)val_curr_itemCtr->getType()).value;

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
    size_t j = 0;
    for (const auto &wantedField : wantedFields) {
      if (!dynamic_cast<const BlockType *>(wantedField.getExpressionType())) {
        continue;
      }
      RecordAttribute attr{wantedField.getRegisteredAs()};
      Value *arg = Builder->CreateLoad(childState[attr].mem);
      Value *old = Builder->CreateLoad(context->getStateVar(old_buffs[j]));
      old = Builder->CreateBitCast(old, charPtrType);

      Function *f = context->getFunction(
          "release_buffers");  // FIXME: Assumes grid launch +
                               // Assumes 1 block per kernel!
      Builder->CreateCall(f, {old});

      Builder->CreateStore(arg, context->getStateVar(old_buffs[j]));

      ++j;
    }
    assert(j == old_buffs.size());
  });

  // FIXME: do we need that?
  // if (dynamic_cast<GpuPipelineGen *>(context->getCurrentPipeline())) {
  //   auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
  //   Builder->CreateCall(warpsync, {activemask});
  // }

  auto relName = [&]() {
    if (wantedFields.empty()) {
      for (const auto &t : childState.getBindings()) {
        if (t.first.getAttrName() == "activeCnt") {
          return t.first.getRelationName();
        }
      }
      assert(false);
    } else {
      return wantedFields[0].getRegisteredRelName();
    }
  }();

  Plugin *pg = Catalog::getInstance().getPlugin(relName);

  RecordAttribute tupleCnt{relName, "activeCnt",
                           pg->getOIDType()};  // FIXME: OID type for blocks ?
  Value *cnt = Builder->CreateLoad(childState[tupleCnt].mem, "cnt");

  auto mem_itemCtr = context->CreateEntryBlockAlloca("i_ptr", cnt->getType());
  Builder->CreateStore(
      Builder->CreateIntCast(context->threadId(), cnt->getType(), false),
      mem_itemCtr);

  ProteusBareValue offset{
      Builder->CreateIntCast(context->threadId(), cnt->getType(), false)};

  pg->forEachInCollection(
      context,
      {nullptr
       /* FIXME: we should pass a real collection here, but the blocks
           are currently multiple ones, that we "zip" */
       ,
       context->createFalse()},
      offset, step(context, static_cast<IntegerType *>(cnt->getType())), {cnt},
      [&](ProteusValueMemory indexPtr, llvm::MDNode *LoopID) {
        auto lhs = Builder->CreateLoad(indexPtr.mem);

        // Get the 'oid' of each record and pass it along.
        // More general/lazy plugins will only perform this action,
        // instead of eagerly 'converting' fields
        // FIXME This action corresponds to materializing the oid. Do we want
        // this?
        RecordAttribute tupleIdentifier{relName, activeLoop, pg->getOIDType()};

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
          if (!dynamic_cast<const BlockType *>(field.getExpressionType())) {
            Value *arg =
                Builder->CreateLoad(childState[field.getRegisteredAs()].mem);
            variableBindings[field.getRegisteredAs()] =
                context->toMem(arg, context->createFalse());
            continue;
          }
          RecordAttribute attr{field.getRegisteredAs()};

          Value *arg = Builder->CreateLoad(childState[attr].mem);

          auto bufVarStr = field.getRegisteredRelName();
          auto currBufVar = bufVarStr + "." + attr.getAttrName();

          Value *ptr = Builder->CreateGEP(arg, lhs);

          // Function    * pfetch =
          // Intrinsic::getDeclaration(Builder->GetInsertBlock()->getParent()->getParent(),
          // Intrinsic::prefetch);

          // Instruction * ins = Builder->CreateCall(pfetch,
          // std::vector<Value*>{
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
              parsed->setMetadata(LLVMContext::MD_mem_parallel_loop_access,
                                  LoopID);
            }
            ptr = parsed;
          }

          variableBindings[field.getRegisteredAs()] =
              context->toMem(ptr, context->createFalse());
        }

        // Triggering parent
        OperatorState state{*this, variableBindings};
        getParent()->consume(context, state);
      });
}

void BlockToTuples::open(Pipeline *pip) {
  //  eventlogger.log(this, log_op::BLOCK2TUPLES_OPEN_START);

  void **buffs;

  if (gpu) {
    buffs =
        (void **)MemoryManager::mallocGpu(sizeof(void *) * old_buffs.size());
    cudaStream_t strm = createNonBlockingStream();
    gpu_run(cudaMemsetAsync(buffs, 0, sizeof(void *) * old_buffs.size(), strm));
    syncAndDestroyStream(strm);
  } else {
    buffs =
        (void **)MemoryManager::mallocPinned(sizeof(void *) * old_buffs.size());
    memset(buffs, 0, sizeof(void *) * old_buffs.size());
  }

  for (size_t i = 0; i < old_buffs.size(); ++i) {
    pip->setStateVar<void *>(old_buffs[i], buffs + i);
  }
  //  eventlogger.log(this, log_op::BLOCK2TUPLES_OPEN_END);
}

void BlockToTuples::close(Pipeline *pip) {
  void **h_buffs;
  void **buffs = pip->getStateVar<void **>(old_buffs.at(0));

  if (gpu) {
    h_buffs = (void **)malloc(sizeof(void *) * old_buffs.size());
    assert(h_buffs);
    cudaStream_t strm = createNonBlockingStream();
    gpu_run(cudaMemcpyAsync(h_buffs, buffs, sizeof(void *) * old_buffs.size(),
                            cudaMemcpyDefault, strm));
    syncAndDestroyStream(strm);
  } else {
    h_buffs = buffs;
  }

  for (size_t i = 0; i < old_buffs.size(); ++i) {
    BlockManager::release_buffer((int32_t *)h_buffs[i]);
  }

  if (gpu)
    MemoryManager::freeGpu(buffs);
  else
    MemoryManager::freePinned(buffs);

  if (gpu) free(h_buffs);
}
