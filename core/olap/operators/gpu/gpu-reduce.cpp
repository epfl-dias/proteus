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

#include "operators/gpu/gpu-reduce.hpp"

#include "operators/gpu/gmonoids.hpp"
#include "util/jit/pipeline.hpp"

using namespace llvm;

namespace opt {

GpuReduce::GpuReduce(std::vector<agg_t> accs, expression_t pred,
                     Operator *child)
    : Reduce(std::move(accs), std::move(pred), child) {
  for (const auto &expr : this->aggs) {
    if (!expr.getExpressionType()->isPrimitive()) {
      string error_msg("[GpuReduce: ] Currently only supports primitive types");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
}

void GpuReduce::consume(ParallelContext *context,
                        const OperatorState &childState) {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Generate condition
  ExpressionGeneratorVisitor predExprGenerator{context, childState};
  ProteusValue condition = pred.accept(predExprGenerator);
  /**
   * Predicate Evaluation:
   */
  BasicBlock *entryBlock = Builder->GetInsertBlock();
  BasicBlock *endBlock =
      BasicBlock::Create(llvmContext, "reduceCondEnd", TheFunction);
  BasicBlock *ifBlock;
  context->CreateIfBlock(context->getGlobalFunction(), "reduceIfCond", &ifBlock,
                         endBlock);

  /**
   * IF(pred) Block
   */
  Builder->SetInsertPoint(entryBlock);

  Builder->CreateCondBr(condition.value, ifBlock, endBlock);

  Builder->SetInsertPoint(ifBlock);

  /* Time to Compute Aggs */
  auto itAcc = aggs.begin();
  auto itMem = mem_accumulators.begin();

  for (; itAcc != aggs.end(); itAcc++, itMem++) {
    auto agg = *itAcc;
    Value *mem_accumulating = nullptr;

    switch (agg.getMonoid()) {
      case SUM:
      case MULTIPLY:
      case MAX:
      case OR:
      case AND: {
        BasicBlock *cBB = Builder->GetInsertBlock();
        Builder->SetInsertPoint(context->getCurrentEntryBlock());

        mem_accumulating = context->getStateVar(*itMem);

        Constant *acc_init = getIdentityElementIfSimple(
            agg.getMonoid(), agg.getExpressionType(), context);
        Value *acc_mem =
            context->CreateEntryBlockAlloca("acc", acc_init->getType());
        Builder->CreateStore(acc_init, acc_mem);

        Builder->SetInsertPoint(context->getEndingBlock());
        generate(agg, context, childState, acc_mem, mem_accumulating);

        Builder->SetInsertPoint(cBB);

        ExpressionGeneratorVisitor outputExprGenerator{context, childState};

        // Load accumulator -> acc_value
        ProteusValue acc_value{Builder->CreateLoad(acc_mem),
                               context->createFalse()};

        // new_value = acc_value op outputExpr
        expressions::ProteusValueExpression val{agg.getExpressionType(),
                                                acc_value};
        auto upd = agg.toReduceExpression(val);
        ProteusValue new_val = upd.accept(outputExprGenerator);

        // store new_val to accumulator
        Builder->CreateStore(new_val.value, acc_mem);
        break;
      }
      case BAGUNION:
        generateBagUnion(agg.getExpression(), context, childState,
                         context->getStateVar(*itMem));
        break;
      case APPEND:
        //      generateAppend(context, childState);
        //      break;
      case UNION:
      default: {
        string error_msg =
            string("[Reduce: ] Unknown / Still Unsupported accumulator");
        LOG(ERROR) << error_msg;
        throw runtime_error(error_msg);
      }
    }
  }

  Builder->CreateBr(endBlock);

  /**
   * END Block
   */
  Builder->SetInsertPoint(endBlock);
}

void GpuReduce::generateBagUnion(expression_t outputExpr,
                                 Context *const context,
                                 const OperatorState &state,
                                 Value *cnt_mem) const {
  auto error_msg = "[Reduce: ] Unknown / Still Unsupported accumulator";
  LOG(ERROR) << error_msg;
  throw runtime_error(error_msg);
}

void GpuReduce::generate(const agg_t &agg, ParallelContext *context,
                         const OperatorState &state, Value *mem_accumulating,
                         Value *global_accumulator_ptr) const {
  IRBuilder<> *Builder = context->getBuilder();
  LLVMContext &llvmContext = context->getLLVMContext();
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  gpu::Monoid *gm = gpu::Monoid::get(agg.getMonoid());

  global_accumulator_ptr->setName("reduce_" + std::to_string(*gm) + "_ptr");

  BasicBlock *entryBlock = Builder->GetInsertBlock();

  BasicBlock *endBlock = context->getEndingBlock();
  Builder->SetInsertPoint(endBlock);

  Value *val_accumulating = Builder->CreateLoad(mem_accumulating);

  // Warp aggregate
  Value *aggr = gm->createWarpAggregateTo0(context, val_accumulating);

  // Store warp-aggregated final result (available to all threads of each warp)
  Builder->CreateStore(aggr, mem_accumulating);

  // Write to global accumulator only from a single thread per warp

  BasicBlock *laneendBlock =
      BasicBlock::Create(llvmContext, "reduceWriteEnd", TheFunction);
  BasicBlock *laneifBlock =
      context->CreateIfBlock(TheFunction, "reduceWriteIf", laneendBlock);

  Value *laneid = context->laneId();
  Builder->CreateCondBr(
      Builder->CreateICmpEQ(laneid, ConstantInt::get(laneid->getType(), 0),
                            "is_pivot"),
      laneifBlock, laneendBlock);

  Builder->SetInsertPoint(laneifBlock);

  gm->createAtomicUpdate(context, global_accumulator_ptr, aggr,
                         llvm::AtomicOrdering::Monotonic);

  Builder->CreateBr(laneendBlock);
  context->setEndingBlock(laneendBlock);

  Builder->SetInsertPoint(entryBlock);
}

}  // namespace opt
