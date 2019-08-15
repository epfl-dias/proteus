/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2018
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

#ifndef IF_STATEMENT_HPP_
#define IF_STATEMENT_HPP_

#include "common/common.hpp"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "memory/memory-allocator.hpp"
#include "values/expressionTypes.hpp"

class Context;

class if_then;

namespace expressions {
class Expression;
}
class expression_t;
class ExpressionGeneratorVisitor;
class OperatorState;

class if_branch {
 private:
  ProteusValue condition;
  Context *const context;
  llvm::BasicBlock *afterBB;

 public:
  inline constexpr if_branch(ProteusValue condition, Context *context,
                             llvm::BasicBlock *afterBB)
      : condition(condition), context(context), afterBB(afterBB) {}

  if_branch(const expression_t &expr, const OperatorState &state,
            Context *context, llvm::BasicBlock *afterBB);

 public:
  inline constexpr if_branch(ProteusValue condition, Context *context)
      : if_branch(condition, context, nullptr) {}

  // inline constexpr if_branch(    expressions::Expression *expr    ,
  //                             const OperatorState     &state    ,
  //                             Context                 *context);

  if_branch(const expression_t &expr, const OperatorState &state,
            Context *context);

 public:
  template <typename Fthen>
  if_then operator()(Fthen then);

  friend class Context;
};

class if_then {
  // template<typename Fcond, typename Fthen>
  // if_then_else(Fcond cond, Fthen then, const Context *context){

  // }

  // template<typename Fthen>
  // if_then_else(llvm::Value * cond, Fthen then, const Context *context){
  //     llvm::BasicBlock *ThenBB  = llvm::BasicBlock::Create(llvmContext,
  //     "IfThen" ); llvm::BasicBlock *AfterBB =
  //     llvm::BasicBlock::Create(llvmContext, "IfAfter");
  // }
 private:
  Context *context;
  llvm::BasicBlock *ElseBB;
  llvm::BasicBlock *AfterBB;

  llvm::IRBuilder<> *getBuilder(Context *context);
  llvm::LLVMContext &getLLVMContext(Context *context);

 public:
  template <typename Fthen>
  if_then(ProteusValue cond, Fthen then, Context *context,
          llvm::BasicBlock *endBB = nullptr)
      : context(context), AfterBB(endBB) {
    llvm::IRBuilder<> *Builder = getBuilder(context);
    llvm::LLVMContext &llvmContext = getLLVMContext(context);
    llvm::Function *F = Builder->GetInsertBlock()->getParent();

    auto ThenBB = llvm::BasicBlock::Create(llvmContext, "IfThen", F);
    ElseBB = llvm::BasicBlock::Create(llvmContext, "IfElse", F);
    if (!AfterBB) {
      AfterBB = llvm::BasicBlock::Create(llvmContext, "IfAfter", F);
    }

    // if (cond.value) {
    Builder->CreateCondBr(cond.value, ThenBB, ElseBB);

    Builder->SetInsertPoint(ThenBB);
    then();
    Builder->CreateBr(AfterBB);
    // }

    Builder->SetInsertPoint(AfterBB);
  }

  if_then(const if_then &) = delete;
  if_then(if_then &&) = delete;

  if_then &operator=(const if_then &) = delete;
  if_then &operator=(if_then &&) = delete;

  template <typename Felse>
  void gen_else(Felse felse) {
    assert(ElseBB && "gen_else* called twice in same gen_if");
    llvm::IRBuilder<> *Builder = getBuilder(context);
    Builder->SetInsertPoint(ElseBB);
    felse();
    Builder->CreateBr(AfterBB);
    Builder->SetInsertPoint(AfterBB);
    ElseBB = nullptr;
  }

  template <typename Felse>
  if_branch gen_else_if(ProteusValue cond) {
    assert(ElseBB && "gen_else* called twice in same gen_if");
    getBuilder(context)->SetInsertPoint(ElseBB);
    ElseBB = nullptr;
    return {cond, context, AfterBB};
  }

  template <typename Felse>
  if_branch gen_else_if(const expression_t &expr, const OperatorState &state) {
    assert(ElseBB && "gen_else* called twice in same gen_if");
    getBuilder(context)->SetInsertPoint(ElseBB);
    ElseBB = nullptr;
    return {expr, state, context, AfterBB};
  }

  ~if_then() {
    if (ElseBB) gen_else([]() {});
    // Just to be sure, reset the insertion pointer to AfterBB
    getBuilder(context)->SetInsertPoint(AfterBB);
  }
};

template <typename Fthen>
if_then if_branch::operator()(Fthen then) {
  return {condition, then, context};
}

#endif /* IF_STATEMENT_HPP_ */
