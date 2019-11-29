/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef EXPRESSIONS_FLUSHER_VISITOR_HPP_
#define EXPRESSIONS_FLUSHER_VISITOR_HPP_

#include "common/common.hpp"
#include "expressions/expressions-generator.hpp"
#include "plugins/json-plugin.hpp"
#include "plugins/plugins.hpp"
#ifdef DEBUG
#define DEBUG_FLUSH
#endif

//===---------------------------------------------------------------------------===//
// "Visitor(s)" responsible for eagerly evaluating and flushing an Expression
//===---------------------------------------------------------------------------===//
/**
 * TODO Also need the concept of a serializer / output plugin.
 * Atm, this flusher is 'hardcoded' to JSON logic
 */
class ExpressionFlusherVisitor : public ExprVisitor {
 public:
  // TODO: should we remove this constructor ?
  ExpressionFlusherVisitor(Context *const context,
                           const OperatorState &currState,
                           const char *outputFile)
      : context(context),
        currState(currState),
        outputFile(outputFile),
        activeRelation(""),
        pg(nullptr) {
    // Only used as a token return value that is passed along by each visitor
    placeholder.isNull = context->createTrue();
    placeholder.value = nullptr;
    outputFileLLVM = nullptr;
    pg = new jsonPipelined::JSONPlugin(context, outputFile, nullptr);
  }
  ExpressionFlusherVisitor(Context *const context,
                           const OperatorState &currState,
                           const char *outputFile, string activeRelation)
      : context(context),
        currState(currState),
        outputFile(outputFile),
        activeRelation(activeRelation) {
    placeholder.isNull = context->createTrue();
    placeholder.value = nullptr;
    outputFileLLVM = nullptr;
    pg = Catalog::getInstance().getPlugin(activeRelation);
  }
  ProteusValue visit(const expressions::IntConstant *e);
  ProteusValue visit(const expressions::Int64Constant *e);
  ProteusValue visit(const expressions::DateConstant *e);
  ProteusValue visit(const expressions::FloatConstant *e);
  ProteusValue visit(const expressions::BoolConstant *e);
  ProteusValue visit(const expressions::StringConstant *e);
  ProteusValue visit(const expressions::DStringConstant *e);
  ProteusValue visit(const expressions::InputArgument *e);
  ProteusValue visit(const expressions::RecordProjection *e);
  ProteusValue visit(const expressions::IfThenElse *e);
  // XXX Do binary operators require explicit handling of nullptr?
  ProteusValue visit(const expressions::EqExpression *e);
  ProteusValue visit(const expressions::NeExpression *e);
  ProteusValue visit(const expressions::GeExpression *e);
  ProteusValue visit(const expressions::GtExpression *e);
  ProteusValue visit(const expressions::LeExpression *e);
  ProteusValue visit(const expressions::LtExpression *e);
  ProteusValue visit(const expressions::AddExpression *e);
  ProteusValue visit(const expressions::SubExpression *e);
  ProteusValue visit(const expressions::MultExpression *e);
  ProteusValue visit(const expressions::DivExpression *e);
  ProteusValue visit(const expressions::ModExpression *e);
  ProteusValue visit(const expressions::AndExpression *e);
  ProteusValue visit(const expressions::OrExpression *e);
  ProteusValue visit(const expressions::RecordConstruction *e);
  ProteusValue visit(const expressions::ProteusValueExpression *e);
  ProteusValue visit(const expressions::MinExpression *e);
  ProteusValue visit(const expressions::MaxExpression *e);
  ProteusValue visit(const expressions::HashExpression *e);
  ProteusValue visit(const expressions::NegExpression *e);
  ProteusValue visit(const expressions::ExtractExpression *e);
  ProteusValue visit(const expressions::TestNullExpression *e);
  ProteusValue visit(const expressions::CastExpression *e);
  /* Reduce produces accumulated value internally.
   * It makes no sense to probe a plugin in order to flush this value out */
  void flushValue(llvm::Value *val, typeID val_type);

  void setActiveRelation(string relName) {
    activeRelation = relName;
    if (relName != "") {
      pg = Catalog::getInstance().getPlugin(activeRelation);
    } else {
      pg = Catalog::getInstance().getPlugin(outputFile);
    }
  }
  string getActiveRelation(string relName) { return activeRelation; }

  /**
   * TODO Push these functions to Serializer
   * NOTE: Hard-coded to JSON case atm
   */
  void beginList() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      llvm::Function *flushFunc = context->getFunction("flushChar");
      vector<llvm::Value *> ArgsV;
      // Start 'array'
      ArgsV.push_back(context->createInt8('['));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushBeginList(outputFileLLVM);
    }
  }
  void beginBag() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    pg->flushBeginBag(outputFileLLVM);
  }
  void beginSet() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    pg->flushBeginSet(outputFileLLVM);
  }
  void endList() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      llvm::Function *flushFunc = context->getFunction("flushChar");
      vector<llvm::Value *> ArgsV;
      // Start 'array'
      ArgsV.push_back(context->createInt8(']'));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushEndList(outputFileLLVM);
    }
  }
  void endBag() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    pg->flushEndBag(outputFileLLVM);
  }
  void endSet() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    pg->flushEndSet(outputFileLLVM);
  }
  void flushOutput() {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    llvm::Function *flushFunc = context->getFunction("flushOutput");
    vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(outputFileLLVM);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }
  void flushDelim(llvm::Value *resultCtr, int depth = 0) {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      llvm::Function *flushFunc = context->getFunction("flushDelim");
      vector<llvm::Value *> ArgsV;
      ArgsV.push_back(resultCtr);
      // XXX JSON-specific -> Serializer business to differentiate
      ArgsV.push_back(context->createInt8(','));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushDelim(resultCtr, outputFileLLVM, depth);
    }
  }
  void flushDelim(int depth = 0) {
    outputFileLLVM = context->CreateGlobalString(this->outputFile);
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      llvm::Function *flushFunc = context->getFunction("flushChar");
      vector<llvm::Value *> ArgsV;
      // XXX JSON-specific -> Serializer business to differentiate
      ArgsV.push_back(context->createInt8(','));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushDelim(outputFileLLVM, depth);
    }
  }

 private:
  Context *const context;
  const OperatorState &currState;
  const char *outputFile;
  llvm::Value *outputFileLLVM;

  ProteusValue placeholder;
  string activeRelation;
  Plugin *pg;
};

#endif /* EXPRESSIONS_FLUSHER_VISITOR_HPP_ */
