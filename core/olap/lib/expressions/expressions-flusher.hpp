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

#include <platform/common/common.hpp>

#include "expressions-generator.hpp"
#include "lib/plugins/json-plugin.hpp"
#include "olap/plugins/plugins.hpp"
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
        outputFileLLVM(nullptr),
        pg(nullptr),
        // Only used as a token return value that is passed along by each
        // visitor
        placeholder{nullptr, context->createFalse()} {
    pg = new jsonPipelined::JSONPlugin(context, outputFile, nullptr);
  }
  ExpressionFlusherVisitor(Context *const context,
                           const OperatorState &currState,
                           const char *outputFile, std::string activeRelation)
      : context(context),
        currState(currState),
        outputFile(outputFile),
        activeRelation(std::move(activeRelation)),
        outputFileLLVM(nullptr),
        // Only used as a token return value that is passed along by each
        // visitor
        placeholder{nullptr, context->createTrue()} {
    pg = Catalog::getInstance().getPlugin(this->activeRelation);
  }
  ProteusValue visit(const expressions::IntConstant *e) override;
  ProteusValue visit(const expressions::Int64Constant *e) override;
  ProteusValue visit(const expressions::DateConstant *e) override;
  ProteusValue visit(const expressions::FloatConstant *e) override;
  ProteusValue visit(const expressions::BoolConstant *e) override;
  ProteusValue visit(const expressions::StringConstant *e) override;
  ProteusValue visit(const expressions::DStringConstant *e) override;
  ProteusValue visit(const expressions::InputArgument *e) override;
  ProteusValue visit(const expressions::RecordProjection *e) override;
  ProteusValue visit(const expressions::IfThenElse *e) override;
  // XXX Do binary operators require explicit handling of nullptr?
  ProteusValue visit(const expressions::EqExpression *e) override;
  ProteusValue visit(const expressions::NeExpression *e) override;
  ProteusValue visit(const expressions::GeExpression *e) override;
  ProteusValue visit(const expressions::GtExpression *e) override;
  ProteusValue visit(const expressions::LeExpression *e) override;
  ProteusValue visit(const expressions::LtExpression *e) override;
  ProteusValue visit(const expressions::AddExpression *e) override;
  ProteusValue visit(const expressions::SubExpression *e) override;
  ProteusValue visit(const expressions::MultExpression *e) override;
  ProteusValue visit(const expressions::DivExpression *e) override;
  ProteusValue visit(const expressions::ModExpression *e) override;
  ProteusValue visit(const expressions::AndExpression *e) override;
  ProteusValue visit(const expressions::OrExpression *e) override;
  ProteusValue visit(const expressions::RecordConstruction *e) override;
  ProteusValue visit(const expressions::PlaceholderExpression *e) override;
  ProteusValue visit(const expressions::ProteusValueExpression *e) override;
  ProteusValue visit(const expressions::MinExpression *e) override;
  ProteusValue visit(const expressions::MaxExpression *e) override;
  ProteusValue visit(const expressions::RandExpression *e) override;
  ProteusValue visit(const expressions::HintExpression *e) override;
  ProteusValue visit(const expressions::HashExpression *e) override;
  ProteusValue visit(const expressions::RefExpression *e) override;
  ProteusValue visit(const expressions::AssignExpression *e) override;
  ProteusValue visit(const expressions::NegExpression *e) override;
  ProteusValue visit(const expressions::ExtractExpression *e) override;
  ProteusValue visit(const expressions::TestNullExpression *e) override;
  ProteusValue visit(const expressions::CastExpression *e) override;

  ProteusValue visit(const expressions::ShiftLeftExpression *e) override;
  ProteusValue visit(
      const expressions::LogicalShiftRightExpression *e) override;
  ProteusValue visit(
      const expressions::ArithmeticShiftRightExpression *e) override;
  ProteusValue visit(const expressions::XORExpression *e) override;
  /* Reduce produces accumulated value internally.
   * It makes no sense to probe a plugin in order to flush this value out */
  void flushValue(llvm::Value *val, typeID val_type);

  void setActiveRelation(const string &relName) {
    activeRelation = relName;
    if (!relName.empty()) {
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
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      outputFileLLVM = context->CreateGlobalString(this->outputFile);
      llvm::Function *flushFunc = context->getFunction("flushChar");
      vector<llvm::Value *> ArgsV;
      // Start 'array'
      ArgsV.push_back(context->createInt8('['));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushBeginList(context, outputFile);
    }
  }
  void beginBag() { pg->flushBeginBag(context, outputFile); }
  void beginSet() { pg->flushBeginSet(context, outputFile); }
  void endList() {
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      outputFileLLVM = context->CreateGlobalString(this->outputFile);
      llvm::Function *flushFunc = context->getFunction("flushChar");
      vector<llvm::Value *> ArgsV;
      // Start 'array'
      ArgsV.push_back(context->createInt8(']'));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushEndList(context, outputFile);
    }
  }
  void endBag() { pg->flushEndBag(context, outputFile); }
  void endSet() { pg->flushEndSet(context, outputFile); }
  void flushOutput(const ExpressionType *type = nullptr) {
    pg->flushOutput(context, outputFile, type);
  }
  void flushDelim(llvm::Value *resultCtr, int depth = 0) {
    if (!pg) {
      // TODO: remove. deprecated exectuion path
      outputFileLLVM = context->CreateGlobalString(this->outputFile);
      llvm::Function *flushFunc = context->getFunction("flushDelim");
      vector<llvm::Value *> ArgsV;
      ArgsV.push_back(resultCtr);
      // XXX JSON-specific -> Serializer business to differentiate
      ArgsV.push_back(context->createInt8(','));
      ArgsV.push_back(outputFileLLVM);
      context->getBuilder()->CreateCall(flushFunc, ArgsV);
    } else {
      pg->flushDelim(context, resultCtr, outputFile, depth);
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
      pg->flushDelim(context, outputFile, depth);
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
