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

#include "expressions/expressions-dot-evaluator.hpp"

#include <expressions/expressions/ref-expression.hpp>

#include "expressions/expressions-generator.hpp"
#include "util/context.hpp"

using namespace llvm;

ProteusValue ExpressionDotVisitor::visit(const expressions::IntConstant *e1,
                                         const expressions::IntConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_int1 =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e1->getVal()));
  Value *val_int2 =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e2->getVal()));

  Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::DStringConstant *e1,
    const expressions::DStringConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_int1 =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e1->getVal()));
  Value *val_int2 =
      ConstantInt::get(context->getLLVMContext(), APInt(32, e2->getVal()));

  Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::Int64Constant *e1,
                                         const expressions::Int64Constant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_int1 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e1->getVal()));
  Value *val_int2 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e2->getVal()));

  Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::DateConstant *e1,
                                         const expressions::DateConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_int1 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e1->getVal()));
  Value *val_int2 =
      ConstantInt::get(context->getLLVMContext(), APInt(64, e2->getVal()));

  Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::FloatConstant *e1,
                                         const expressions::FloatConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_double1 =
      ConstantFP::get(context->getLLVMContext(), APFloat(e1->getVal()));
  Value *val_double2 =
      ConstantFP::get(context->getLLVMContext(), APFloat(e2->getVal()));

  Value *val_result = Builder->CreateFCmpOEQ(val_double1, val_double2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::BoolConstant *e1,
                                         const expressions::BoolConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_int1 =
      ConstantInt::get(context->getLLVMContext(), APInt(8, e1->getVal()));
  Value *val_int2 =
      ConstantInt::get(context->getLLVMContext(), APInt(8, e2->getVal()));

  Value *val_result = Builder->CreateICmpEQ(val_int1, val_int2);
  ProteusValue valWrapper;
  valWrapper.value = val_result;
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::StringConstant *e1,
    const expressions::StringConstant *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  vector<Value *> ArgsV;
  ArgsV.push_back(left.value);
  ArgsV.push_back(right.value);
  Function *stringEquality = context->getFunction("equalStringObjs");
  valWrapper.value =
      Builder->CreateCall(stringEquality, ArgsV, "equalStringObjsCall");

  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::EqExpression *e1,
                                         const expressions::EqExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::NeExpression *e1,
                                         const expressions::NeExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::GeExpression *e1,
                                         const expressions::GeExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::GtExpression *e1,
                                         const expressions::GtExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::LeExpression *e1,
                                         const expressions::LeExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::LtExpression *e1,
                                         const expressions::LtExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::AndExpression *e1,
                                         const expressions::AndExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::OrExpression *e1,
                                         const expressions::OrExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::AddExpression *e1,
                                         const expressions::AddExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::SubExpression *e1,
                                         const expressions::SubExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::MultExpression *e1,
    const expressions::MultExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::DivExpression *e1,
                                         const expressions::DivExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::ModExpression *e1,
                                         const expressions::ModExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::compareThroughEvaluation(
    const expressions::Expression *e1, const expressions::Expression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  ExpressionGeneratorVisitor exprGenerator1(context, currStateLeft);
  ProteusValue left = e1->accept(exprGenerator1);

  ExpressionGeneratorVisitor exprGenerator2(context, currStateRight);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper{nullptr, context->createFalse()};

  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::ShiftLeftExpression *e1,
    const expressions::ShiftLeftExpression *e2) {
  return compareThroughEvaluation(e1, e2);
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::LogicalShiftRightExpression *e1,
    const expressions::LogicalShiftRightExpression *e2) {
  return compareThroughEvaluation(e1, e2);
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::ArithmeticShiftRightExpression *e1,
    const expressions::ArithmeticShiftRightExpression *e2) {
  return compareThroughEvaluation(e1, e2);
}

ProteusValue ExpressionDotVisitor::visit(const expressions::XORExpression *e1,
                                         const expressions::XORExpression *e2) {
  return compareThroughEvaluation(e1, e2);
}

// XXX Careful here
ProteusValue ExpressionDotVisitor::visit(
    const expressions::RecordConstruction *e1,
    const expressions::RecordConstruction *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  Value *val_true = context->createTrue();
  Value *val_result = val_true;

  const list<expressions::AttributeConstruction> &atts1 = e1->getAtts();
  const list<expressions::AttributeConstruction> &atts2 = e2->getAtts();
  list<expressions::AttributeConstruction>::const_iterator it1 = atts1.begin();
  list<expressions::AttributeConstruction>::const_iterator it2 = atts2.begin();

  for (; it1 != atts1.end() && it2 != atts2.end(); it1++, it2++) {
    auto expr1 = it1->getExpression();
    auto expr2 = it2->getExpression();

    ProteusValue val_partialWrapper = expr1.acceptTandem(*this, expr2);
    val_result = Builder->CreateAnd(val_result, val_partialWrapper.value);
  }
  ProteusValue val_resultWrapper;
  val_resultWrapper.value = val_result;
  val_resultWrapper.isNull = context->createFalse();
  return val_resultWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::HashExpression *e1,
    const expressions::HashExpression *e2) {
  assert(false && "This does not really make sense");
  return e1->getExpr().acceptTandem(*this, e2->getExpr());
}

ProteusValue ExpressionDotVisitor::visit(const expressions::InputArgument *e1,
                                         const expressions::InputArgument *e2) {
  /* What would be needed is a per-pg 'dotCmp' method
   * -> compare piece by piece at this level, don't reconstruct here*/
  string error_msg =
      string("[Expression Dot: ] No explicit InputArg support yet");
  LOG(ERROR) << error_msg;
  throw runtime_error(error_msg);
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::ProteusValueExpression *e1,
    const expressions::ProteusValueExpression *e2) {
  // left or right should not matter for ProteusValueExpressions, as
  // the bindings will not be used
  ExpressionGeneratorVisitor visitor{context, currStateLeft};
  return eq(*e1, *e2).accept(visitor);
}

/* Probably insufficient for complex datatypes */
ProteusValue ExpressionDotVisitor::visit(
    const expressions::RecordProjection *e1,
    const expressions::RecordProjection *e2) {
  IRBuilder<> *const Builder = context->getBuilder();

  typeID id = e1->getExpressionType()->getTypeID();
  bool primitive =
      id == INT || id == FLOAT || id == BOOL || id == INT64 || id == STRING;
  if (primitive) {
    const OperatorState &currState1 = currStateLeft;
    ExpressionGeneratorVisitor exprGenerator1 =
        ExpressionGeneratorVisitor(context, currState1);
    ProteusValue left = e1->accept(exprGenerator1);

    const OperatorState &currState2 = currStateRight;
    ExpressionGeneratorVisitor exprGenerator2 =
        ExpressionGeneratorVisitor(context, currState2);
    ProteusValue right = e2->accept(exprGenerator2);

    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();
    switch (id) {
      case INT:
        //#ifdef DEBUG
        //        {
        //            /* Printing the pos. to be marked */
        //            if(e1->getProjectionName() == "age") {
        //                cout << "AGE! " << endl;
        //                Function* debugInt = context->getFunction("printi");
        //                vector<Value*> ArgsV;
        //                ArgsV.clear();
        //                ArgsV.push_back(left.value);
        //                Builder->CreateCall(debugInt, ArgsV);
        //            }
        //            else
        //            {
        //                cout << "Other projection - " <<
        //                e1->getProjectionName()
        //                << endl;
        //            }
        //        }
        //#endif
        valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
        return valWrapper;
      case FLOAT:
        valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
        return valWrapper;
      case BOOL:
        valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
        return valWrapper;
      case STRING: {
        vector<Value *> ArgsV;
        ArgsV.push_back(left.value);
        ArgsV.push_back(right.value);
        Function *stringEquality = context->getFunction("equalStringObjs");
        valWrapper.value =
            Builder->CreateCall(stringEquality, ArgsV, "equalStringObjsCall");
        return valWrapper;
      }
      default:
        LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
        throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
    }
    return valWrapper;
  } else {
    /* XXX
     * Stick to returning hash of result for now
     * Obviously can cause false positives
     *
     * What would be needed is a per-pg 'dotCmp' method
     * -> compare piece by piece at this level, don't reconstruct here
     */
    const OperatorState &currState1 = currStateLeft;
    ExpressionHasherVisitor aggrExprGenerator1 =
        ExpressionHasherVisitor(context, currState1);
    ProteusValue hashLeft = e1->accept(aggrExprGenerator1);

    const OperatorState &currState2 = currStateRight;
    ExpressionHasherVisitor aggrExprGenerator2 =
        ExpressionHasherVisitor(context, currState2);
    ProteusValue hashRight = e2->accept(aggrExprGenerator2);
    ProteusValue valWrapper;
    valWrapper.isNull = context->createFalse();
    valWrapper.value = Builder->CreateICmpEQ(hashLeft.value, hashRight.value);
    return valWrapper;
  }
}

ProteusValue ExpressionDotVisitor::visit(const expressions::IfThenElse *e1,
                                         const expressions::IfThenElse *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();
  switch (id) {
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case STRING: {
      vector<Value *> ArgsV;
      ArgsV.push_back(left.value);
      ArgsV.push_back(right.value);
      Function *stringEquality = context->getFunction("equalStringObjs");
      valWrapper.value =
          Builder->CreateCall(stringEquality, ArgsV, "equalStringObjsCall");
      return valWrapper;
    }
    default: {
      string error_msg =
          string("[Expression Dot: ] No explicit non-primitive support yet");
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(const expressions::MaxExpression *e1,
                                         const expressions::MaxExpression *e2) {
  return e1->getCond()->acceptTandem(*this, e2->getCond());
}

ProteusValue ExpressionDotVisitor::visit(const expressions::MinExpression *e1,
                                         const expressions::MinExpression *e2) {
  return e1->getCond()->acceptTandem(*this, e2->getCond());
}

ProteusValue ExpressionDotVisitor::visit(const expressions::NegExpression *e1,
                                         const expressions::NegExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1 =
      ExpressionGeneratorVisitor(context, currState1);
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2 =
      ExpressionGeneratorVisitor(context, currState2);
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case DSTRING:
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::ExtractExpression *e1,
    const expressions::ExtractExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1{context, currState1};
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2{context, currState2};
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case DSTRING:
    case INT:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    case FLOAT:
      valWrapper.value = Builder->CreateFCmpOEQ(left.value, right.value);
      return valWrapper;
    case BOOL:
      valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionDotVisitor]: Invalid Input";
      throw runtime_error(string("[ExpressionDotVisitor]: Invalid Input"));
  }

  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::TestNullExpression *e1,
    const expressions::TestNullExpression *e2) {
  IRBuilder<> *const Builder = context->getBuilder();
  const OperatorState &currState1 = currStateLeft;
  ExpressionGeneratorVisitor exprGenerator1{context, currState1};
  ProteusValue left = e1->accept(exprGenerator1);

  const OperatorState &currState2 = currStateRight;
  ExpressionGeneratorVisitor exprGenerator2{context, currState2};
  ProteusValue right = e2->accept(exprGenerator2);

  typeID id = e1->getExpressionType()->getTypeID();
  ProteusValue valWrapper;
  valWrapper.value = Builder->CreateICmpEQ(left.value, right.value);
  valWrapper.isNull = context->createFalse();
  return valWrapper;
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::CastExpression *e1,
    const expressions::CastExpression *e2) {
  return e1->getExpr().acceptTandem(*this, e2->getExpr());
}

ProteusValue ExpressionDotVisitor::visit(const expressions::RefExpression *e1,
                                         const expressions::RefExpression *e2) {
  return e1->getExpr().acceptTandem(*this, e2->getExpr());
}

ProteusValue ExpressionDotVisitor::visit(
    const expressions::AssignExpression *,
    const expressions::AssignExpression *) {
  throw std::runtime_error(
      "[ExpressionDotVisitor]: unsupported dot evaluation of assignment "
      "operation");
}
