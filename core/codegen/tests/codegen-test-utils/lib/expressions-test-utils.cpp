/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "codegen/test/expressions-test-utils.hpp"

using namespace codegen;

void codegen::printInt(Context *context, llvm::Value *value) {
  llvm::Function *printInt = context->getFunction("printi");
  context->gen_call(printInt, {value});
}

void codegen::printChar(Context *context, llvm::Value *value) {
  // All these complications because printc function's argument it pointer to
  // the char array from which the first element will be printed
  llvm::Type *charType = llvm::Type::getInt8Ty(context->getLLVMContext());
  llvm::Value *charMem = context->CreateEntryBlockAlloca(
      "charPtr", charType, context->createInt32(1));
  llvm::Value *charArrayPtr =
      context->getArrayElemMem(charMem, context->createInt32(0));
  context->getBuilder()->CreateStore(value, charArrayPtr);
  llvm::Function *printChar = context->getFunction("printc");
  context->gen_call(printChar, {charArrayPtr});
}

ProteusValue TestExpressionVisitor::visit(const expressions::IntConstant *e) {
  ProteusValue valWrapper;
  valWrapper.value = llvm::ConstantInt::get(context->getLLVMContext(),
                                            llvm::APInt(32, e->getVal()));
  valWrapper.isNull = context->createFalse();

  printInt(valWrapper.value);

  return valWrapper;
}

ProteusValue TestExpressionVisitor::visit(const expressions::AddExpression *e) {
  return generateArithmeticVisit(
      e, '+', [this](llvm::Value *lhs, llvm::Value *rhs) {
        return context->getBuilder()->CreateAdd(lhs, rhs);
      });
}

ProteusValue TestExpressionVisitor::visit(const expressions::SubExpression *e) {
  return generateArithmeticVisit(
      e, '-', [this](llvm::Value *lhs, llvm::Value *rhs) {
        return context->getBuilder()->CreateSub(lhs, rhs);
      });
}

ProteusValue TestExpressionVisitor::visit(
    const expressions::MultExpression *e) {
  return generateArithmeticVisit(
      e, '*', [this](llvm::Value *lhs, llvm::Value *rhs) {
        return context->getBuilder()->CreateMul(lhs, rhs);
      });
}

ProteusValue TestExpressionVisitor::visit(const expressions::DivExpression *e) {
  return generateArithmeticVisit(
      e, '/', [this](llvm::Value *lhs, llvm::Value *rhs) {
        return context->getBuilder()->CreateSDiv(lhs, rhs);
      });
}

void TestExpressionVisitor::printInt(llvm::Value *value) {
  if (isPrintEnabled) {
    ::codegen::printInt(context, value);
  }
}

void TestExpressionVisitor::printChar(llvm::Value *value) {
  if (isPrintEnabled) {
    ::codegen::printChar(context, value);
  }
}

ProteusValue TestExpressionVisitor::visit(
    const expressions::ExternExpression *e) {
  ProteusValue valWrapper;

  // Check if the function was properly linked
  auto fun = context->getModule()->getFunction(e->getName());
  if (!fun) {
    LOG(ERROR) << "[ExpressionGeneratorVisitor]: extern function "
               << e->getName() << " not found! Did you link the module?";
    throw runtime_error(
        std::string("[ExpressionGeneratorVisitor]: extern function " +
                    e->getName() + " not found! Did you link the module?"));
  }

  context->registerFunction(e->getName(), fun);
  auto funType = fun->getFunctionType();

  // check if number of arguments matches
  if (funType->getNumParams() != e->getArgs().size()) {
    //  && "Mismatch in argument count for extern function"
    LOG(ERROR) << "[ExpressionGeneratorVisitor]: number of arguments for "
                  "extern function "
               << e->getName()
               << " doesn't match! Expected: " << funType->getNumParams()
               << " Have: " << e->getArgs().size();
    throw runtime_error(std::string(
        "[ExpressionGeneratorVisitor]: number of arguments for "
        "extern function " +
        e->getName() +
        " doesn't match! Expected: " + std::to_string(funType->getNumParams()) +
        " Have: " + std::to_string(e->getArgs().size())));
  }

  if (funType->getReturnType() !=
      e->getExpressionType()->getLLVMType(context->getLLVMContext())) {
    LOG(ERROR)
        << "[ExpressionGeneratorVisitor]: return type for extern function "
        << e->getName() << " doesn't match!";
    throw runtime_error(std::string(
        "[ExpressionGeneratorVisitor]: return type for extern function " +
        e->getName() + " doesn't match!"));
  }

  // Compute and convert arguments
  std::vector<llvm::Value *> convertedArgs;
  convertedArgs.reserve(e->getArgs().size());

  for (size_t idx = 0; const auto &arg_expr : e->getArgs()) {
    auto computed_value = arg_expr.accept(*this);

    const auto computedType = computed_value.value->getType();
    const auto expectedType = funType->getParamType(idx);
    if (computedType != expectedType) {
      LOG(ERROR)
          << "[ExpressionGeneratorVisitor]: argument type for extern function "
          << e->getName() << " with id " << std::to_string(idx)
          << " doesn't match!";
      throw runtime_error(std::string(
          "[ExpressionGeneratorVisitor]: argument type for extern function " +
          e->getName() + " with id " + std::to_string(idx) +
          " doesn't match!"));
    }

    convertedArgs.push_back(computed_value.value);
    idx++;
  }

  valWrapper.value = context->gen_call(fun, convertedArgs);
  valWrapper.isNull = fun->getReturnType()->isVoidTy() ? context->createTrue()
                                                       : context->createFalse();
  return valWrapper;
}

ProteusValue TestExpressionVisitor::generateArithmeticVisit(
    const expressions::BinaryExpression *e, char operationChar,
    const ArithmeticOperation &operation) {
  llvm::Value *lBracket = context->createInt8('(');
  printChar(lBracket);

  ProteusValue left = e->getLeftOperand().accept(*this);

  llvm::Value *opChar = context->createInt8(operationChar);
  printChar(opChar);

  ProteusValue right = e->getRightOperand().accept(*this);

  llvm::Value *rBracket = context->createInt8(')');
  printChar(rBracket);

  const ExpressionType *childType = e->getLeftOperand().getExpressionType();
  typeID id = childType->getTypeID();
  ProteusValue valWrapper;
  valWrapper.isNull = context->createFalse();

  switch (id) {
    case INT:
      valWrapper.value = operation(left.value, right.value);
      return valWrapper;
    default:
      LOG(ERROR) << "[ExpressionGeneratorVisitor]: Unknown Input";
      throw runtime_error(
          string("[ExpressionGeneratorVisitor]: Unknown Input"));
  }
}
