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

#include "expressions/expressions.hpp"

namespace expressions {

BinaryExpression::~BinaryExpression() {}

ProteusValue IntConstant::accept(ExprVisitor &v) const { return v.visit(this); }

ProteusValue Int64Constant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue DateConstant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue FloatConstant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue BoolConstant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue StringConstant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue DStringConstant::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue InputArgument::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue RecordProjection::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue IfThenElse::accept(ExprVisitor &v) const { return v.visit(this); }

ProteusValue RecordConstruction::accept(ExprVisitor &v) const {
  return v.visit(this);
}

/**
 * The binary expressions
 */
ProteusValue EqExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue NeExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue GeExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue GtExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue LeExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue LtExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue AddExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue SubExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue MultExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue DivExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue AndExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue OrExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue ProteusValueExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue MaxExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue MinExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue HashExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue NegExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue ExtractExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue TestNullExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

ProteusValue CastExpression::accept(ExprVisitor &v) const {
  return v.visit(this);
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
ProteusValue IntConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == INT) {
      auto rInt = dynamic_cast<const IntConstant *>(expr);
      return v.visit(this, rInt);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
ProteusValue Int64Constant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == INT64) {
      auto rInt = dynamic_cast<const Int64Constant *>(expr);
      return v.visit(this, rInt);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
ProteusValue DateConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == DATE) {
      auto rInt = dynamic_cast<const DateConstant *>(expr);
      return v.visit(this, rInt);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue FloatConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == FLOAT) {
      auto rFloat = dynamic_cast<const FloatConstant *>(expr);
      return v.visit(this, rFloat);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue BoolConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == BOOL) {
      auto rBool = dynamic_cast<const BoolConstant *>(expr);
      return v.visit(this, rBool);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue StringConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == STRING) {
      auto rString = dynamic_cast<const StringConstant *>(expr);
      return v.visit(this, rString);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue DStringConstant::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rConst = dynamic_cast<const Constant *>(expr);
    if (rConst->getConstantType() == DSTRING) {
      auto rString = dynamic_cast<const DStringConstant *>(expr);
      return v.visit(this, rString);
    }
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue InputArgument::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rInputArg = dynamic_cast<const InputArgument *>(expr);
    return v.visit(this, rInputArg);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue RecordProjection::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rRecordProjection = dynamic_cast<const RecordProjection *>(expr);
    return v.visit(this, rRecordProjection);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue IfThenElse::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rIfThenElse = dynamic_cast<const IfThenElse *>(expr);
    return v.visit(this, rIfThenElse);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue RecordConstruction::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rRecordConstruction = dynamic_cast<const RecordConstruction *>(expr);
    return v.visit(this, rRecordConstruction);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue ProteusValueExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rVariable = dynamic_cast<const ProteusValueExpression *>(expr);
    return v.visit(this, rVariable);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

/**
 * The binary expressions
 */

ProteusValue EqExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rEqExpression = dynamic_cast<const EqExpression *>(expr);
    return v.visit(this, rEqExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue NeExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rNeExpression = dynamic_cast<const NeExpression *>(expr);
    return v.visit(this, rNeExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue GtExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rGtExpression = dynamic_cast<const GtExpression *>(expr);
    return v.visit(this, rGtExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue GeExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rGeExpression = dynamic_cast<const GeExpression *>(expr);
    return v.visit(this, rGeExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue LeExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rLeExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rLeExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue LtExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rLtExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rLtExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue AddExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rAddExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rAddExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue SubExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rSubExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rSubExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue MultExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rMultExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rMultExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue DivExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rDivExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rDivExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue AndExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rAndExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rAndExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue OrExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto rOrExpression = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, rOrExpression);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue MaxExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue MinExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue HashExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue NegExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue ExtractExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue TestNullExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

ProteusValue CastExpression::acceptTandem(
    ExprTandemVisitor &v, const expressions::Expression *expr) const {
  if (this->getTypeID() == expr->getTypeID()) {
    auto r = dynamic_cast<decltype(this)>(expr);
    return v.visit(this, r);
  }
  string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
  LOG(ERROR) << error_msg;
  throw runtime_error(string(error_msg));
}

}  // namespace expressions

expression_t toExpression(Monoid m, expression_t lhs, expression_t rhs) {
  switch (m) {
    case SUM:
      return lhs + rhs;
    case MULTIPLY:
      return lhs * rhs;
    case MAX:
      return expression_t::make<expressions::MaxExpression>(lhs, rhs);
    case OR:
      return lhs | rhs;
    case AND:
      return lhs & rhs;
    default:
      string error_msg = string("Unknown monoid");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
  }
}

llvm::Constant *getIdentityElementIfSimple(Monoid m, const ExpressionType *type,
                                           Context *context) {
  llvm::Type *llvmType = type->getLLVMType(context->getLLVMContext());
  typeID outputType = type->getTypeID();
  switch (m) {
    case SUM: {
      switch (outputType) {
        case DATE:
        case INT64:
        case INT: {
          return llvm::ConstantInt::get((llvm::IntegerType *)llvmType, 0);
        }
        case FLOAT: {
          return llvm::ConstantFP::get(llvmType, 0.0);
        }
        default: {
          string error_msg =
              string("[Monoid: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
    }
    case MULTIPLY: {
      switch (outputType) {
        case INT64:
        case INT: {
          return llvm::ConstantInt::get((llvm::IntegerType *)llvmType, 1);
        }
        case FLOAT: {
          return llvm::ConstantFP::get(llvmType, 1.0);
        }
        default: {
          string error_msg =
              string("[Monoid: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
    }
    case MAX: {
      switch (outputType) {
        case DATE:
        case INT64: {
          return llvm::ConstantInt::get((llvm::IntegerType *)llvmType,
                                        std::numeric_limits<int64_t>::min());
        }
        case INT: {
          return llvm::ConstantInt::get((llvm::IntegerType *)llvmType,
                                        std::numeric_limits<int32_t>::min());
        }
        case FLOAT: {
          return llvm::ConstantFP::getInfinity(llvmType, true);
        }
        default: {
          string error_msg =
              string("[Monoid: ] Sum/Multiply/Max operate on numerics");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
    }
    case OR: {
      switch (outputType) {
        case BOOL: {
          return llvm::ConstantInt::getFalse(context->getLLVMContext());
        }
        default: {
          string error_msg = string("[Monoid: ] Or/And operate on booleans");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
    }
    case AND: {
      switch (outputType) {
        case BOOL: {
          return llvm::ConstantInt::getTrue(context->getLLVMContext());
        }
        default: {
          string error_msg = string("[Monoid: ] Or/And operate on booleans");
          LOG(ERROR) << error_msg;
          throw runtime_error(error_msg);
        }
      }
    }
    default: { return NULL; }
  }
}

ExpressionType *expressions::ExtractExpression::createReturnType(
    extract_unit u) {
  switch (u) {
    case extract_unit::SECOND:
    case extract_unit::MINUTE:
    case extract_unit::HOUR:
    case extract_unit::DAYOFWEEK:
    case extract_unit::DAYOFMONTH:
    case extract_unit::DAYOFYEAR:
    case extract_unit::WEEK:
    case extract_unit::MONTH:
    case extract_unit::QUARTER:
    case extract_unit::YEAR:
    case extract_unit::MILLENNIUM:
    case extract_unit::CENTURY:
    case extract_unit::DECADE: {
      return new IntType();
    }
    default: {
      string error_msg =
          "[extract_unit: ] Unknown return type for extract unit";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
}
