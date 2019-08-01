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
    default: {
      return nullptr;
    }
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
