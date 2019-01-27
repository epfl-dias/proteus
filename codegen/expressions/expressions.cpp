/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2014
		Data Intensive Applications and Systems Labaratory (DIAS)
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

RawValue IntConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue Int64Constant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue DateConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue FloatConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue BoolConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue StringConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue DStringConstant::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue InputArgument::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue RecordProjection::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue IfThenElse::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue RecordConstruction::accept(ExprVisitor &v) const {
	return v.visit(this);
}

/**
 * The binary expressions
 */
RawValue EqExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue NeExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue GeExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue GtExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue LeExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue LtExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue AddExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue SubExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue MultExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue DivExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue AndExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue OrExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue RawValueExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue MaxExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue MinExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue HashExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue NegExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue ExtractExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue TestNullExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

RawValue CastExpression::accept(ExprVisitor &v) const {
	return v.visit(this);
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
RawValue IntConstant::acceptTandem(ExprTandemVisitor &v, const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == INT) {
			auto rInt = dynamic_cast<const IntConstant*>(expr);
			return v.visit(this, rInt);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
RawValue Int64Constant::acceptTandem(ExprTandemVisitor &v, const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == INT64) {
			auto rInt = dynamic_cast<const Int64Constant*>(expr);
			return v.visit(this, rInt);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

/*XXX My responsibility to provide appropriate (i.e., compatible) input */
RawValue DateConstant::acceptTandem(ExprTandemVisitor &v, const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == DATE) {
			auto rInt = dynamic_cast<const DateConstant*>(expr);
			return v.visit(this, rInt);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue FloatConstant::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == FLOAT) {
			auto rFloat = dynamic_cast<const FloatConstant*>(expr);
			return v.visit(this, rFloat);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue BoolConstant::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == BOOL) {
			auto rBool = dynamic_cast<const BoolConstant*>(expr);
			return v.visit(this, rBool);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue StringConstant::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == STRING) {
			auto rString = dynamic_cast<const StringConstant*>(expr);
			return v.visit(this, rString);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue DStringConstant::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rConst = dynamic_cast<const Constant*>(expr);
		if (rConst->getConstantType() == DSTRING) {
			auto rString = dynamic_cast<const DStringConstant*>(expr);
			return v.visit(this, rString);
		}
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue InputArgument::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rInputArg = dynamic_cast<const InputArgument*>(expr);
		return v.visit(this, rInputArg);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue RecordProjection::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rRecordProjection =
				dynamic_cast<const RecordProjection*>(expr);
		return v.visit(this, rRecordProjection);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue IfThenElse::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rIfThenElse = dynamic_cast<const IfThenElse*>(expr);
		return v.visit(this, rIfThenElse);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue RecordConstruction::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rRecordConstruction =
				dynamic_cast<const RecordConstruction*>(expr);
		return v.visit(this, rRecordConstruction);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue RawValueExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rVariable = dynamic_cast<const RawValueExpression*>(expr);
		return v.visit(this, rVariable);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

/**
 * The binary expressions
 */

RawValue EqExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rEqExpression =
				dynamic_cast<const EqExpression*>(expr);
		return v.visit(this, rEqExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue NeExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rNeExpression =
				dynamic_cast<const NeExpression*>(expr);
		return v.visit(this, rNeExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue GtExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rGtExpression =
				dynamic_cast<const GtExpression*>(expr);
		return v.visit(this, rGtExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue GeExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rGeExpression =
				dynamic_cast<const GeExpression*>(expr);
		return v.visit(this, rGeExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue LeExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rLeExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rLeExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue LtExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rLtExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rLtExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue AddExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rAddExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rAddExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue SubExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rSubExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rSubExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue MultExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rMultExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rMultExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue DivExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rDivExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rDivExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue AndExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rAndExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rAndExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue OrExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto rOrExpression = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, rOrExpression);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue MaxExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue MinExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue HashExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue NegExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue ExtractExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue TestNullExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

RawValue CastExpression::acceptTandem(ExprTandemVisitor &v,
		const expressions::Expression* expr) const {
	if (this->getTypeID() == expr->getTypeID()) {
		auto r = dynamic_cast<decltype(this)>(expr);
		return v.visit(this, r);
	}
	string error_msg = string("[Tandem Visitor: ] Incompatible Pair");
	LOG(ERROR)<< error_msg;
	throw runtime_error(string(error_msg));
}

}

expression_t toExpression(Monoid m, expression_t lhs, expression_t rhs){
	switch (m){
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
			LOG(ERROR)<< error_msg;
			throw runtime_error(string(error_msg));
	}
}

llvm::Constant * getIdentityElementIfSimple(Monoid m, const ExpressionType * type, RawContext * context){
	Type * llvmType   = type->getLLVMType(context->getLLVMContext());
	typeID outputType = type->getTypeID();
	switch (m) {
		case SUM: {
			switch (outputType) {
				case DATE:
				case INT64:
				case INT: {
					return ConstantInt::get((IntegerType *) llvmType, 0);
				}
				case FLOAT: {
					return ConstantFP::get(llvmType, 0.0);
				}
				default: {
					string error_msg = string(
							"[Monoid: ] Sum/Multiply/Max operate on numerics");
					LOG(ERROR)<< error_msg;
					throw runtime_error(error_msg);
				}
			}
		}
		case MULTIPLY: {
			switch (outputType) {
				case INT64:
				case INT: {
					return ConstantInt::get((IntegerType *) llvmType, 1);
				}
				case FLOAT: {
					return ConstantFP::get(llvmType, 1.0);
				}
				default: {
					string error_msg = string(
							"[Monoid: ] Sum/Multiply/Max operate on numerics");
					LOG(ERROR)<< error_msg;
					throw runtime_error(error_msg);
				}
			}
		}
		case MAX: {
			switch (outputType) {
				case DATE:
				case INT64:{
					return ConstantInt::get((IntegerType *) llvmType, 
										std::numeric_limits<int64_t>::min());
				}
				case INT: {
					return ConstantInt::get((IntegerType *) llvmType, 
										std::numeric_limits<int32_t>::min());
				}
				case FLOAT: {
					return ConstantFP::getInfinity(llvmType, true);
				}
				default: {
					string error_msg = string(
							"[Monoid: ] Sum/Multiply/Max operate on numerics");
					LOG(ERROR)<< error_msg;
					throw runtime_error(error_msg);
				}
			}
		}
		case OR: {
			switch (outputType) {
				case BOOL: {
					return ConstantInt::getFalse(context->getLLVMContext());
				}
				default: {
					string error_msg = string(
							"[Monoid: ] Or/And operate on booleans");
					LOG(ERROR)<< error_msg;
					throw runtime_error(error_msg);
				}
			}
		}
		case AND: {
			switch (outputType) {
				case BOOL: {
					return ConstantInt::getTrue(context->getLLVMContext());
				}
				default: {
					string error_msg = string(
							"[Monoid: ] Or/And operate on booleans");
					LOG(ERROR)<< error_msg;
					throw runtime_error(error_msg);
				}
			}
		}
		default: {
			return NULL;
		}
	}
}

ExpressionType * expressions::ExtractExpression::createReturnType(extract_unit u){
	switch (u){
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
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
	}
}
