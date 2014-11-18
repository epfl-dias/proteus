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

Value* IntConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* FloatConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* BoolConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* StringConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* InputArgument::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* RecordProjection::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* IfThenElse::accept(ExprVisitor &v) {
	return v.visit(this);
}

/**
 * The binary expressions
 */
Value* EqExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* NeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* GeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* GtExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* LeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* LtExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* AddExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* SubExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* MultExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* AndExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

Value* OrExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

}
