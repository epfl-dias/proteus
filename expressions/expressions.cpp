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

RawValue IntConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue FloatConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue BoolConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue StringConstant::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue InputArgument::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue RecordProjection::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue IfThenElse::accept(ExprVisitor &v) {
	return v.visit(this);
}

/**
 * The binary expressions
 */
RawValue EqExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue NeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue GeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue GtExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue LeExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue LtExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue AddExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue SubExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue MultExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue AndExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

RawValue OrExpression::accept(ExprVisitor &v) {
	return v.visit(this);
}

}
