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

#ifndef EXPRESSIONS_HPP_
#define EXPRESSIONS_HPP_

#include "operators/operators.hpp"
#include "operators/operator-state.hpp"
#include "util/raw-context.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"


class ExprVisitor; //Forward declaration

//Using a namespace to avoid conflicts with LLVM namespace
namespace expressions
{
class Expression	{
public:
	Expression(ExpressionType* type) : type(type)	{}
	virtual ~Expression()							{}

	ExpressionType* getExpressionType()				{ return type; }
	virtual Value* accept(ExprVisitor &v) = 0;
private:
	ExpressionType* type;
};

class Constant : public Expression		{
public:
	Constant(ExpressionType* type) : Expression(type)	{}
	~Constant()										  	{}
};

class IntConstant : public Constant		{
public:
	IntConstant(int val)
		: Constant(new IntType()), val(val) 		{}
	~IntConstant()									{}

	int getVal()									{ return val; }
	Value* accept(ExprVisitor &v);
private:
	int val;
};

class BoolConstant : public Constant	{
public:
	BoolConstant(int val)
		: Constant(new BoolType()), val(val) 		{}
	~BoolConstant()									{}

	bool getVal()									{ return val; }
	Value* accept(ExprVisitor &v);
private:
	bool val;
};

class FloatConstant : public Constant	{
public:
	FloatConstant(int val) :
		Constant(new FloatType()), val(val) 		{}
	~FloatConstant()								{}

	double getVal()									{ return val; }

	Value* accept(ExprVisitor &v);
private:
	double val;
};

class StringConstant : public Constant	{
public:
	StringConstant(string& val) :
		Constant(new StringType()), val(val) 		{}
	~StringConstant()								{}

	string& getVal()								{ return val; }
	Value* accept(ExprVisitor &v);
private:
	string& val;
};

class InputArgument	: public Expression	{
public:
	//FIXME remove name from constructor
	InputArgument(ExpressionType* type,int argNo)
		: Expression(type), argNo(argNo)						{}
	~InputArgument()											{}

	int getArgNo()												{ return argNo; }
	Value* accept(ExprVisitor &v);
private:
	/**
	 * ArgumentNo is meant to represent e.g. the left or right child of a Join,
	 * NOT the projections that we need!
	 *
	 * argNo = 0 => lhs of Join
	 * argNo = 1 => rhs of Join, and so on.
	 */
	int argNo;
};

class RecordProjection : public Expression	{
public:
	RecordProjection(ExpressionType* type, Expression* expr, const char* name)	:
		Expression(type), expr(expr), name(name)	{}
	~RecordProjection()								{}

	Expression* getExpr()							{ return expr; }
	const char*	getProjectionName()					{ return name; }
	Value* accept(ExprVisitor &v);
private:
	Expression* expr;
	const char* name;
};

class BinaryExpression : public Expression	{
public:
	BinaryExpression(ExpressionType* type, expressions::BinaryOperator* op, Expression* lhs, Expression* rhs) :
		Expression(type), lhs(lhs), rhs(rhs), op(op)			{}
	Expression* getLeftOperand() 								{ return lhs; }
	Expression* getRightOperand()								{ return rhs; }
	expressions::BinaryOperator* getOp()						{ return op; }

	virtual Value* accept(ExprVisitor &v) = 0;
	~BinaryExpression() = 0;
private:
	Expression* lhs;
	Expression* rhs;
	BinaryOperator* op;
};

class EqExpression : public BinaryExpression	{
public:
	EqExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Eq(),lhs,rhs) 	{}
	~EqExpression()									{}

	Value* accept(ExprVisitor &v);
};

class NeExpression : public BinaryExpression	{
public:
	NeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Neq(),lhs,rhs) 	{}
	~NeExpression()									{}

	Value* accept(ExprVisitor &v);
};

class GeExpression : public BinaryExpression	{
public:
	GeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Ge(),lhs,rhs) 	{}
	~GeExpression()									{}

	Value* accept(ExprVisitor &v);
};

class GtExpression : public BinaryExpression	{
public:
	GtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Gt(),lhs,rhs)		{}
	~GtExpression()									{}

	Value* accept(ExprVisitor &v);
};

class LeExpression : public BinaryExpression	{
public:
	LeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Le(),lhs,rhs) 	{}
	~LeExpression()									{}

	Value* accept(ExprVisitor &v);
};

class LtExpression : public BinaryExpression	{
public:
	LtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Lt(),lhs,rhs) 	{}
	~LtExpression()									{}

	Value* accept(ExprVisitor &v);
};

class AddExpression : public BinaryExpression	{
public:
	AddExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Add(),lhs,rhs) 	{}
	~AddExpression()								{}

	Value* accept(ExprVisitor &v);
};

class SubExpression : public BinaryExpression	{
public:
	SubExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Sub(),lhs,rhs) 	{}
	~SubExpression()								{}

	Value* accept(ExprVisitor &v);
};

class MultExpression : public BinaryExpression	{
public:
	MultExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Mult(),lhs,rhs) 	{}
	~MultExpression()								{}

	Value* accept(ExprVisitor &v);
};

class DivExpression : public BinaryExpression	{
public:
	DivExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Div(),lhs,rhs) 	{}
	~DivExpression()								{}

	Value* accept(ExprVisitor &v);
};

}

//===----------------------------------------------------------------------===//
// "Visitor" responsible for generating the appropriate code per Expression 'node'
//===----------------------------------------------------------------------===//
class ExprVisitor
{
public:
	virtual Value* visit(expressions::IntConstant *e)    	= 0;
	virtual Value* visit(expressions::FloatConstant *e)  	= 0;
	virtual Value* visit(expressions::BoolConstant *e)   	= 0;
	virtual Value* visit(expressions::StringConstant *e) 	= 0;
	virtual Value* visit(expressions::InputArgument *e)  	= 0;
	virtual Value* visit(expressions::RecordProjection *e)	= 0;
	virtual Value* visit(expressions::EqExpression *e)   	= 0;
	virtual Value* visit(expressions::NeExpression *e)   	= 0;
	virtual Value* visit(expressions::GeExpression *e)   	= 0;
	virtual Value* visit(expressions::GtExpression *e)   	= 0;
	virtual Value* visit(expressions::LeExpression *e)   	= 0;
	virtual Value* visit(expressions::LtExpression *e)   	= 0;
	virtual Value* visit(expressions::AddExpression *e)  	= 0;
	virtual Value* visit(expressions::SubExpression *e)  	= 0;
	virtual Value* visit(expressions::MultExpression *e) 	= 0;
	virtual Value* visit(expressions::DivExpression *e)  	= 0;
	virtual ~ExprVisitor() {}
};

#endif /* EXPRESSIONS_HPP_ */
