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
	Constant(ExpressionType* type) : Expression(type) {};
};

class IntConstant : public Constant		{
public:
	IntConstant(int val)
		: Constant(new IntType()), val(val) 		{}

	int getVal()									{ return val; }
	Value* accept(ExprVisitor &v);
private:
	int val;
};

class BoolConstant : public Constant	{
public:
	BoolConstant(int val)
		: Constant(new BoolType()), val(val) 		{}

	bool getVal()									{ return val; }
	Value* accept(ExprVisitor &v);
private:
	bool val;
};

class FloatConstant : public Constant	{
public:
	FloatConstant(int val) :
		Constant(new FloatType()), val(val) 		{}

	double getVal()									{ return val; }

	Value* accept(ExprVisitor &v);
private:
	double val;
};

class StringConstant : public Constant	{
public:
	StringConstant(string& val) :
		Constant(new StringType()), val(val) 		{}

	string& getVal()								{ return val; }
	Value* accept(ExprVisitor &v);
private:
	string& val;
};

class InputArgument	: public Expression	{
public:
	InputArgument(ExpressionType* type,int argNo, string& name)
		: Expression(type), argNo(argNo), argName(name)			{}
	int getArgNo()												{ return argNo; }
	string& getArgName()										{ return argName; }

	Value* accept(ExprVisitor &v);
private:
	/* FIXME What is the convenient way to 'connect' the info about arguments from the plan?
	 * If I just need to probe the symbol table, the name is convenient.
	 * A Number has been useful so far because of all the 'anonymous' transformations
	 */
	//Is argNo used anywhere as is? It is quite confusing
	int argNo;
	string& argName;
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
	Value* accept(ExprVisitor &v);
};

class NeExpression : public BinaryExpression	{
public:
	NeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Neq(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class GeExpression : public BinaryExpression	{
public:
	GeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Ge(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class GtExpression : public BinaryExpression	{
public:
	GtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Gt(),lhs,rhs)		{}
	Value* accept(ExprVisitor &v);
};

class LeExpression : public BinaryExpression	{
public:
	LeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Le(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class LtExpression : public BinaryExpression	{
public:
	LtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Lt(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class AddExpression : public BinaryExpression	{
public:
	AddExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Add(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class SubExpression : public BinaryExpression	{
public:
	SubExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Sub(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class MultExpression : public BinaryExpression	{
public:
	MultExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Mult(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

class DivExpression : public BinaryExpression	{
public:
	DivExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Div(),lhs,rhs) 	{}
	Value* accept(ExprVisitor &v);
};

}

//===----------------------------------------------------------------------===//
// "Visitor" responsible for generating the appropriate code per Expression 'node'
//===----------------------------------------------------------------------===//
class ExprVisitor
{
public:
	virtual Value* visit(expressions::IntConstant *e)    = 0;
	virtual Value* visit(expressions::FloatConstant *e)  = 0;
	virtual Value* visit(expressions::BoolConstant *e)   = 0;
	virtual Value* visit(expressions::StringConstant *e) = 0;
	virtual Value* visit(expressions::InputArgument *e)  = 0;
	virtual Value* visit(expressions::EqExpression *e)   = 0;
	virtual Value* visit(expressions::NeExpression *e)   = 0;
	virtual Value* visit(expressions::GeExpression *e)   = 0;
	virtual Value* visit(expressions::GtExpression *e)   = 0;
	virtual Value* visit(expressions::LeExpression *e)   = 0;
	virtual Value* visit(expressions::LtExpression *e)   = 0;
	virtual Value* visit(expressions::AddExpression *e)  = 0;
	virtual Value* visit(expressions::SubExpression *e)  = 0;
	virtual Value* visit(expressions::MultExpression *e) = 0;
	virtual Value* visit(expressions::DivExpression *e)  = 0;
	virtual ~ExprVisitor() {}
};

class ExpressionGeneratorVisitor: public ExprVisitor
{
public:
	ExpressionGeneratorVisitor(RawContext* const context, const OperatorState& currState)
		: context(context) , currState(currState) 	{}
	Value* visit(expressions::IntConstant *e);
	Value* visit(expressions::FloatConstant *e);
	Value* visit(expressions::BoolConstant *e);
	Value* visit(expressions::StringConstant *e);
	//Needs access to symbol table
	Value* visit(expressions::InputArgument *e);
	Value* visit(expressions::EqExpression *e);
	Value* visit(expressions::NeExpression *e);
	Value* visit(expressions::GeExpression *e);
	Value* visit(expressions::GtExpression *e);
	Value* visit(expressions::LeExpression *e);
	Value* visit(expressions::LtExpression *e);
	Value* visit(expressions::AddExpression *e);
	Value* visit(expressions::SubExpression *e);
	Value* visit(expressions::MultExpression *e);
	Value* visit(expressions::DivExpression *e);

private:
	RawContext* const context;
	const OperatorState& currState;
};

#endif /* EXPRESSIONS_HPP_ */
