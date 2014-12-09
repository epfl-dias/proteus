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
#include "operators/monoids.hpp"
#include "util/raw-context.hpp"
#include "util/raw-catalog.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"


class ExprVisitor; //Forward declaration

//Careful: Using a namespace to avoid conflicts with LLVM namespace
namespace expressions
{
enum ExpressionId	{ CONSTANT, ARGUMENT, RECORD_PROJECTION, RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE };

class Expression	{
public:
	Expression(ExpressionType* type) : type(type)	{}
	virtual ~Expression()							{}

	ExpressionType* getExpressionType()				{ return type; }
	virtual RawValue accept(ExprVisitor &v) = 0;
	virtual ExpressionId getTypeID() = 0;
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
	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return CONSTANT; }
private:
	int val;
};

class BoolConstant : public Constant	{
public:
	BoolConstant(int val)
		: Constant(new BoolType()), val(val) 		{}
	~BoolConstant()									{}

	bool getVal()									{ return val; }
	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return CONSTANT; }
private:
	bool val;
};

class FloatConstant : public Constant	{
public:
	FloatConstant(int val) :
		Constant(new FloatType()), val(val) 		{}
	~FloatConstant()								{}

	double getVal()									{ return val; }

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return CONSTANT; }
private:
	double val;
};

class StringConstant : public Constant	{
public:
	StringConstant(string& val) :
		Constant(new StringType()), val(val) 		{}
	~StringConstant()								{}

	string& getVal()								{ return val; }
	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return CONSTANT; }
private:
	string& val;
};

/*
 * Conceptually:  What every next() call over a collection produces
 * In the general case, it is a record.
 * However it can be a primitive (i.e., if iterating over [1,2,3,4,5]
 * or even a random collection - (i.e., if iterating over [1,[5,9],[{"a":1},{"a":2}]]
 *
 * XXX How do we specify the schema of the last expression?
 */
class InputArgument	: public Expression	{
public:
//	InputArgument(ExpressionType* type, int argNo)
//		: Expression(type), argNo(argNo)								{}
	InputArgument(ExpressionType* type, int argNo,
			list<RecordAttribute> projections) :
			Expression(type), argNo(argNo), projections(projections)	{}

	~InputArgument()													{}

	int getArgNo()														{ return argNo; }
	list<RecordAttribute> getProjections()								{ return projections; }
	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()											{ return ARGUMENT; }
private:
	/**
	 * ArgumentNo is meant to represent e.g. the left or right child of a Join,
	 * NOT the projections that we need!
	 *
	 * argNo = 0 => lhs of Join
	 * argNo = 1 => rhs of Join, and so on.
	 */
	int argNo;
	list<RecordAttribute> projections;
};

class RecordProjection : public Expression	{
public:
	RecordProjection(ExpressionType* type, Expression* expr, const RecordAttribute& attribute)	:
			Expression(type), expr(expr), attribute(attribute)	{}
	~RecordProjection()								{}

	Expression* getExpr()							{ return expr; }
	string 	getOriginalRelationName() const			{ return attribute.getOriginalRelationName(); }
	string 	getRelationName() const			{ return attribute.getRelationName(); }
	string  getProjectionName()						{ return attribute.getAttrName(); }
	RawValue  accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return RECORD_PROJECTION; }
private:
	Expression* expr;
	const RecordAttribute& attribute;
};

class AttributeConstruction	{
public:
	AttributeConstruction(const string& name, Expression* expr) :
			name(name), expr(expr) 									{}
	~AttributeConstruction()										{}
private:
	const string& name;
	Expression* expr;
};

class RecordConstruction : public Expression	{
public:
	RecordConstruction(ExpressionType* type,
			const list<AttributeConstruction>& atts) :
			Expression(type), atts(atts) 							{}
	~RecordConstruction()											{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()										{ return RECORD_CONSTRUCTION; }
private:
	const list<AttributeConstruction>& atts;
};

class IfThenElse : public Expression	{
public:
	IfThenElse(ExpressionType* type, Expression* expr1, Expression* expr2, Expression* expr3) :
			Expression(type), expr1(expr1), expr2(expr2), expr3(expr3)			{}
	~IfThenElse()																{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()													{ return IF_THEN_ELSE; }
	Expression* getIfCond()														{ return expr1; }
	Expression* getIfResult()													{ return expr2; }
	Expression* getElseResult()													{ return expr3; }
private:
	Expression *expr1;
	Expression *expr2;
	Expression *expr3;
};

class BinaryExpression : public Expression	{
public:
	BinaryExpression(ExpressionType* type, expressions::BinaryOperator* op, Expression* lhs, Expression* rhs) :
		Expression(type), lhs(lhs), rhs(rhs), op(op)			{}
	Expression* getLeftOperand() 								{ return lhs; }
	Expression* getRightOperand()								{ return rhs; }
	expressions::BinaryOperator* getOp()						{ return op; }

	virtual RawValue accept(ExprVisitor &v) = 0;
	virtual ExpressionId getTypeID()							{ return BINARY; }
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

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class NeExpression : public BinaryExpression	{
public:
	NeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Neq(),lhs,rhs) 	{}
	~NeExpression()									{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class GeExpression : public BinaryExpression	{
public:
	GeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Ge(),lhs,rhs) 	{}
	~GeExpression()									{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class GtExpression : public BinaryExpression	{
public:
	GtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Gt(),lhs,rhs)		{}
	~GtExpression()									{}

	RawValue accept(ExprVisitor &v);
};

class LeExpression : public BinaryExpression	{
public:
	LeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Le(),lhs,rhs) 	{}
	~LeExpression()									{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class LtExpression : public BinaryExpression	{
public:
	LtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Lt(),lhs,rhs) 	{}
	~LtExpression()									{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class AddExpression : public BinaryExpression	{
public:
	AddExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Add(),lhs,rhs) 	{}
	~AddExpression()								{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class SubExpression : public BinaryExpression	{
public:
	SubExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Sub(),lhs,rhs) 	{}
	~SubExpression()								{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class MultExpression : public BinaryExpression	{
public:
	MultExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Mult(),lhs,rhs) 	{}
	~MultExpression()								{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class DivExpression : public BinaryExpression	{
public:
	DivExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Div(),lhs,rhs) 	{}
	~DivExpression()								{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class AndExpression : public BinaryExpression	{
public:
	AndExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new And(),lhs,rhs) 	{}
	~AndExpression()								{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

class OrExpression : public BinaryExpression	{
public:
	OrExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Or(),lhs,rhs) 	{}
	~OrExpression()									{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()						{ return BINARY; }
};

/**
 * XXX Relevant to
 * General monoid merging
 * User-provided collections (since [1,2,3] is sugar for [1] union [2] union [3]
 */
class MergeExpression : public Expression	{
public:
	MergeExpression(ExpressionType* type, Monoid acc, Expression* lhs, Expression* rhs) :
		Expression(type), monoidType(acc), lhs(lhs), rhs(rhs) 		{}
	~MergeExpression()												{}

	RawValue accept(ExprVisitor &v);
	ExpressionId getTypeID()										{ return MERGE; }
	Monoid getMonoidType()										{ return monoidType; }
private:
	Monoid monoidType;
	Expression* lhs;
	Expression* rhs;
};


}

//===----------------------------------------------------------------------===//
// "Visitor" responsible for generating the appropriate code per Expression 'node'
//===----------------------------------------------------------------------===//
class ExprVisitor
{
public:
	virtual RawValue visit(expressions::IntConstant *e)    		= 0;
	virtual RawValue visit(expressions::FloatConstant *e)  		= 0;
	virtual RawValue visit(expressions::BoolConstant *e)   		= 0;
	virtual RawValue visit(expressions::StringConstant *e) 		= 0;
	virtual RawValue visit(expressions::InputArgument *e)  		= 0;
	virtual RawValue visit(expressions::RecordProjection *e)	= 0;
	virtual RawValue visit(expressions::IfThenElse *e)  		= 0;
	virtual RawValue visit(expressions::EqExpression *e)   		= 0;
	virtual RawValue visit(expressions::NeExpression *e)   		= 0;
	virtual RawValue visit(expressions::GeExpression *e)   		= 0;
	virtual RawValue visit(expressions::GtExpression *e)   		= 0;
	virtual RawValue visit(expressions::LeExpression *e)   		= 0;
	virtual RawValue visit(expressions::LtExpression *e)   		= 0;
	virtual RawValue visit(expressions::AddExpression *e)  		= 0;
	virtual RawValue visit(expressions::SubExpression *e)  		= 0;
	virtual RawValue visit(expressions::MultExpression *e) 		= 0;
	virtual RawValue visit(expressions::DivExpression *e)  		= 0;
	virtual RawValue visit(expressions::AndExpression *e)  		= 0;
	virtual RawValue visit(expressions::OrExpression *e)  		= 0;
	virtual RawValue visit(expressions::MergeExpression *e)  	= 0;
	virtual ~ExprVisitor() {}
};

#endif /* EXPRESSIONS_HPP_ */
