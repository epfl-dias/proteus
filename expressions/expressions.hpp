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

#include "util/raw-context.hpp"
#include "values/expressionTypes.hpp"
#include "expressions/binary-operators.hpp"


class ExprVisitor; 		 //Forward declaration
class ExprTandemVisitor; //Forward declaration

//Careful: Using a namespace to avoid conflicts with LLVM namespace
namespace expressions
{


enum ExpressionId	{ CONSTANT, ARGUMENT, RECORD_PROJECTION, RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE };

class Expression	{
public:
	Expression(ExpressionType* type) : type(type)		{}
	Expression(const ExpressionType* type) : type(type)	{}
	virtual ~Expression()								{}

	const ExpressionType* getExpressionType()	const		{ return type; }
	virtual RawValue accept(ExprVisitor &v) = 0;
	virtual RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*) = 0;
	virtual ExpressionId getTypeID() const = 0;


	virtual inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			string error_msg = string(
					"[This operator is NOT responsible for this case!]");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	const ExpressionType* type;
};

struct less_map: std::binary_function<const Expression *,
		const Expression *, bool> {
	bool operator()(const Expression *a,
			const Expression *b) const {
		return *a < *b;
	}
};

class Constant : public Expression		{
public:
	enum ConstantType {
		INT, BOOL, FLOAT, STRING
	};
	Constant(ExpressionType* type) : Expression(type)	{}
	~Constant()										  	{}
	virtual ConstantType getConstantType() const = 0;
};

class IntConstant : public Constant		{
public:
	IntConstant(int val)
		: Constant(new IntType()), val(val) 		{}
	~IntConstant()									{}

	int getVal() const								{ return val; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return CONSTANT; }
	ConstantType getConstantType() const			{ return INT; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const Constant& rConst = dynamic_cast<const Constant&>(r);
			if (rConst.getConstantType() == INT) {
				const IntConstant& rInt = dynamic_cast<const IntConstant&>(r);
				cout << "1. Compatible (int)! " << rConst.getConstantType() << endl;
				cout << this->getVal() << " vs " << rInt.getVal() << endl;
				return this->getVal() < rInt.getVal();
			}
			else {
				return this->getConstantType() < rConst.getConstantType();
			}

		}
		cout << "Not compatible (int) " << this->getTypeID() << endl;
		return this->getTypeID() < r.getTypeID();

	}
private:
	int val;
};

class BoolConstant : public Constant	{
public:
	BoolConstant(int val)
		: Constant(new BoolType()), val(val) 		{}
	~BoolConstant()									{}

	bool getVal() const								{ return val; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return CONSTANT; }
	ConstantType getConstantType() const			{ return BOOL; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const Constant& rConst = dynamic_cast<const Constant&>(r);
			if (rConst.getConstantType() == BOOL) {
				const BoolConstant& rBool = dynamic_cast<const BoolConstant&>(r);
				return this->getVal() < rBool.getVal();
			}
			else
			{
				return this->getConstantType() < rConst.getConstantType();
			}
		}
		cout << "Not compatible (bool)" << endl;
		return this->getTypeID() < r.getTypeID();
	}
private:
	bool val;
};

class FloatConstant : public Constant	{
public:
	FloatConstant(double val) :
		Constant(new FloatType()), val(val) 		{}
	~FloatConstant()								{}

	double getVal()	const							{ return val; }

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return CONSTANT; }
	ConstantType getConstantType() const			{ return FLOAT; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const Constant& rConst = dynamic_cast<const Constant&>(r);
			if (rConst.getConstantType() == FLOAT) {
				const FloatConstant& rFloat =
						dynamic_cast<const FloatConstant&>(r);
				cout << "1. Not compatible (float) " << rConst.getConstantType() << endl;
				return this->getVal() < rFloat.getVal();
			}
			else
			{
				return this->getConstantType() < rConst.getConstantType();
			}
		}
		cout << "Not compatible (float) " << r.getTypeID() << endl;
		return this->getTypeID() < r.getTypeID();
	}
private:
	double val;
};

class StringConstant : public Constant	{
public:
	StringConstant(string& val) :
		Constant(new StringType()), val(val) 		{}
	~StringConstant()								{}

	string& getVal() const							{ return val; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return CONSTANT; }
	ConstantType getConstantType() const			{ return STRING; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const Constant& rConst = dynamic_cast<const Constant&>(r);
			if (rConst.getConstantType() == STRING) {
				const StringConstant& rString =
						dynamic_cast<const StringConstant&>(r);
				return this->getVal() < rString.getVal();
			}
			else {
				return this->getConstantType() < rConst.getConstantType();
			}
		}
		cout << "Not compatible (string)" << endl;
		return this->getTypeID() < r.getTypeID();
	}
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
	InputArgument(const ExpressionType* type, int argNo,
			list<RecordAttribute> projections) :
			Expression(type), argNo(argNo), projections(projections)	{}

	~InputArgument()													{}

	int getArgNo() const												{ return argNo; }
	list<RecordAttribute> getProjections() const						{ return projections; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const										{ return ARGUMENT; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			cout << "next thing" << endl;
			const InputArgument& rInputArg =
					dynamic_cast<const InputArgument&>(r);
			/* Is it the same record? */
			const ExpressionType *lExpr = this->getExpressionType();
			const ExpressionType *rExpr = rInputArg.getExpressionType();
			bool cmpExprType1 = *lExpr < *rExpr;
			bool cmpExprType2 = *rExpr < *rExpr;
			bool eqExprType = !cmpExprType1 && !cmpExprType2;
			/* Does this make sense? Do I need equality? */
			if (eqExprType) {
				list<RecordAttribute> lProj = this->getProjections();
				list<RecordAttribute> rProj = rInputArg.getProjections();
				if (lProj.size() != rProj.size()) {
					return lProj.size() < rProj.size();
				}

				list<RecordAttribute>::iterator itLeftArgs = lProj.begin();
				list<RecordAttribute>::iterator itRightArgs = rProj.begin();

				while (itLeftArgs != lProj.end()) {
					RecordAttribute attrLeft = (*itLeftArgs);
					RecordAttribute attrRight = (*itRightArgs);

					bool eqAttr = !(attrLeft < attrRight)
							&& !(attrRight < attrLeft);
					if (!eqAttr) {
						return attrLeft < attrRight;
					}
					itLeftArgs++;
					itRightArgs++;
				}
				return false;
			} else {
				return cmpExprType1;
			}
		} else {
			cout << "InputArg: Not compatible" << endl;
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	/**
	 * ArgumentNo is meant to represent e.g. the left or right child of a Join,
	 * NOT the projections that we need!
	 *
	 * argNo = 0 => lhs of Join
	 * argNo = 1 => rhs of Join, and so on.
	 */
	int argNo;
	/**
	 * Used as if 'slicing' a record ->
	 * Treated as record fields
	 * NOTE: One of them (activeLoop) is virtual
	 */
	list<RecordAttribute> projections;
};

class RecordProjection : public Expression	{
public:
	RecordProjection(ExpressionType* type, Expression* expr, const RecordAttribute& attribute)	:
			Expression(type), expr(expr), attribute(attribute)	{}
	RecordProjection(const ExpressionType* type, Expression* expr, const RecordAttribute& attribute)	:
				Expression(type), expr(expr), attribute(attribute)	{}
	~RecordProjection()								{}

	Expression* getExpr() const						{ return expr; }
	string 	getOriginalRelationName() const			{ return attribute.getOriginalRelationName(); }
	string 	getRelationName() const					{ return attribute.getRelationName(); }
	string  getProjectionName()						{ return attribute.getAttrName(); }
	RecordAttribute  getAttribute() const			{ return attribute; }
	RawValue  accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return RECORD_PROJECTION; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			cout << "Record Proj Hashing" << endl;
			const RecordProjection& rProj =
					dynamic_cast<const RecordProjection&>(r);
			bool cmpAttribute1 = this->getAttribute() < rProj.getAttribute();
			bool cmpAttribute2 = rProj.getAttribute() < this->getAttribute();
			bool eqAttribute = !cmpAttribute1 && !cmpAttribute2;
			/* Does this make sense? Do I need equality? */
			if (eqAttribute) {
				cout << this->getAttribute().getAttrName() << " vs " << rProj.getAttribute().getAttrName() << endl;
				cout << this->getAttribute().getRelationName() << " vs " << rProj.getAttribute().getRelationName() << endl;
				//return this->getExpr() < rProj.getExpr();
				return this->getRelationName() < rProj.getRelationName();
			} else {
				cout << "No record proj match "<<endl;
				cout << this->getAttribute().getAttrName() << " vs "
						<< rProj.getAttribute().getAttrName() << endl;
				cout << this->getAttribute().getRelationName() << " vs " << rProj.getAttribute().getRelationName() << endl;
//				return cmpAttribute1;
				return cmpAttribute1 ? cmpAttribute1 : cmpAttribute2;
			}
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	Expression* expr;
	const RecordAttribute& attribute;
};

class AttributeConstruction	{
public:
	AttributeConstruction(string name, Expression* expr) :
			name(name), expr(expr) 									{}
	~AttributeConstruction()										{}
	string getBindingName() const									{ return name; }
	Expression* getExpression()	const								{ return expr; }
	/* Don't need explicit op. overloading */
private:
	string name;
	Expression* expr;
};

class RecordConstruction : public Expression	{
public:
	RecordConstruction(ExpressionType* type,
			const list<AttributeConstruction>& atts) :
			Expression(type), atts(atts) 							{}
	~RecordConstruction()											{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const									{ return RECORD_CONSTRUCTION; }
	const list<AttributeConstruction>& getAtts() const				{ return atts; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const RecordConstruction& rCons =
					dynamic_cast<const RecordConstruction&>(r);

			list<AttributeConstruction> lAtts = this->getAtts();
			list<AttributeConstruction> rAtts = rCons.getAtts();

			if (lAtts.size() != rAtts.size()) {
				return lAtts.size() < rAtts.size();
			}
			list<AttributeConstruction>::iterator itLeft = lAtts.begin();
			list<AttributeConstruction>::iterator itRight = rAtts.begin();

			while (itLeft != lAtts.end()) {
				if (itLeft->getExpression() != itRight->getExpression()) {
					return itLeft->getExpression() < itRight->getExpression();
				}
				itLeft++;
				itRight++;
			}
			return false;
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	const list<AttributeConstruction>& atts;
};

class IfThenElse : public Expression	{
public:
	IfThenElse(ExpressionType* type, Expression* expr1, Expression* expr2, Expression* expr3) :
			Expression(type), expr1(expr1), expr2(expr2), expr3(expr3)			{}
	~IfThenElse()																{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const												{ return IF_THEN_ELSE; }
	Expression* getIfCond()	const												{ return expr1; }
	Expression* getIfResult() const												{ return expr2; }
	Expression* getElseResult()	const											{ return expr3; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const IfThenElse& rIf = dynamic_cast<const IfThenElse&>(r);

			Expression *lCond = this->getIfCond();
			Expression *lIfResult = this->getIfResult();
			Expression *lElseResult = this->getElseResult();

			Expression *rCond = rIf.getIfCond();
			Expression *rIfResult = rIf.getIfResult();
			Expression *rElseResult = rIf.getElseResult();

			bool eqCond = !((*lCond) < (*rCond)) && !((*rCond) < (*lCond));
			bool eqIfResult = !((*lIfResult) < (*rIfResult))
					&& !((*rIfResult) < (*lIfResult));
			bool eqElseResult = !((*lElseResult) < (*rElseResult))
					&& !((*rElseResult) < (*lElseResult));

			if (eqCond) {
				if (eqIfResult) {
					if (eqElseResult) {
						return false;
					} else {
						return (*lElseResult) < (*rElseResult);
					}
				} else {
					return (*lIfResult) < (*rIfResult);
				}
			} else {
				return (*lCond) < (*rCond);
			}

		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	Expression *expr1;
	Expression *expr2;
	Expression *expr3;
};

class BinaryExpression : public Expression	{
public:
	BinaryExpression(ExpressionType* type, expressions::BinaryOperator* op, Expression* lhs, Expression* rhs) :
		Expression(type), lhs(lhs), rhs(rhs), op(op)			{}
	virtual Expression* getLeftOperand() const					{ return lhs; }
	virtual Expression* getRightOperand() const					{ return rhs; }
	expressions::BinaryOperator* getOp() const					{ return op; }

	virtual RawValue accept(ExprVisitor &v) = 0;
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*) = 0;
	virtual ExpressionId getTypeID() const						{ return BINARY; }
	~BinaryExpression() = 0;
	virtual inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				string error_msg =
						string(
								"[This abstract bin. operator is NOT responsible for this case!]");
				LOG(ERROR)<< error_msg;
				throw runtime_error(error_msg);
			}	else	{
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	Expression* lhs;
	Expression* rhs;
	BinaryOperator* op;
};

class EqExpression: public BinaryExpression {
public:
	EqExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
			BinaryExpression(type, new Eq(), lhs, rhs) {
	}
	~EqExpression() {
	}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const {
		return BINARY;
	}
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const EqExpression& rEq = dynamic_cast<const EqExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rEq.getLeftOperand();
				Expression *r2 = rEq.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class NeExpression : public BinaryExpression	{
public:
	NeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Neq(),lhs,rhs) 	{}
	~NeExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const NeExpression& rNe = dynamic_cast<const NeExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rNe.getLeftOperand();
				Expression *r2 = rNe.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class GeExpression : public BinaryExpression	{
public:
	GeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Ge(),lhs,rhs) 	{}
	~GeExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const GeExpression& rGe = dynamic_cast<const GeExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rGe.getLeftOperand();
				Expression *r2 = rGe.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class GtExpression : public BinaryExpression	{
public:
	GtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Gt(),lhs,rhs)		{}
	~GtExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const GtExpression& rGt = dynamic_cast<const GtExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rGt.getLeftOperand();
				Expression *r2 = rGt.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class LeExpression : public BinaryExpression	{
public:
	LeExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Le(),lhs,rhs) 	{}
	~LeExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const LeExpression& rNe = dynamic_cast<const LeExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rNe.getLeftOperand();
				Expression *r2 = rNe.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class LtExpression : public BinaryExpression	{
public:
	LtExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Lt(),lhs,rhs) 	{}
	~LtExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const LtExpression& rNe = dynamic_cast<const LtExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rNe.getLeftOperand();
				Expression *r2 = rNe.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class AddExpression : public BinaryExpression	{
public:
	AddExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Add(),lhs,rhs) 	{}
	~AddExpression()								{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const AddExpression& rAdd =
						dynamic_cast<const AddExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rAdd.getLeftOperand();
				Expression *r2 = rAdd.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class SubExpression : public BinaryExpression	{
public:
	SubExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Sub(),lhs,rhs) 	{}
	~SubExpression()								{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const SubExpression& rSub =
						dynamic_cast<const SubExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rSub.getLeftOperand();
				Expression *r2 = rSub.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class MultExpression : public BinaryExpression	{
public:
	MultExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Mult(),lhs,rhs) 	{}
	~MultExpression()								{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const MultExpression& rMul =
						dynamic_cast<const MultExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rMul.getLeftOperand();
				Expression *r2 = rMul.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class DivExpression : public BinaryExpression	{
public:
	DivExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Div(),lhs,rhs) 	{}
	~DivExpression()								{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const DivExpression& rDiv =
						dynamic_cast<const DivExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rDiv.getLeftOperand();
				Expression *r2 = rDiv.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class AndExpression : public BinaryExpression	{
public:
	AndExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new And(),lhs,rhs) 	{}
	~AndExpression()								{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const AndExpression& rSub =
						dynamic_cast<const AndExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rSub.getLeftOperand();
				Expression *r2 = rSub.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};

class OrExpression : public BinaryExpression	{
public:
	OrExpression(ExpressionType* type, Expression* lhs, Expression* rhs) :
		BinaryExpression(type,new Or(),lhs,rhs) 	{}
	~OrExpression()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return BINARY; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin =
					dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const OrExpression& rSub = dynamic_cast<const OrExpression&>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rSub.getLeftOperand();
				Expression *r2 = rSub.getRightOperand();

				bool eq1 = *l1 < *r1;
				bool eq2 = *l2 < *r2;

				if (eq1) {
					if (eq2) {
						return false;
					} else {
						return *l2 < *r2;
					}
				} else {
					return *l1 < *r1;
				}
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}
};
inline bool operator<(const expressions::BinaryExpression& l,
		const expressions::BinaryExpression& r) {

	expressions::BinaryOperator *lOp = l.getOp();
	expressions::BinaryOperator *rOp = r.getOp();

	bool sameOp = !(lOp->getID() < rOp->getID())
			&& !(rOp->getID() < lOp->getID());
	if (sameOp) {
		Expression *l1 = l.getLeftOperand();
		Expression *l2 = l.getRightOperand();

		Expression *r1 = r.getLeftOperand();
		Expression *r2 = r.getRightOperand();

		bool eq1;
		if (l1->getTypeID() == r1->getTypeID()) {
			eq1 = !(*l1 < *r1) && !(*r1 < *l1);
		} else {
			eq1 = false;
		}

		bool eq2;
		if (l2->getTypeID() == r2->getTypeID()) {
			eq2 = !(*l2 < *r2) && !(*r2 < *l2);
		} else {
			eq2 = false;
		}

		if (eq1) {
			if (eq2) {
				return false;
			} else {
				return *l2 < *r2;
			}
		} else {
			return *l1 < *r1;
		}
	} else {
		return lOp->getID() < rOp->getID();
	}
}

/* (Hopefully) won't be needed */
//inline bool operator<(const expressions::EqExpression& l,
//		const expressions::EqExpression& r)	{
//
//	Expression *l1 = l.getLeftOperand();
//	Expression *l2 = l.getRightOperand();
//
//	Expression *r1 = r.getLeftOperand();
//	Expression *r2 = r.getRightOperand();
//
//	bool eq1 = !(*l1 < *r1) && !(*r1 < *l1);
//	bool eq2 = !(*l2 < *r2) && !(*r2 < *l2);
//
//	if(eq1)	{
//		if(eq2)
//		{
//			return false;
//		}
//		else
//		{
//			return *l2 < *r2;
//		}
//	}
//	else
//	{
//		return *l1 < *r1;
//	}
//}
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
	virtual RawValue visit(expressions::RecordConstruction *e)	= 0;
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
//	virtual RawValue visit(expressions::MergeExpression *e)  	= 0;
	virtual ~ExprVisitor() {}
};

class ExprTandemVisitor
{
public:
	virtual RawValue visit(expressions::IntConstant *e1,
			expressions::IntConstant *e2) = 0;
	virtual RawValue visit(expressions::FloatConstant *e1,
			expressions::FloatConstant *e2) = 0;
	virtual RawValue visit(expressions::BoolConstant *e1,
			expressions::BoolConstant *e2) = 0;
	virtual RawValue visit(expressions::StringConstant *e1,
			expressions::StringConstant *e2) = 0;
	virtual RawValue visit(expressions::InputArgument *e1,
			expressions::InputArgument *e2) = 0;
	virtual RawValue visit(expressions::RecordProjection *e1,
			expressions::RecordProjection *e2) = 0;
	virtual RawValue visit(expressions::RecordConstruction *e1,
			expressions::RecordConstruction *e2) = 0;
	virtual RawValue visit(expressions::IfThenElse *e1,
			expressions::IfThenElse *e2) = 0;
	virtual RawValue visit(expressions::EqExpression *e1,
			expressions::EqExpression *e2) = 0;
	virtual RawValue visit(expressions::NeExpression *e1,
			expressions::NeExpression *e2) = 0;
	virtual RawValue visit(expressions::GeExpression *e1,
			expressions::GeExpression *e2) = 0;
	virtual RawValue visit(expressions::GtExpression *e1,
			expressions::GtExpression *e2) = 0;
	virtual RawValue visit(expressions::LeExpression *e1,
			expressions::LeExpression *e2) = 0;
	virtual RawValue visit(expressions::LtExpression *e1,
			expressions::LtExpression *e2) = 0;
	virtual RawValue visit(expressions::AddExpression *e1,
			expressions::AddExpression *e2) = 0;
	virtual RawValue visit(expressions::SubExpression *e1,
			expressions::SubExpression *e2) = 0;
	virtual RawValue visit(expressions::MultExpression *e1,
			expressions::MultExpression *e2) = 0;
	virtual RawValue visit(expressions::DivExpression *e1,
			expressions::DivExpression *e2) = 0;
	virtual RawValue visit(expressions::AndExpression *e1,
			expressions::AndExpression *e2) = 0;
	virtual RawValue visit(expressions::OrExpression *e1,
			expressions::OrExpression *e2) = 0;
	virtual ~ExprTandemVisitor() {}
};


#endif /* EXPRESSIONS_HPP_ */
