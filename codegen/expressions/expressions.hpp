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
#include "operators/monoids.hpp"

class ExprVisitor; 		 //Forward declaration
class ExprTandemVisitor; //Forward declaration

//Careful: Using a namespace to avoid conflicts witfh LLVM namespace
namespace expressions
{


enum ExpressionId	{ CONSTANT, RAWVALUE, ARGUMENT, RECORD_PROJECTION, RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE, EXPRESSION_HASHER, TESTNULL_EXPRESSION, EXTRACT, NEG_EXPRESSION, CAST_EXPRESSION};

class Expression	{
public:
	Expression(const ExpressionType* type) : type(type), registered(false){}
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

	virtual inline bool isRegistered(){
		return registered;
	}

	virtual inline void registerAs(string relName, string attrName){
		registered     = true;
		this->relName  = relName ;
		this->attrName = attrName;
	}

	virtual inline void registerAs(RecordAttribute * attr){
		registerAs(attr->getRelationName(), attr->getAttrName());
	}

	virtual inline string getRegisteredAttrName(){
		if (!registered){
			string error_msg = string("Expression not registered");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		return attrName;
	}

	virtual inline string getRegisteredRelName(){
		if (!registered){
			string error_msg = string("Expression not registered");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		return relName;
	}

	virtual RecordAttribute getRegisteredAs(){
		if (!registered){
			string error_msg = string("Expression not registered");
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}
		return RecordAttribute{
								getRegisteredRelName() , 
								getRegisteredAttrName(), 
								getExpressionType()
							};
	}

private:
	const ExpressionType* type;
protected:
	bool registered;
	string relName ;
	string attrName;
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
		INT, INT64, BOOL, FLOAT, STRING, DSTRING, DATE
	};
	Constant(const ExpressionType* type) : Expression(type)	{}
	~Constant()										  	{}
	virtual ConstantType getConstantType() const = 0;
};

template<typename T, typename Tproteus, Constant::ConstantType TcontType>
class TConstant: public Constant{
private:
	T val;
protected:
	TConstant(T val, const ExpressionType *type): Constant(type), val(val){}
public:
	TConstant(T val): TConstant(val, new Tproteus()){}
	~TConstant(){}

	T getVal() const								{ return val; }

	RawValue accept(ExprVisitor &v) = 0;
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*) = 0;

	ExpressionId getTypeID() const					{ return CONSTANT; }

	ConstantType getConstantType() const { return TcontType; }

	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const Constant& rConst = dynamic_cast<const Constant&>(r);
			if (rConst.getConstantType() == getConstantType()) {
				const auto &r2 = 
					dynamic_cast<const TConstant<T, Tproteus, TcontType> &>(r);
				return this->getVal() < r2.getVal();
			}
			else {
				return this->getConstantType() < rConst.getConstantType();
			}
		}
		cout << "Not compatible" << endl;
		return this->getTypeID() < r.getTypeID();
	}
};

class IntConstant    : public TConstant<int        , IntType    , Constant::ConstantType::INT   > {
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class StringConstant : public TConstant<std::string, StringType , Constant::ConstantType::STRING> {
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class Int64Constant  : public TConstant<int64_t    , Int64Type  , Constant::ConstantType::INT64 > {
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class DateConstant   : public TConstant<int64_t    , DateType   , Constant::ConstantType::DATE  > {
static_assert(sizeof(time_t) == sizeof(int64_t), "expected 64bit time_t");
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class BoolConstant   : public TConstant<bool       , BoolType   , Constant::ConstantType::BOOL  > {
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class FloatConstant  : public TConstant<double     , FloatType  , Constant::ConstantType::FLOAT > {
public:
	using TConstant::TConstant;
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class DStringConstant : public TConstant<int       , DStringType, Constant::ConstantType::FLOAT > {
public:
	DStringConstant(int val, void * dictionary)
		: TConstant(val, new DStringType(dictionary)){}
	~DStringConstant()									{}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
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
	[[deprecated]]
	RecordProjection(ExpressionType* type, Expression* expr, RecordAttribute attribute)	:
			Expression(type), expr(expr), attribute(attribute)	{
					assert(type->getTypeID() == attribute.getOriginalType()->getTypeID());
					registered = true;
					relName    = getRelationName();
					attrName   = getProjectionName();
				}
	[[deprecated]]
	RecordProjection(const ExpressionType* type, Expression* expr, RecordAttribute attribute)	:
				Expression(type), expr(expr), attribute(attribute)	{
					assert(type->getTypeID() == attribute.getOriginalType()->getTypeID());
					registered = true;
					relName    = getRelationName();
					attrName   = getProjectionName();
				}
	RecordProjection(Expression* expr, RecordAttribute attribute)	:
			Expression(attribute.getOriginalType()), expr(expr), attribute(attribute)	{
					registered = true;
					relName    = getRelationName();
					attrName   = getProjectionName();
				}
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
	//		if (this->getTypeID() == r.getTypeID()) {
	//			cout << "Record Proj Hashing" << endl;
	//			const RecordProjection& rProj =
	//					dynamic_cast<const RecordProjection&>(r);
	//			bool cmpAttribute1 = this->getAttribute() < rProj.getAttribute();
	//			bool cmpAttribute2 = rProj.getAttribute() < this->getAttribute();
	//			bool eqAttribute = !cmpAttribute1 && !cmpAttribute2;
	//			/* Does this make sense? Do I need equality? */
	//			if (eqAttribute) {
	//				cout << this->getAttribute().getAttrName() << " vs " << rProj.getAttribute().getAttrName() << endl;
	//				cout << this->getAttribute().getRelationName() << " vs " << rProj.getAttribute().getRelationName() << endl;
	//				//return this->getExpr() < rProj.getExpr();
	//				return this->getRelationName() < rProj.getRelationName();
	//			} else {
	//				cout << "No record proj match "<<endl;
	//				cout << this->getAttribute().getAttrName() << " vs "
	//						<< rProj.getAttribute().getAttrName() << endl;
	//				cout << this->getAttribute().getRelationName() << " vs " << rProj.getAttribute().getRelationName() << endl;
	////				return cmpAttribute1;
	//				return cmpAttribute1 ? cmpAttribute1 : cmpAttribute2;
	//			}
	//		} else {
	//			return this->getTypeID() < r.getTypeID();
	//		}
			if (this->getTypeID() == r.getTypeID()) {
				//cout << "Record Proj Hashing" << endl;
				const RecordProjection& rProj =
						dynamic_cast<const RecordProjection&>(r);

				string n1 = this->getRelationName();
				string n2 = rProj.getRelationName();

				bool cmpRel1 = this->getRelationName() < rProj.getRelationName();
				bool cmpRel2 = rProj.getRelationName() < this->getRelationName();
				bool eqRelation = !cmpRel1 && !cmpRel2;
				if (eqRelation) {
					bool cmpAttribute1 = this->getAttribute()
							< rProj.getAttribute();
					bool cmpAttribute2 = rProj.getAttribute()
							< this->getAttribute();
					bool eqAttribute = !cmpAttribute1 && !cmpAttribute2;
					if (eqAttribute) {
						return false;
					} else {
						return cmpAttribute1;
					}
				} else {
					return cmpRel1;
				}

			} else {
				return this->getTypeID() < r.getTypeID();
			}
		}
private:
	Expression* expr;
	RecordAttribute attribute;
};

class HashExpression : public Expression	{
public:
	HashExpression(Expression* expr):
			Expression(new Int64Type()), expr(expr) {}

	~HashExpression()								{}

	Expression* getExpr() const						{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return EXPRESSION_HASHER; }
	inline bool operator<(const expressions::Expression& r) const {
			if (this->getTypeID() == r.getTypeID()) {
				const HashExpression& rHash = dynamic_cast<const HashExpression&>(r);
				return *(this->getExpr()) < *(rHash.getExpr());
			} else {
				return this->getTypeID() < r.getTypeID();
			}
		}
private:
	Expression* expr;
};

class RawValueExpression : public Expression	{
public:
	RawValueExpression(const ExpressionType* type, RawValue expr)	:
				Expression(type), expr(expr){}
	~RawValueExpression()								{}

	RawValue getValue() const							{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const						{ return RAWVALUE; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const RawValueExpression& rProj = dynamic_cast<const RawValueExpression&>(r);
			return (expr.value == NULL && rProj.expr.value == NULL) ? (expr.isNull < rProj.expr.isNull) : (expr.value < rProj.expr.value);
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	RawValue expr;
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

/*
 * XXX
 * I think that unless it belongs to the final result, it is desugarized!!
 */
class RecordConstruction : public Expression	{
public:
	[[deprecated]]
	RecordConstruction(const ExpressionType* type,
			const list<AttributeConstruction>& atts) :
			Expression(type), atts(atts) 							{
				assert(type->getTypeID() == RECORD);
			}

	RecordConstruction(const list<AttributeConstruction>& atts) :
			Expression(constructRecordType(atts)), atts(atts) {
			}
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

	RecordType *constructRecordType(const list<AttributeConstruction> &attrs){
		vector<RecordAttribute *> recs;
		for (const auto &a: attrs) {
			auto *type = a.getExpression()->getExpressionType();
			auto  attr = new RecordAttribute{"tmp", a.getBindingName(), type};
			recs.emplace_back(attr);
		}
		return new RecordType(recs);
	}
private:
	const list<AttributeConstruction>& atts;
};

class IfThenElse : public Expression	{
public:
	[[deprecated]]
	IfThenElse(const ExpressionType* type, Expression* expr1, Expression* expr2, Expression* expr3) :
			Expression(type), expr1(expr1), expr2(expr2), expr3(expr3)			{}
	IfThenElse(Expression* expr1, Expression* expr2, Expression* expr3) :
			Expression(expr2->getExpressionType()), expr1(expr1), expr2(expr2), expr3(expr3){
				assert(expr2->getExpressionType()->getTypeID() == expr3->getExpressionType()->getTypeID());
			}
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
	BinaryExpression(const ExpressionType* type, expressions::BinaryOperator* op, Expression* lhs, Expression* rhs) :
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

template<typename Top>
class TBinaryExpression: public BinaryExpression{
protected:
	TBinaryExpression(const ExpressionType* type, Expression* lhs, Expression* rhs) :
			BinaryExpression(type, new Top(), lhs, rhs) {
	}

public:
	~TBinaryExpression() {
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
				const TBinaryExpression<Top> &rOp = 
					dynamic_cast<const TBinaryExpression<Top> &>(r);
				Expression *l1 = this->getLeftOperand();
				Expression *l2 = this->getRightOperand();

				Expression *r1 = rOp.getLeftOperand();
				Expression *r2 = rOp.getRightOperand();

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


class EqExpression: public TBinaryExpression<Eq> {
public:
	// [[deprecated]]
	// EqExpression(const ExpressionType* type, Expression* lhs, Expression* rhs):
	// 	TBinaryExpression(type, lhs, rhs){}
	EqExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~EqExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class NeExpression: public TBinaryExpression<Neq> {
public:
	NeExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~NeExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class GeExpression: public TBinaryExpression<Ge> {
public:
	GeExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~GeExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class GtExpression: public TBinaryExpression<Gt> {
public:
	GtExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~GtExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class LeExpression: public TBinaryExpression<Le> {
public:
	LeExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~LeExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class LtExpression: public TBinaryExpression<Lt> {
public:
	LtExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(new BoolType(), lhs, rhs){}
	~LtExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class AddExpression: public TBinaryExpression<Add> {
public:
	AddExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~AddExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class SubExpression: public TBinaryExpression<Sub> {
public:
	SubExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~SubExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class MultExpression: public TBinaryExpression<Mult> {
public:
	MultExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~MultExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class DivExpression: public TBinaryExpression<Div> {
public:
	DivExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~DivExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class AndExpression: public TBinaryExpression<And> {
public:
	AndExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~AndExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class OrExpression: public TBinaryExpression<Or> {
public:
	OrExpression(Expression* lhs, Expression* rhs) :
		TBinaryExpression(lhs->getExpressionType(), lhs, rhs){}
	~OrExpression(){}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
};

class MaxExpression: public BinaryExpression {
public:
	MaxExpression(Expression* lhs, Expression* rhs) :
			BinaryExpression(lhs->getExpressionType(), new Max(), lhs, rhs),
			cond(
				new GtExpression(
					lhs,
					rhs
				),
				lhs,
				rhs
			){
	}
	~MaxExpression() {
	}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	IfThenElse * getCond(){return &cond;};
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin = dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const MaxExpression& rMax = dynamic_cast<const MaxExpression&>(r);
				return this->cond < rMax.cond;
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}

private:
	IfThenElse cond;
};

class MinExpression: public BinaryExpression {
public:
	MinExpression(Expression* lhs, Expression* rhs) :
			BinaryExpression(lhs->getExpressionType(), new Min(), lhs, rhs),
			cond(
				new LtExpression(
					lhs,
					rhs
				),
				lhs,
				rhs
			){
	}
	~MinExpression() {
	}

	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	IfThenElse * getCond(){return &cond;};
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const BinaryExpression& rBin = dynamic_cast<const BinaryExpression&>(r);
			if (this->getOp()->getID() == rBin.getOp()->getID()) {
				const MinExpression& rMin = dynamic_cast<const MinExpression&>(r);
				return this->cond < rMin.cond;
			} else {
				return this->getOp()->getID() < rBin.getOp()->getID();
			}
		}
		return this->getTypeID() < r.getTypeID();
	}

private:
	IfThenElse cond;
};




class NegExpression : public Expression	{
public:
	NegExpression(Expression* expr):
			Expression(expr->getExpressionType()), expr(expr) {}

	~NegExpression()								{}

	Expression* getExpr() const						{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return NEG_EXPRESSION; }
	inline bool operator<(const expressions::Expression& r) const {
			if (this->getTypeID() == r.getTypeID()) {
				const NegExpression& rHash = dynamic_cast<const NegExpression&>(r);
				return *(this->getExpr()) < *(rHash.getExpr());
			} else {
				return this->getTypeID() < r.getTypeID();
			}
		}
private:
	Expression* expr;
};

class TestNullExpression : public Expression	{
public:
	TestNullExpression(Expression * expr, bool nullTest = true):
			Expression(new BoolType()), expr(expr), nullTest(nullTest) {}

	~TestNullExpression()								{}
	
	bool isNullTest() const { return nullTest; }
	Expression * getExpr() const						{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const				{ return TESTNULL_EXPRESSION; }
	inline bool operator<(const expressions::Expression& r) const {
			if (this->getTypeID() == r.getTypeID()) {
				const TestNullExpression& rHash = dynamic_cast<const TestNullExpression&>(r);
				return *(this->getExpr()) < *(rHash.getExpr());
			} else {
				return this->getTypeID() < r.getTypeID();
			}
		}
private:
	Expression* expr;
	bool nullTest;
};

enum class extract_unit{
	MILLISECOND,
	SECOND,
	MINUTE,
	HOUR,
	DAYOFWEEK,
	ISO_DAYOFWEEK,
	DAYOFMONTH,
	DAYOFYEAR,
	WEEK,
	MONTH,
	QUARTER,
	YEAR,
	ISO_YEAR,
	DECADE,
	CENTURY,
	MILLENNIUM
};

class ExtractExpression : public Expression	{
public:
	ExtractExpression(Expression * expr, extract_unit unit):
			Expression(createReturnType(unit)), expr(expr), unit(unit) {
		assert(expr->getExpressionType()->getTypeID() == DATE);
	}

	~ExtractExpression()								{}

	extract_unit getExtractUnit() 						{ return unit; }
	Expression * getExpr() const						{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const						{ return EXTRACT; }
	inline bool operator<(const expressions::Expression& r) const {
		if (this->getTypeID() == r.getTypeID()) {
			const ExtractExpression& rHash = dynamic_cast<const ExtractExpression&>(r);
			return *(this->getExpr()) < *(rHash.getExpr());
		} else {
			return this->getTypeID() < r.getTypeID();
		}
	}
private:
	static ExpressionType * createReturnType(extract_unit u);

	Expression * expr;
	extract_unit unit;
};

class CastExpression : public Expression	{
public:
	CastExpression(ExpressionType *cast_to, Expression* expr):
			Expression(cast_to), expr(expr) {}

	~CastExpression()								{}

	Expression* getExpr() const						{ return expr; }
	RawValue accept(ExprVisitor &v);
	RawValue acceptTandem(ExprTandemVisitor &v, expressions::Expression*);
	ExpressionId getTypeID() const					{ return CAST_EXPRESSION; }
	inline bool operator<(const expressions::Expression& r) const {
			if (this->getTypeID() == r.getTypeID()) {
				const CastExpression& rHash = dynamic_cast<const CastExpression&>(r);
				return *(this->getExpr()) < *(rHash.getExpr());
			} else {
				return this->getTypeID() < r.getTypeID();
			}
		}
private:
	Expression* expr;
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
	virtual RawValue visit(expressions::Int64Constant *e) 		= 0;
	virtual RawValue visit(expressions::DateConstant *e) 		= 0;
	virtual RawValue visit(expressions::FloatConstant *e)  		= 0;
	virtual RawValue visit(expressions::BoolConstant *e)   		= 0;
	virtual RawValue visit(expressions::StringConstant *e) 		= 0;
	virtual RawValue visit(expressions::DStringConstant *e) 	= 0;
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
	virtual RawValue visit(expressions::RawValueExpression *e)  = 0;
	virtual RawValue visit(expressions::MinExpression *e) 		= 0;
	virtual RawValue visit(expressions::MaxExpression *e) 		= 0;
	virtual RawValue visit(expressions::HashExpression *e) 		= 0;
	virtual RawValue visit(expressions::TestNullExpression *e) 	= 0;
	virtual RawValue visit(expressions::NegExpression *e) 		= 0;
	virtual RawValue visit(expressions::ExtractExpression *e) 	= 0;
	virtual RawValue visit(expressions::CastExpression *e1) 	= 0;
//	virtual RawValue visit(expressions::MergeExpression *e)  	= 0;
	virtual ~ExprVisitor() {}
};

class ExprTandemVisitor
{
public:
	virtual RawValue visit(expressions::IntConstant *e1,
			expressions::IntConstant *e2) = 0;
	virtual RawValue visit(expressions::Int64Constant *e1,
			expressions::Int64Constant *e2) = 0;
	virtual RawValue visit(expressions::DateConstant *e1,
			expressions::DateConstant *e2) = 0;
	virtual RawValue visit(expressions::FloatConstant *e1,
			expressions::FloatConstant *e2) = 0;
	virtual RawValue visit(expressions::BoolConstant *e1,
			expressions::BoolConstant *e2) = 0;
	virtual RawValue visit(expressions::StringConstant *e1,
			expressions::StringConstant *e2) = 0;
	virtual RawValue visit(expressions::DStringConstant *e1,
			expressions::DStringConstant *e2) = 0;
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
	virtual RawValue visit(expressions::RawValueExpression *e1,
			expressions::RawValueExpression *e2) = 0;
	virtual RawValue visit(expressions::MinExpression *e1,
			expressions::MinExpression *e2) = 0;
	virtual RawValue visit(expressions::MaxExpression *e1,
			expressions::MaxExpression *e2) = 0;
	virtual RawValue visit(expressions::HashExpression *e1,
			expressions::HashExpression *e2) = 0;
	virtual RawValue visit(expressions::NegExpression *e1,
			expressions::NegExpression *e2) = 0;
	virtual RawValue visit(expressions::ExtractExpression *e1,
			expressions::ExtractExpression *e2) = 0;
	virtual RawValue visit(expressions::TestNullExpression *e1,
			expressions::TestNullExpression *e2) = 0;
	virtual RawValue visit(expressions::CastExpression *e1,
			expressions::CastExpression *e2) = 0;
	virtual ~ExprTandemVisitor() {}
};

expressions::Expression * toExpression(Monoid m, expressions::Expression * lhs, expressions::Expression * rhs);

llvm::Constant * getIdentityElementIfSimple(Monoid m, const ExpressionType * type, RawContext * context);

#endif /* EXPRESSIONS_HPP_ */
