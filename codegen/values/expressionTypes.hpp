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

#ifndef EXPRESSIONTYPES_HPP_
#define EXPRESSIONTYPES_HPP_

#include "common/common.hpp"

/* Original.*/
//enum typeID	{ BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET, BLOCK };
/* Extended due to caching (for now)*/
enum typeID	{ BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET, INT64, COMPOSITE, BLOCK};

class ExpressionType {
public:
	virtual string getType()   const = 0;
	virtual typeID getTypeID() const = 0;
	virtual ~ExpressionType()  {}
	virtual bool isPrimitive() const = 0;
	virtual Type *getLLVMType(LLVMContext &ctx) const {
		string error_msg = string("Type " + getType() + " is not mapped into an LLVM-type.");
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
};

class PrimitiveType : public ExpressionType	{
public:
	virtual Type *getLLVMType(LLVMContext &ctx) const = 0;
	virtual size_t getSizeInBits(LLVMContext &ctx) const{
		Type * llvm_type = getLLVMType(ctx);
		return llvm_type->getPrimitiveSizeInBits();
	}
};

class BoolType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Bool"); }
	typeID getTypeID() 	const	{ return BOOL; }
	bool isPrimitive()	const	{ return true; }
	Type *getLLVMType(LLVMContext &ctx) const { return Type::getInt1Ty(ctx);}
};

class StringType : public PrimitiveType {
public:
	string getType() 	const	{ return string("String"); }
	typeID getTypeID() 	const	{ return STRING; }
	bool isPrimitive() 	const	{ return true; }
	Type *getLLVMType(LLVMContext &ctx) const { return Type::getInt8PtrTy(ctx);}
};

class FloatType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Float"); }
	typeID getTypeID()	const	{ return FLOAT; }
	bool isPrimitive() 	const	{ return true; }
	Type *getLLVMType(LLVMContext &ctx) const { return Type::getDoubleTy(ctx);}
};

class IntType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Int"); }
	typeID getTypeID()	const	{ return INT; }
	bool isPrimitive() 	const	{ return true; }
	Type *getLLVMType(LLVMContext &ctx) const { return Type::getInt32Ty(ctx);}
};

class Int64Type : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Int64"); }
	typeID getTypeID()	const	{ return INT64; }
	bool isPrimitive() 	const	{ return true; }
	Type *getLLVMType(LLVMContext &ctx) const { return Type::getInt64Ty(ctx);}
};

class CollectionType : public ExpressionType	{
public:
	CollectionType(	const ExpressionType& type) :
		type(type)									{}
	virtual string getType() 	const				{ return string("CollectionType(")+type.getType()+string(")"); }
	virtual typeID getTypeID() 	const = 0;
	virtual bool isPrimitive() 	const				{ return false; }
	const ExpressionType& getNestedType() const		{ return type; }
	virtual ~CollectionType() 						{}

private:
	const ExpressionType& type;
};

class BlockType : public CollectionType	{
public:
	BlockType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("BlockType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID() 	const									{ return BLOCK; }
	bool isPrimitive() 	const									{ return false; }
	~BlockType() 												{}
	Type *getLLVMType(LLVMContext &ctx) const 					{ return PointerType::get(getNestedType().getLLVMType(ctx), 0);}
};

class ListType : public CollectionType	{
public:
	ListType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("ListType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID() 	const									{ return LIST; }
	bool isPrimitive() 	const									{ return false; }
	~ListType() 												{}
};

class BagType : public CollectionType	{
public:
	BagType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("BagType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID() 	const									{ return BAG; }
	bool isPrimitive() 	const									{ return false; }
	~BagType() 													{}
};

class SetType : public CollectionType	{
public:
	SetType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("SetType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID()	const									{ return SET; }
	bool isPrimitive() 	const									{ return false; }
	~SetType() 													{}
};


class RecordAttribute	{
public:
	RecordAttribute() : attrNo(-1), projected(false), type(NULL), relName(""), attrName("")	{
		cout << "ANONYMOUS CONSTRUCTOR!!" << endl;
	}
	RecordAttribute(int no, string relName, string attrName, const ExpressionType* type, bool make_block = false)
		: attrNo(no), relName(relName), originalRelName(relName), attrName(attrName), type(make_block ? new BlockType(*type) : type), projected(false) 	{}
	RecordAttribute(int no, string relName, const char* attrName, const ExpressionType* type)
			: attrNo(no), relName(relName), originalRelName(relName), type(type), projected(false) 	{
		this->attrName = string(attrName);
	}
	RecordAttribute(int no, string originalRelName, string relName, string attrName, const ExpressionType* type)
			: attrNo(no), relName(relName), originalRelName(originalRelName), attrName(attrName), type(type), projected(false) 	{}

	//Constructor used STRICTLY for comparisons in maps
//	RecordAttribute(string relName, string attrName)
//			: attrNo(-1), relName(relName), attrName(attrName), type(NULL), projected(false) 	{}

	/* OID Type needed so that we know what we materialize
	 * => Subsequent functions / programs use info to parse caches */
	RecordAttribute(string relName, string attrName, const ExpressionType* type)
			: attrNo(-1), relName(relName), attrName(attrName), type(type), projected(false) 	{
		//cout << "RELNAME:[" << relName << "]" << endl;
		if(relName == "")	{
			string error_msg = string("Unexpected, no-relname attribute: ") + attrName;
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}
	}

	RecordAttribute(const RecordAttribute& obj, bool make_block) : type(make_block ? new BlockType(*(obj.getOriginalType())) : obj.getOriginalType()) {
		this->attrNo = obj.attrNo;
		this->attrName = obj.attrName;
		this->relName = obj.relName;
		this->originalRelName = obj.originalRelName;
		this->projected = obj.projected;
	}

//	RecordAttribute(const RecordAttribute& obj) : type(obj.getOriginalType()) {
//		this->attrNo = obj.attrNo;
//		this->relName = obj.attrName;
//		this->projected = obj.projected;
//	}

	string getType() const											{
		return attrName +" "+type->getType();
	}
	const ExpressionType* getOriginalType() const					{ return type; }
	Type * getLLVMType(LLVMContext &ctx) 	const					{ return getOriginalType()->getLLVMType(ctx); }
	string getName()						const					{ return attrName; }
	string getRelationName() 				const					{ return relName; }
	string getOriginalRelationName() 		const					{ return originalRelName; }
	string getAttrName() 					const					{ return attrName; }
	//CONVENTION: Fields requested can be 1-2-3-etc.
	int getAttrNo()							const					{ return attrNo; }
	void setProjected()												{ projected = true; }
	bool isProjected()												{ return projected; }

private:
	string relName;
	string attrName;
	string originalRelName;
	const ExpressionType* type;
	//Atm, used by eager CSV plugin (for the native relations)
	int attrNo;
	bool projected;

};

class RecordType : public ExpressionType	{
public:

	RecordType() {}
	RecordType(list<RecordAttribute*> args) : args(args) {
		list<RecordAttribute*>::const_iterator it = args.begin();
		for(; it != args.end(); it++)	{
			RecordAttribute* arg = *it;
			argsMap[arg->getAttrName()] = arg;
		}
	}

	RecordType(vector<RecordAttribute*> args) {
		vector<RecordAttribute*>::iterator it = args.begin();
		for (; it != args.end(); it++) {
			RecordAttribute* arg = *it;
			this->args.push_back(arg);
			argsMap[arg->getAttrName()] = arg;
		}
	}

	RecordType(const RecordType &rec) {
		argsMap.insert(rec.argsMap.begin(), rec.argsMap.end());
		args.insert(args.begin(), rec.args.begin(), rec.args.end());
	}

	string getType() const {
		stringstream ss;
		ss<<"Record(";
		int count = 0;
		int size = args.size();
		for (list<RecordAttribute*>::const_iterator it = args.begin(); it != args.end(); it++) {
			ss<<(*it)->getType();
			count++;
			if(count != size) {
				ss<<", ";
			}
		}
		ss<<")";
		return ss.str();
	}

	virtual Type *getLLVMType(LLVMContext &ctx) const {
		std::vector<Type *> body;
		for (const auto &attr: args) body.push_back(attr->getLLVMType(ctx));
		return StructType::get(ctx, body);
	}

	typeID getTypeID()	const					{ return RECORD; }
	list<RecordAttribute*> getArgs() const 		{ return args; }
	map<string, RecordAttribute*>& getArgsMap()	{ return argsMap; }
	int getArgsNo() const						{ return args.size(); }
	bool isPrimitive() 	const					{ return false; }
	~RecordType() 								{}

	void appendAttribute(RecordAttribute * attr){
#ifndef NDEBUG
		auto inserted = 
#endif
		argsMap.emplace(attr->getAttrName(), attr);
		assert(inserted.second && "Attribute already exists!");
		args.push_back(attr);
	}

	const RecordAttribute * getArg(string name) const {
		auto r = argsMap.find(name);
		if (r == argsMap.end()) return NULL;
		return r->second;
	}

	Value * projectArg(Value * record, RecordAttribute * attr, IRBuilder<>* const Builder) const;

private:
	list<RecordAttribute*> args;
	map<string, RecordAttribute*> argsMap;

	int getIndex(RecordAttribute * x) const;
};

class ExprTypeVisitor
{
public:
	virtual void visit(IntType type) = 0;
	virtual void visit(BoolType type) = 0;
	virtual void visit(FloatType type) = 0;
	virtual void visit(StringType type) = 0;
	virtual void visit(RecordType type) = 0;
	virtual ~ExprTypeVisitor();
};


/* XXX Not too sure these comparators make sense.
 * If difference between hashed expressions boils down to this
 * point, I am doing sth wrong. */
inline bool operator<(const ExpressionType& l, const ExpressionType& r) {
	cout << "Comparing GENERIC EXPRESSION TYPE" << endl;
	return l.getType() < r.getType();
}

bool recordComparator (RecordAttribute* x, RecordAttribute* y);
inline bool operator<(const RecordAttribute& l, const RecordAttribute& r) {
	if (l.getRelationName() == r.getRelationName()) {
		return l.getAttrName() < r.getAttrName();
	} else {
		return l.getRelationName() < r.getRelationName();
	}
}
//
//
//inline bool operator<(const RecordType& l, const RecordType& r) {
//	list<RecordAttribute*>& leftArgs = l.getArgs();
//	list<RecordAttribute*>& rightArgs = r.getArgs();
//
//	if (leftArgs.size() != rightArgs.size()) {
//		return leftArgs.size() < rightArgs.size();
//	}
//
//	list<RecordAttribute*>::iterator itLeftArgs = leftArgs.begin();
//	list<RecordAttribute*>::iterator itRightArgs = rightArgs.begin();
//
//	while (itLeftArgs != leftArgs.end()) {
//		RecordAttribute attrLeft = *(*itLeftArgs);
//		RecordAttribute attrRight = *(*itRightArgs);
//
//		bool eqAttr = !(attrLeft < attrRight) && !(attrRight < attrLeft);
//		if (!eqAttr) {
//			return attrLeft < attrRight;
//		}
//		itLeftArgs++;
//		itRightArgs++;
//	}
//	return false;
//}
//;

#endif /* EXPRESSIONTYPES_HPP_ */
