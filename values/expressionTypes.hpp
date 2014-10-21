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

enum typeID	{ BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET };

class ExpressionType {
public:
	virtual string getType()   const = 0;
	virtual typeID getTypeID() const = 0;
	virtual ~ExpressionType()  {}
	virtual bool isPrimitive() = 0;
};

class PrimitiveType : public ExpressionType	{};

class BoolType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Bool"); }
	typeID getTypeID() 	const	{ return BOOL; }
	bool isPrimitive()			{ return true; }
};

class StringType : public PrimitiveType {
public:
	string getType() 	const	{ return string("String"); }
	typeID getTypeID() 	const	{ return STRING; }
	bool isPrimitive() 			{ return true; }
};

class FloatType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Float"); }
	typeID getTypeID()	const	{ return FLOAT; }
	bool isPrimitive() 			{ return true; }
};

class IntType : public PrimitiveType {
public:
	string getType() 	const 	{ return string("Int"); }
	typeID getTypeID()	const	{ return INT; }
	bool isPrimitive() 			{ return true; }
};

class CollectionType : public ExpressionType	{
public:
	CollectionType(	const ExpressionType& type) :
		type(type)									{}
	virtual string getType() 	const				{ return string("CollectionType(")+type.getType()+string(")"); }
	virtual typeID getTypeID() 	const = 0;
	virtual bool isPrimitive() 						{ return false; }
	const ExpressionType& getNestedType() const		{ return type; }
	virtual ~CollectionType() 						{}

private:
	const ExpressionType& type;
};

class ListType : public CollectionType	{
public:
	ListType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("ListType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID() 	const									{ return LIST; }
	bool isPrimitive() 											{ return false; }
	~ListType() 												{}
};

class BagType : public CollectionType	{
public:
	BagType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("BagType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID() 	const									{ return BAG; }
	bool isPrimitive() 											{ return false; }
	~BagType() 													{}
};

class SetType : public CollectionType	{
public:
	SetType(const ExpressionType& type) : CollectionType(type)	{}
	string getType() 	const									{ return string("SetType(")+this->getNestedType().getType()+string(")"); }
	typeID getTypeID()	const									{ return SET; }
	bool isPrimitive() 											{ return false; }
	~SetType() 													{}
};


class RecordAttribute	{
public:

	RecordAttribute(int no, string name_, ExpressionType* type_)
		: attrNo(no), name(name_), type(type_), projected(false) 	{}

	string getType() const											{ return name +" "+type->getType(); }
	ExpressionType* getOriginalType() 								{ return type; }
	string getName()												{ return name; }
	//CONVENTION: Fields requested can be 1-2-3-etc.
	int getAttrNo()													{ return attrNo; }
	void setProjected()												{ projected = true; }
	bool isProjected()												{ return projected; }
private:
	string name;
	ExpressionType* type;
	int attrNo;
	bool projected;
};

bool recordComparator (RecordAttribute* x, RecordAttribute* y);

class RecordType : public ExpressionType	{
public:

	RecordType(list<RecordAttribute*>& args_) : args(args_) {}

	string getType() const {
		stringstream ss;
		ss<<"Record(";
		int count = 0;
		int size = args.size();
		for (std::list<RecordAttribute*>::iterator it = args.begin(); it != args.end(); it++) {
			ss<<(*it)->getType();
			count++;
			if(count != size) {
				ss<<", ";
			}
		}
		ss<<")";
		return ss.str();
	}
	typeID getTypeID()	const					{ return RECORD; }
	std::list<RecordAttribute*>& getArgs() 		{ return args; }
	int getArgsNo() 							{ return args.size(); }
	bool isPrimitive() 							{ return false; }
	~RecordType() 								{}

private:
	list<RecordAttribute*>& args;

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

#endif /* EXPRESSIONTYPES_HPP_ */
