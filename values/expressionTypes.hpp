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

#include <iostream>
#include <stdexcept>
#include <list>
#include <sstream>

using std::cout;
using std::runtime_error;
using std::string;
using std::list;
using std::stringstream;


enum typeID	{BOOL, STRING, FLOAT, INT, COLLECTION, RECORD};

class ExpressionType {
public:
	virtual string getType() = 0;
	virtual typeID getTypeID() = 0;
	virtual ~ExpressionType() { }
	virtual bool isPrimitive() = 0;
};

class PrimitiveType : public ExpressionType	{};


class BoolType : public PrimitiveType {
public:
	string getType() {return string("Bool");}
	typeID getTypeID()	{return BOOL;}
	bool isPrimitive() {return true;}
};
class StringType : public PrimitiveType {
public:
	string getType() {return string("String");}
	typeID getTypeID()	{return STRING;}
	bool isPrimitive() {return true;}
};
class FloatType : public PrimitiveType {
public:
	string getType() {return string("Float");}
	typeID getTypeID()	{return FLOAT;}
	bool isPrimitive() {return true;}
};
class IntType : public PrimitiveType {
public:
	string getType() {return string("Int");}
	typeID getTypeID()	{return INT;}
	bool isPrimitive() {return true;}
};

//TODO missing constructor
class CollectionType : public ExpressionType	{
public:
	string getType() {
		return string("CollectionType(")+type->getType()+string(")");
	};
	typeID getTypeID()	{return COLLECTION;}
	bool isPrimitive() {return false;}
	~CollectionType() { };
private:
	ExpressionType* type;

};


class RecordAttribute	{
public:

	RecordAttribute(int no, string name_, ExpressionType* type_)
		: attrNo(no), name(name_), type(type_), projected(false) 	{}

	string getType() 												{ return name +" "+type->getType(); };
	ExpressionType* getOriginalType() 								{	return type; }
	string getName()												{ return name;	}
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

	string getType() {
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
	typeID getTypeID()	{ return RECORD; }
	std::list<RecordAttribute*>& getArgs() { return args;}
	int getArgsNo() {return args.size();}
	bool isPrimitive() {return false;}
	~RecordType() { };

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
	virtual void visit(CollectionType type) = 0;
	virtual void visit(RecordType type) = 0;
	virtual ~ExprTypeVisitor();
};

#endif /* EXPRESSIONTYPES_HPP_ */
