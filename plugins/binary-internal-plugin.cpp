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

#include "plugins/binary-internal-plugin.hpp"

BinaryInternalPlugin::BinaryInternalPlugin(RawContext* const context,
		string structName) :
		context(context), structName(structName) {}

BinaryInternalPlugin::~BinaryInternalPlugin() {}

void BinaryInternalPlugin::init()	{};

void BinaryInternalPlugin::generate(const RawOperator &producer) {
	/* XXX Later on, populate this function to simplify Nest */
	string error_msg = string("[BinaryInternalPlugin: ] Not to be used by Scan op.");
	LOG(ERROR) << error_msg;
	throw runtime_error(error_msg);
}

/**
 * The work of readPath() and readValue() has been taken care of scanCSV()
 */
RawValueMemory BinaryInternalPlugin::readPath(string activeRelation, Bindings bindings, const char* pathVar)	{
	RawValueMemory mem_valWrapper;
	{
		const OperatorState* state = bindings.state;
		const map<RecordAttribute, RawValueMemory>& binProjections = state->getBindings();
		RecordAttribute tmpKey = RecordAttribute(structName,pathVar,this->getOIDType());
		map<RecordAttribute, RawValueMemory>::const_iterator it;
		it = binProjections.find(tmpKey);
			if (it == binProjections.end()) {
				string error_msg = string("[BinaryInternalPlugin - readPath ]: Unknown variable name ")+pathVar;
				LOG(ERROR) << error_msg;
				throw runtime_error(error_msg);
			}
		mem_valWrapper = it->second;
	}
	return mem_valWrapper;
}

RawValueMemory BinaryInternalPlugin::readValue(RawValueMemory mem_value, const ExpressionType* type)	{
	return mem_value;
}

RawValue BinaryInternalPlugin::hashValue(RawValueMemory mem_value, const ExpressionType* type)	{
	IRBuilder<>* Builder = context->getBuilder();
	switch (type->getTypeID())
	{
	case BOOL:
	{
		Function *hashBoolean = context->getFunction("hashBoolean");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashBoolean,
				ArgsV, "hashBoolean");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case STRING:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] String datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] String datatypes not supported yet"));
	}
	case FLOAT:
	{
		Function *hashDouble = context->getFunction("hashDouble");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashDouble, ArgsV, "hashDouble");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case INT:
	{
		Function *hashInt = context->getFunction("hashInt");
		vector<Value*> ArgsV;
		ArgsV.push_back(Builder->CreateLoad(mem_value.mem));
		Value *hashResult = context->getBuilder()->CreateCall(hashInt, ArgsV, "hashInt");

		RawValue valWrapper;
		valWrapper.value = hashResult;
		valWrapper.isNull = context->createFalse();
		return valWrapper;
	}
	case BAG:
	case LIST:
	case SET:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Collection datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Collection datatypes not supported yet"));
	}
	case RECORD:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Record-valued datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Record-valued datatypes not supported yet"));
	}
	default:
	{
		LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
		throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
	}
}
}

void BinaryInternalPlugin::flushValue(RawValueMemory mem_value, const ExpressionType *type,
		Value* fileName)
{
	IRBuilder<>* Builder = context->getBuilder();
	Function *flushFunc;
	Value* val_attr = Builder->CreateLoad(mem_value.mem);
	switch (type->getTypeID())
	{
	case BOOL:
	{
		flushFunc = context->getFunction("flushBoolean");
		vector<Value*> ArgsV;
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
		return;
	}
	case STRING:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] String datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] String datatypes not supported yet"));
	}
	case FLOAT:
	{
		flushFunc = context->getFunction("flushDouble");
		vector<Value*> ArgsV;
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc,ArgsV);
		return;
	}
	case INT:
	{
		vector<Value*> ArgsV;
		flushFunc = context->getFunction("flushInt");
		ArgsV.push_back(val_attr);
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc,ArgsV);
		return;
	}
	case BAG:
	case LIST:
	case SET:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Collection datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Collection datatypes not supported yet"));
	}
	case RECORD:
	{
		LOG(ERROR)<< "[BinaryInternalPlugin: ] Record-valued datatypes not supported yet";
		throw runtime_error(string("[BinaryInternalPlugin: ] Record-valued datatypes not supported yet"));
	}
	default:
	{
		LOG(ERROR) << "[CSV PLUGIN: ] Unknown datatype";
		throw runtime_error(string("[CSV PLUGIN: ] Unknown datatype"));
	}
}
}


void BinaryInternalPlugin::finish()	{

}

Value* BinaryInternalPlugin::getValueSize(RawValueMemory mem_value,
		const ExpressionType* type) {
	switch (type->getTypeID()) {
	case BOOL:
	case INT:
	case FLOAT: {
		Type *explicitType = (mem_value.mem)->getAllocatedType();
		return ConstantExpr::getSizeOf(explicitType);
		//return context->createInt32(explicitType->getPrimitiveSizeInBits() / 8);
	}
	case STRING: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Strings not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case BAG:
	case LIST:
	case SET: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Collections not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	case RECORD: {
		string error_msg = string(
				"[BinaryInternalPlugin]: Records not supported yet");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	default: {
		string error_msg = string("[BinaryInternalPlugin]: Unknown datatype");
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	}
}

