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

#ifndef BINARY_INTERNAL_PLUGIN_HPP_
#define BINARY_INTERNAL_PLUGIN_HPP_

#include "plugins/plugins.hpp"
#include "util/atois.hpp"

//#define DEBUGBINCACHE

class BinaryInternalPlugin	: public Plugin {
public:

	/**
	 * Plugin to be used for already serialized bindings.
	 * Example: After a nesting operator has taken place.
	 *
	 * Why not RecordType info?
	 * -> Because it requires declaring RecordAttributes
	 *    -> ..but RecordAttributes require an associated plugin
	 *    	 (chicken - egg prob.)
	 */
	BinaryInternalPlugin(RawContext* const context, string structName);
	/* Radix-related atm.
	 * Resembles BinaryRowPg */
	BinaryInternalPlugin(RawContext* const context, RecordType rec, string structName,
			vector<RecordAttribute*> whichOIDs, vector<RecordAttribute*> whichFields, CacheInfo info);
	~BinaryInternalPlugin();
	virtual string& getName() { return structName; }
	void init();
	void generate(const RawOperator& producer);
	void finish();
	virtual RawValueMemory readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr);
	virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue readCachedValue(CacheInfo info, const OperatorState& currState) {
		string error_msg = "[BinaryInternalPlugin: ] No caching support applicable";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue readCachedValue(CacheInfo info,
			const map<RecordAttribute, RawValueMemory>& bindings) {
		string error_msg =
				"[BinaryInternalPlugin: ] No caching support applicable";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue hashValueEager(RawValue value, const ExpressionType* type);

	virtual void flushTuple(RawValueMemory mem_value, Value* fileName)	{
		string error_msg = "[BinaryInternalPlugin: ] Functionality not supported yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushValue(RawValueMemory mem_value, const ExpressionType *type, Value* fileName);
	virtual void flushValueEager(RawValue value, const ExpressionType *type, Value* fileName);

	virtual RawValueMemory initCollectionUnnest(RawValue val_parentObject) {
		string error_msg = "[BinaryInternalPlugin: ] Functionality not supported yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue collectionHasNext(RawValue val_parentObject,
			RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryInternalPlugin: ] Functionality not supported yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryInternalPlugin: ] Functionality not supported yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual Value* getValueSize(RawValueMemory mem_value, const ExpressionType* type);

	virtual ExpressionType *getOIDType() {
		return new IntType();
	}

	virtual PluginType getPluginType() { return PGBINARY; }

	virtual void flushBeginList	(Value *fileName					) {}

	virtual void flushBeginBag	(Value *fileName					) {
		string error_msg = "[BinaryInternalPlugin: ] Binary-internal files do not contain BAGs";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushBeginSet	(Value *fileName					) {
		string error_msg = "[BinaryInternalPlugin: ] Binary-internal files do not contain SETs";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushEndList	(Value *fileName					) {}

	virtual void flushEndBag	(Value *fileName					) {
		string error_msg = "[BinaryInternalPlugin: ] Binary-internal files do not contain BAGs";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushEndSet	(Value *fileName					) {
		string error_msg = "[BinaryInternalPlugin: ] Binary-internal files do not contain SETs";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushDelim		(Value *fileName					, int depth) {
		Function *flushFunc = context->getFunction("flushChar");
		vector<Value*> ArgsV;
		//XXX JSON-specific -> Serializer business to differentiate
		ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}

	virtual void flushDelim		(Value *resultCtr, Value* fileName	, int depth) {
		Function *flushFunc = context->getFunction("flushDelim");
		vector<Value*> ArgsV;
		ArgsV.push_back(resultCtr);
		//XXX JSON-specific -> Serializer business to differentiate
		ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
		ArgsV.push_back(fileName);
		context->getBuilder()->CreateCall(flushFunc, ArgsV);
	}

private:
	string structName;

	/**
	 * Code-generation-related
	 */
	RawContext* const context;
	/* Radix-related atm */
	void scan(const RawOperator& producer);
	void scanStruct(const RawOperator& producer);
	//
	RecordType rec;
	//Number of entries, if applicable
	Value *val_entriesNo;
	/* Necessary if we are to iterate over the internal caches */

	vector<RecordAttribute*> fields;
	vector<RecordAttribute*> OIDs;

	StructType *payloadType;
	Value *mem_buffer;
	Value *val_structBufferPtr;
	/* Binary offset in file */
	AllocaInst *mem_pos;
	/* Tuple counter */
	AllocaInst *mem_cnt;

	/* Since we allow looping over cache, we must also extract fields
	 * while looping */
	void skipLLVM(Value* offset);
	void readAsIntLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	void readAsInt64LLVM(RecordAttribute attName,
				map<RecordAttribute, RawValueMemory>& variables);
	void readAsStringLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	void readAsFloatLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	void readAsBooleanLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
};

#endif
