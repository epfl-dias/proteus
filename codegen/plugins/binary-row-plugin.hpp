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

#include "plugins/plugins.hpp"

//XXX Tmp Assumption: String are of length 5!
class BinaryRowPlugin	: public Plugin {
public:

	/**
	 * Plugin for binary files, organized in tabular format
	 */
	BinaryRowPlugin(RawContext* const context, string& fname, RecordType& rec_, vector<RecordAttribute*>& whichFields);
	~BinaryRowPlugin();
	virtual string& getName() { return fname; }
	void init();
	void generate(const RawOperator& producer);
	void finish();
	virtual RawValueMemory readPath(string activeRelation, Bindings bindings, const char* pathVar, RecordAttribute attr);
	virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue readCachedValue(CacheInfo info,
			const OperatorState& currState) {
		string error_msg =
				"[BinaryRowPlugin: ] No caching support should be needed";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue readCachedValue(CacheInfo info,
			const map<RecordAttribute, RawValueMemory>& bindings) {
		string error_msg =
				"[BinaryRowPlugin: ] No caching support should be needed";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue hashValueEager(RawValue value, const ExpressionType* type);

	virtual RawValueMemory initCollectionUnnest(RawValue val_parentObject) {
		string error_msg = "[BinaryRowPlugin: ] Binary row files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue collectionHasNext(RawValue val_parentObject,
			RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryRowPlugin: ] Binary row files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryRowPlugin: ] Binary row files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushTuple(RawValueMemory mem_value, Value* fileName) {
			string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
			LOG(ERROR)<< error_msg;
			throw runtime_error(error_msg);
		}

	virtual void flushValue(RawValueMemory mem_value, const ExpressionType *type, Value* fileName)	{
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushValueEager(RawValue value, const ExpressionType *type,
			Value* fileName) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushChunk(RawValueMemory mem_value, Value* fileName)	{
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushBeginList	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushBeginBag	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushBeginSet	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushEndList	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushEndBag	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushEndSet	(Value *fileName					) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushDelim		(Value *fileName					, int depth) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushDelim		(Value *resultCtr, Value* fileName	, int depth) {
		string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual Value* getValueSize(RawValueMemory mem_value, const ExpressionType* type);
//	virtual typeID getOIDSize() { return INT; }

	virtual ExpressionType *getOIDType() {
		return new IntType();
	}

	virtual PluginType getPluginType() { return PGBINARY; }

private:
	string& fname;
	off_t fsize;
	int fd;
	char *buf;

	//Schema info provided
	RecordType& rec;
	vector<RecordAttribute*>& wantedFields;

	/**
	 * Code-generation-related
	 */
	//Used to store memory positions of offset, buf and filesize in the generated code
	map<string, AllocaInst*> NamedValuesBinaryRow;
	RawContext* const context;

	const char* posVar;// = "offset";
	const char* bufVar;// = "fileBuffer";
	const char* fsizeVar;// = "fileSize";

	//Used to generate code
	void skipLLVM(Value* offset);
	void readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);

	//Generates a for loop that performs the file scan
	void scan(const RawOperator& producer, Function *f);
};
