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
#include "util/atois.hpp"
#include "util/raw-caching.hpp"


#define DEBUGPM

typedef struct pmCSV {
	size_t *newlines;
	short **offsets;
} pmCSV;

namespace pm {

class CSVPlugin: public Plugin {
public:

	/**
	 * Fully eager CSV Plugin.
	 * Populates PM.
	 *
	 */
	CSVPlugin(RawContext* const context, string& fname, RecordType& rec,
			vector<RecordAttribute*> whichFields, int lineHint, int policy);
	/* PM Ready */
	CSVPlugin(RawContext* const context, string& fname, RecordType& rec,
				vector<RecordAttribute*> whichFields, int lineHint, int policy,
				size_t *newlines, short **offsets);
	~CSVPlugin();
	virtual string& getName() {
		return fname;
	}
	void init();
	void generate(const RawOperator& producer);
	void finish();
	virtual RawValueMemory readPath(string activeRelation, Bindings bindings,
			const char* pathVar);
	virtual RawValueMemory readValue(RawValueMemory mem_value,
			const ExpressionType* type);
	virtual RawValue readCachedValue(CacheInfo info, const OperatorState& currState);

	virtual RawValue hashValue(RawValueMemory mem_value,
			const ExpressionType* type);

	virtual void flushTuple(RawValueMemory mem_value, Value* fileName) {
		string error_msg = "[CSVPlugin: ] Functionality not supported yet";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushValue(RawValueMemory mem_value, const ExpressionType *type,
			Value* fileName);

	virtual RawValueMemory initCollectionUnnest(RawValue val_parentObject) {
		string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue collectionHasNext(RawValue val_parentObject,
			RawValueMemory mem_currentChild) {
		string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentChild) {
		string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
		LOG(ERROR)<< error_msg;
		throw runtime_error(error_msg);
	}

	virtual Value* getValueSize(RawValueMemory mem_value,
			const ExpressionType* type);

	/* Export PM */
	/* XXX I think it's the 'Caching Service' that should
	 * be making the PM available later on */
	short** getOffsetsPM()	{ return pm; }
	size_t* getNewlinesPM() { return newlines; }
//	virtual typeID getOIDSize() {
//		return INT;
//	}
	virtual ExpressionType *getOIDType() {
		return new IntType();
	}

	virtual PluginType getPluginType() { return PGCSV; }

private:
	string& fname;
	off_t fsize;
	int fd;
	char *buf;
	//Schema info provided
	RecordType& rec;
	vector<RecordAttribute*> wantedFields;

	/**
	 * PM-related
	 */
	int lines;
	int policy;
	/* Indicates whether a PM was provided at construction time*/
	bool hasPM;
	size_t *newlines;
	/* All pm entries are relevant to linesStart!!! */
	short **pm;


	AllocaInst *mem_newlines;
	AllocaInst *mem_pm;
	AllocaInst *mem_lineCtr;

	/**
	 * Code-generation-related
	 */
	//Used to store memory positions of offset, buf and filesize in the generated code
	map<string, AllocaInst*> NamedValuesCSV;
	RawContext* const context;

	const char* posVar; // = "offset";
	const char* bufVar; // = "fileBuffer";
	const char* fsizeVar; // = "fileSize";

	//Used to generate code
	void skipDelimLLVM(Value* delim, Function* debugChar, Function* debugInt);
	void skipDelimLLVM(Value* delim);
	void skipDelimBackwardsLLVM(Value* delim);
	void skipLLVM();
	void skipToEndLLVM();
	void getFieldEndLLVM();
	void readField(typeID id, RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	void readAsIntLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables, Function* atoi_,
			Function* debugChar, Function* debugInt);
	void readAsIntLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	void readAsFloatLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables, Function* atof_,
			Function* debugChar, Function* debugFloat);
	void readAsFloatLLVM(RecordAttribute attName,
				map<RecordAttribute, RawValueMemory>& variables);
	void readAsBooleanLLVM(RecordAttribute attName,
			map<RecordAttribute, RawValueMemory>& variables);
	/* Assumption: String comes bracketted */
	void readAsStringLLVM(RecordAttribute attName,
				map<RecordAttribute, RawValueMemory>& variables);

	//Generates a for loop that performs the file scan
	//No assumption on auxiliary structures yet
	void scanAndPopulatePM(const RawOperator& producer);
	void scanPM(const RawOperator& producer);
};

}
