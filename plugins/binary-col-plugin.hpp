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

class BinaryColPlugin	: public Plugin {
public:

	/**
	 * Plugin for binary files, organized in COLUMNAR format
	 * Emulating access to a column store's native storage
	 *
	 * Support for
	 * -> integers
	 * -> floats
	 * -> chars & booleans
	 * -> varchars & dates
	 *
	 *
	 * Conventions / Assumptions:
	 * -> filenames have the form fnamePrefix.<fieldName>
	 * -> integers, floats, chars and booleans are stored as tight arrays
	 * -> varchars are stored using a dictionary.
	 * 		Assuming the dictionary file is named fnamePrefix.<fieldName>.dict
	 * -> dates require more sophisticated serialization (boost?)
	 *
	 * EACH FILE CONTAINS A size_t COUNTER AS ITS FIRST ENTRY!
	 * => Looping does not depends on fsize: it depends on this 'size/length'
	 */

	BinaryColPlugin(RawContext* const context, string& fnamePrefix, RecordType& rec, vector<RecordAttribute*>& whichFields);
	~BinaryColPlugin();
	virtual string& getName() { return fnamePrefix; }
	void init();
	void generate(const RawOperator& producer);
	void finish();
	virtual RawValueMemory readPath(string activeRelation, Bindings bindings, const char* pathVar);
	virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);

	virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);

	virtual RawValueMemory initCollectionUnnest(RawValue val_parentObject) {
		string error_msg = "[BinaryColPlugin: ] Binary col. files do not contain collections";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValue collectionHasNext(RawValue val_parentObject,
			RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryColPlugin: ] Binary col. files do not contain collections";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentChild) {
		string error_msg = "[BinaryColPlugin: ] Binary col. files do not contain collections";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushTuple(RawValueMemory mem_value, Value* fileName) {
			string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}

	virtual void flushValue(RawValueMemory mem_value, ExpressionType *type, Value* fileName)	{
		string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}

	virtual void flushChunk(RawValueMemory mem_value, Value* fileName)	{
		string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
		LOG(ERROR) << error_msg;
		throw runtime_error(error_msg);
	}

private:
	//Schema info provided
	RecordType& rec;
	vector<RecordAttribute*>& wantedFields;

	string& fnamePrefix;
	off_t *colFilesize; //Size of each column
	int *fd; //One per column
	char **buf;

	//Mapping attrNumber to
	//	-> file descriptor of its dictionary
	//	-> file size of its dictionary
	//	-> mapped input of its dictionary
	map<int,int> dictionaries;
	map<int,off_t> dictionaryFilesizes;
	//Note: char* buf can be cast to appropriate struct
	//Struct will probably look like { implicit_oid, len, char* }
	map<int,char*> dictionariesBuf;

	/**
	 * Code-generation-related
	 */
	//Used to store memory positions of offset, buf and filesize in the generated code
	map<string, AllocaInst*> NamedValuesBinaryRow;
	RawContext* const context;

	const char* posVar;		// = "offset";
	const char* bufVar;		// = "fileBuffer";
	const char* fsizeVar;	// = "fileSize";
	const char* sizeVar;	// = "size";

	//Used to generate code
	void skipLLVM(string attName, Value *offset);
	void readAsInt64LLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	Value* readAsInt64LLVM(RecordAttribute attName);
	void readAsIntLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsStringLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsFloatLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);
	void readAsBooleanLLVM(RecordAttribute attName, map<RecordAttribute, RawValueMemory>& variables);

	//Generates a for loop that performs the file scan
	void scan(const RawOperator& producer, Function *f);
};
