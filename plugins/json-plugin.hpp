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

#ifndef JSON_PLUGIN_HPP_
#define JSON_PLUGIN_HPP_

#include "plugins/plugins.hpp"
#include "util/atois.hpp"
#include "util/raw-catalog.hpp"
#include "util/raw-caching.hpp"

//#define DEBUGJSON

namespace jsonPipelined	{

typedef struct pmJSON {
	size_t *newlines;
	jsmntok_t **tokens;
} pmJSON;

/**
 * JSON's basic types are:
 * Number: A signed decimal number that may contain a fractional part and may use exponential E notation.
 * 		  JSON does not allow non-numbers like NaN, nor does it make any distinction between integer and
 * 		  floating-point.
 *
 * String: A sequence of zero or more Unicode characters.
 *         Strings are delimited with double-quotation marks and support a backslash escaping syntax.
 *
 * Boolean: either of the values true or false
 *
 * Array: An ORDERED LIST of zero or more values, each of which may be of any type.
 * 		  Arrays use square bracket notation with elements being comma-separated.
 *
 * Object: An UNORDERED ASSOCIATIVE ARRAY (name/value pairs).
 *         Objects are delimited with curly brackets and use commas to separate each pair.
 *         All keys must be strings and should be DISTINCT from each other within that object.
 *
 * null — An empty value, using the word null
 *
 * Token struct contents:
 * 0: jsmntype_t type;
 * 1: int start;
 * 2: int end;
 * 3: int size;
 */

/**
 * Assumptions:
 * 1. Each row contains A JSON OBJECT.
 * 	  This is the format that the majority of (db) vendors require
 * 	  when working with JSON
 * 2. FIXME Currently constructs PM from scratch, in pipelined fashion.
 * 	  Need one more plugin (or one more constructor), taking the PM as granted
 * 	  (Actually, we check if a PM is already cached)
 */

/**
 * FIXME offset can reside in a single array, much like newlines in csv_pm ->
 * No need for it to be a part of the 'tupleIdentifier'
 */
class JSONPlugin : public Plugin	{
public:
	/* XXX Do NOT use this constructor with large inputs until realloc() is implemented for lines */
	/* Deprecated */
//	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema);
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema, size_t linehint = 1000);
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema, size_t linehint, bool staticSchema);
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema, size_t linehint, jsmntok_t **tokens);
	~JSONPlugin();
	void init()															{}
	void generate(const RawOperator& producer);
	void finish();
	string& getName() 													{ return fname; }

	//1-1 correspondence with 'RecordProjection' expression
	virtual RawValueMemory readPath(string activeRelation, Bindings wrappedBindings, const char* pathVar, RecordAttribute attr);
	virtual RawValueMemory readPredefinedPath(string activeRelation, Bindings wrappedBindings, RecordAttribute attr);
	virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue readCachedValue(CacheInfo info, const OperatorState& currState);
	virtual RawValue readCachedValue(CacheInfo info, const map<RecordAttribute, RawValueMemory>& bindings);


	virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);
	virtual RawValue hashValueEager(RawValue value, const ExpressionType* type);

	/**
	 * XXX VERY strong JSON-specific assumption (pretty much hard-coding) that we can just grab a chunk
	 * of the input and flush it w/o caring what is the final serialization format
	 *
	 * Both also assume that input is an OID (complex one)
	 */
	virtual void flushTuple(RawValueMemory mem_value, Value* fileName)	{ flushChunk(mem_value, fileName); }
	virtual void flushValue(RawValueMemory mem_value, const ExpressionType *type, Value* fileName)  { flushChunk(mem_value, fileName); }
	virtual void flushValueEager(RawValue value, const ExpressionType *type, Value* fileName);
	void flushChunk(RawValueMemory mem_value, Value* fileName);

	virtual Value* getValueSize(RawValueMemory mem_value, const ExpressionType* type);

	//Used by unnest
	virtual RawValueMemory initCollectionUnnest(RawValue val_parentTokenNo);
	virtual RawValue collectionHasNext(RawValue parentTokenId, RawValueMemory mem_currentTokenId);
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentToken);

	void scanObjects(const RawOperator& producer, Function* debug);

//	virtual typeID getOIDSize() { return INT; }
	virtual ExpressionType *getOIDType() {
		Int64Type *int64Type = new Int64Type();

		string field1 = string("offset");
		string field2 = string("rowId");
		string field3 = string("tokenNo");

		RecordAttribute *attr1 = new RecordAttribute(1, fname, field1, int64Type);
		RecordAttribute *attr2 = new RecordAttribute(2, fname, field2, int64Type);
		RecordAttribute *attr3 = new RecordAttribute(3, fname, field3, int64Type);
		list<RecordAttribute*> atts = list<RecordAttribute*>();
		atts.push_back(attr1);
		atts.push_back(attr2);
		atts.push_back(attr3);
		RecordType *inner = new RecordType(atts);
		return inner;
	}

	virtual PluginType getPluginType() { return PGJSON; }

	jsmntok_t** getTokens()	{ return tokens; }

	//	void freeTokens() {
//		for(int i = 0; i < lines; i++)	{
//			free(tokens[i]);
//		}
//		free(tokens);
//	}

private:
	string& fname;
	size_t fsize;
	int fd;
	const char* buf;
	bool staticSchema;

	StructType *tokenType;

	/* Specify whether the tokens array will be provided to the PG */
	bool cache;
	/* Specify whether the newlines array will be provided to the PG */
	bool cacheNewlines;
	/* 1-D array of tokens PER ROW => 2D */
	/* 1-D array of newlines offsets per row */
	jsmntok_t **tokens;
	char *tokenBuf;
	size_t *newLines;
	AllocaInst *mem_tokenArray;
	AllocaInst *mem_newlineArray;

	/* Start with token value - increase when needed */
	size_t lines;

	//Code-generation-related
	//Used to store memory positions of offset, buf and filesize in the generated code
	map<string, AllocaInst*> NamedValuesJSON;
	RawContext* const context;

	//Assumption (1) applies here
	ExpressionType* schema;

	/* Remember: token != OID */
	StructType *getOIDLLVMType()	{
		LLVMContext& llvmContext = context->getLLVMContext();
		Type* int64Type = Type::getInt64Ty(llvmContext);
		vector<Type*> tokenIdMembers;
		tokenIdMembers.push_back(int64Type);
		tokenIdMembers.push_back(int64Type);
		tokenIdMembers.push_back(int64Type);
		return StructType::get(context->getLLVMContext(),tokenIdMembers);
	}

	//Cannot implement such a function. Arrays have non-fixed number of values.
	//Similarly, objects don't always have same number of fields
	//int calculateTokensPerItem(const ExpressionType& expr);

	/*
	 * Codegen part
	 */
	const char* var_buf;
	const char* var_tokenPtr;
	const char* var_tokenOffset;
	const char* var_tokenOffsetHash; //Needed to guide hashing process
	void skipToEnd();
	RawValueMemory readPathInternal(RawValueMemory mem_parentTokenId, const char* pathVar);

};

}
#endif /* JSON_PLUGIN_HPP_ */
