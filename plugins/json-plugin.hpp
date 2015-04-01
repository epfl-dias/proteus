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
#include "util/raw-catalog.hpp"

#define DEBUGJSON

namespace jsonPipelined	{

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
 */

/**
 * Assumptions:
 * 1. Each row contains A JSON OBJECT.
 * 	  This is the format that the majority of (db) vendors require
 * 	  when working with JSON
 * 2. FIXME Currently constructs PM from scratch, in pipelined fashion.
 * 	  Need one more plugin (or one more constructor), taking the PM as granted
 */
class JSONPlugin : public Plugin	{
public:
	/* XXX Do NOT use this constructor with large inputs until realloc() is implemented for lines */
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema);
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema, size_t linehint);
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema, size_t linehint, jsmntok_t **tokens);
	~JSONPlugin();
	void init()															{}
	void generate(const RawOperator& producer);
	void finish();
	string& getName() 													{ return fname; }

	//1-1 correspondence with 'RecordProjection' expression
	virtual RawValueMemory readPath(string activeRelation, Bindings wrappedBindings, const char* pathVar);
	virtual RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);

	virtual RawValue hashValue(RawValueMemory mem_value, const ExpressionType* type);

	/**
	 * XXX VERY strong JSON-specific assumption (pretty much hard-coding) that we can just grab a chunk
	 * of the input and flush it w/o caring what is the final serialization format
	 */
	virtual void flushTuple(RawValueMemory mem_value, Value* fileName)	{ flushChunk(mem_value, fileName); }
	virtual void flushValue(RawValueMemory mem_value, ExpressionType *type, Value* fileName)  { flushChunk(mem_value, fileName); }
	void flushChunk(RawValueMemory mem_value, Value* fileName);

	virtual Value* getValueSize(RawValueMemory mem_value, const ExpressionType* type);

	//Used by unnest
	virtual RawValueMemory initCollectionUnnest(RawValue val_parentTokenNo);
	virtual RawValue collectionHasNext(RawValue parentTokenId, RawValueMemory mem_currentTokenId);
	virtual RawValueMemory collectionGetNext(RawValueMemory mem_currentToken);

	void scanObjects(const RawOperator& producer, Function* debug);
	void scanObjectsInterpreted(list<string> path, list<ExpressionType*> types);
	void scanObjectsEagerInterpreted(list<string> path, list<ExpressionType*> types);
	void unnestObjectsInterpreted(list<string> path);
	void unnestObjectInterpreted(int parentToken);
	int readPathInterpreted(int parentToken, list<string> path);
	void readValueInterpreted(int tokenNo, const ExpressionType* type);
	void readValueEagerInterpreted(int tokenNo, const ExpressionType* type);

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

	StructType *tokenType;

	/* Specify whether the tokens array will be provided to the PG */
	bool cache;
	/* 1-D array of tokens PER ROW => 2D */
	jsmntok_t **tokens;
	char *tokenBuf;
	AllocaInst *mem_tokenArray;

	/* Start with token value - increase when needed */
	size_t lines;

	//Code-generation-related
	//Used to store memory positions of offset, buf and filesize in the generated code
	map<string, AllocaInst*> NamedValuesJSON;
	RawContext* const context;

	//Assumption (1) applies here
	ExpressionType* schema;


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