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

#ifndef JSON_JSMN_PLUGIN_HPP_
#define JSON_JSMN_PLUGIN_HPP_

#include "plugins/plugins.hpp"
#include "util/raw-catalog.hpp"

namespace jsmn	{

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
 * 1. Outermost element is either an object ("List") holding identical elements,
 * 	  or an array of same logic. The former is handled by the parser, but is not strictly valid JSON
 */
class JSONPlugin : public Plugin	{
public:
	JSONPlugin(RawContext* const context, string& fname, ExpressionType* schema);
	~JSONPlugin();
	void init()															{}
	void generate(const RawOperator& producer);
	void finish();
	string& getName() 													{ return fname; }

	//1-1 correspondence with 'RecordProjection' expression
	RawValueMemory readPath(string activeRelation, Bindings wrappedBindings, const char* pathVar);
	RawValueMemory readValue(RawValueMemory mem_value, const ExpressionType* type);

	//Used by unnest
	RawValueMemory initCollectionUnnest(RawValue val_parentTokenNo);
	RawValue collectionHasNext(RawValue val_parentTokenNo, RawValueMemory mem_currentTokenNo);
	RawValueMemory collectionGetNext(RawValueMemory mem_currentToken);

	void scanObjects(const RawOperator& producer, Function* debug);
	void scanObjectsInterpreted(list<string> path, list<ExpressionType*> types);
	void unnestObjectsInterpreted(list<string> path);
	void unnestObjectInterpreted(int parentToken);
	int readPathInterpreted(int parentToken, list<string> path);
	void readValueInterpreted(int tokenNo, const ExpressionType* type);

private:
	string& fname;
	size_t fsize;
	int fd;
	const char* buf;

	//Code-generation-related
	//Used to store memory positions of offset, buf and filesize in the generated code
	std::map<std::string, AllocaInst*> NamedValuesJSON;
	RawContext* const context;

	//Assumption (1) applies here
	ExpressionType* schema;
	jsmntok_t *tokens;

	//Cannot implement such a function. Arrays have non-fixed number of values.
	//Similarly, objects don't always have same number of fields
	//int calculateTokensPerItem(const ExpressionType& expr);

	/*
	 * Codegen part
	 */
	const char* var_buf;
	const char* var_tokenPtr;
	const char* var_tokenOffset;
	void skipToEnd();
};

}
#endif /* JSON_JSMN_PLUGIN_HPP_ */
