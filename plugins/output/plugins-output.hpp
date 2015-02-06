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

#ifndef PLUGINS_OUTPUT_HPP_
#define PLUGINS_OUTPUT_HPP_

#include "plugins/plugins.hpp"

class RawContext;

//TODO Refactor into multiple materializers
/**
 * What to do with each attribute contained in the payload of a materializing operation
 */
//Eager: Place attribute converted to its original datatype
//Lazy: Place pointer to attribute in the raw file
//Intermediate: Copy (unparsed) attribute and place the copy
//Binary: Convert to consise form before placing (current example: JSON to BSON)
enum materialization_mode {
	EAGER, LAZY, INTERMEDIATE, BINARY, CSV, JSON
};

class Materializer {

public:
	Materializer(const vector<RecordAttribute*>& whichFields,
//			const vector<expressions::Expression*>& wantedExpressions,
			const vector<materialization_mode>& outputMode_);
	~Materializer() {
	}

//	const vector<RecordAttribute*>& getWantedExpressions() const {
//			return wantedExpressions;
//	}
	const vector<RecordAttribute*>& getWantedFields() const {
		return wantedFields;
	}
	const vector<materialization_mode>& getOutputMode() const {
		return outputMode;
	}
	void addTupleIdentifier(RecordAttribute attr) {
		tupleIdentifiers.insert(attr);
		//XXX There is obviously a better way to remove duplicates
//		tupleIdentifiers.erase(unique(tupleIdentifiers.begin(), tupleIdentifiers.end()), tupleIdentifiers.end());
	}
	const set<RecordAttribute>& getTupleIdentifiers() const {
		return tupleIdentifiers;
	}
private:
	/**
	 *  XXX At some point, might Need expressions to (re-)evaluate bindings that
	   have not been eagerly evaluated.
	 */
	 //	const vector<expressions::Expression*>& wantedExpressions;


	const vector<RecordAttribute*>& wantedFields;
	const vector<materialization_mode>& outputMode;
	//int tupleIdentifiers;
	set<RecordAttribute> tupleIdentifiers;
};

class OutputPlugin {
public:
	OutputPlugin(RawContext* const context, Materializer& materializer,
			const map<RecordAttribute, RawValueMemory>& bindings);
	~OutputPlugin() {
	}

	llvm::StructType* getPayloadType() {
		return payloadType;
	}
	int getPayloadTypeSize() {
		return payloadTypeSize;
	}
	const map<RecordAttribute, RawValueMemory>& getBindings() const {
		return currentBindings;
	}
	std::vector<Type*>* getMaterializedTypes() {
		return materializedTypes;
	}
	Value* convert(Type* currType, Type* materializedType, Value* val);

private:
	//Schema info provided
	const Materializer& materializer;

	//Code-generation-related
	RawContext* const context;
	const map<RecordAttribute, RawValueMemory>& currentBindings;
	llvm::StructType* payloadType;
	std::vector<Type*>* materializedTypes;
	int payloadTypeSize;
	Type* chooseType(const ExpressionType* exprType, Type* currType,
			materialization_mode mode);
};

#endif /* PLUGINS_OUTPUT_HPP_ */
