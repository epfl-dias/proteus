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

#include "plugins/output/plugins-output.hpp"

Materializer::Materializer(const vector<RecordAttribute*>& wantedFields,
//		const vector<expressions::Expression*>& wantedExpressions,
		const vector<materialization_mode>& outputMode) :
//		wantedExpressions(wantedExpressions),
		wantedFields(wantedFields), outputMode(outputMode)
{
}

OutputPlugin::OutputPlugin(RawContext* const context, Materializer& materializer, const map<RecordAttribute, RawValueMemory>& bindings )
	: context(context), materializer(materializer), currentBindings(bindings)	{

	/**
	 * TODO PAYLOAD SIZE IS NOT A DECISION TO BE MADE BY THE PLUGIN
	 */
	const vector<RecordAttribute*>& wantedFields = materializer.getWantedFields();
	// Declare our result (value) type
	materializedTypes = new vector<Type*>();
	int payload_type_size = 0;
	int tupleIdentifiers = 0;
	//Always materializing the 'active tuples' pointer
	RawValueMemory mem_activeTuple;
	{
		map<RecordAttribute, RawValueMemory>::const_iterator memSearch;
		//FIXME use 'find' instead of looping
		for (memSearch = currentBindings.begin(); memSearch != currentBindings.end(); memSearch++) {

			RecordAttribute currAttr = memSearch->first;
			//cout << "HINT: " << currAttr.getOriginalRelationName() << "_" << currAttr.getName() << endl;
			if (currAttr.getAttrName() == activeLoop) {
				Type* currType = (memSearch->second).mem->getAllocatedType();
				materializedTypes->push_back(currType);
				payload_type_size += (currType->getPrimitiveSizeInBits() / 8);
				tupleIdentifiers++;
				materializer.addTupleIdentifier(currAttr);
			}
		}
	}
	int attrNo=0;
	/**
	 * XXX
	 * What if fields not materialized (yet)? This will crash w/o cause
	 * Atm, that's always the case when dealing with e.g. JSON that is lazily materialized
	 */
	for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		map<RecordAttribute, RawValueMemory>::const_iterator itSearch = currentBindings.find(*(*it));
		//Field needed
		if(itSearch != currentBindings.end())
		{
			LOG(INFO)<<"[MATERIALIZER: ] PART OF PAYLOAD: "<<(*it)->getAttrName();
			materialization_mode mode = (materializer.getOutputMode()).at(attrNo++);
			//gather datatypes
			Type* currType = (itSearch->second).mem->getAllocatedType();
			Type* requestedType = chooseType((*it)->getOriginalType(),currType, mode);
			materializedTypes->push_back(requestedType);
			payload_type_size += (requestedType->getPrimitiveSizeInBits() / 8);
		}
		/*Commented because of previous comment */
		else {
			string error_msg = string("[OUTPUT PG: ] INPUT ERROR AT OPERATOR - UNKNOWN WANTED FIELD ") + (*it)->getAttrName();
			LOG(ERROR) << error_msg;
			throw runtime_error(error_msg);
		}
	}
	LOG(INFO)<<"[OUTPUT PLUGIN: ] Size of tuple (payload): "<<payload_type_size;

	//Result type specified
	payloadType = llvm::StructType::get(context->getLLVMContext(),*materializedTypes);
	payloadTypeSize = payload_type_size;
}

//TODO Many more cases to cater for
//Current most relevant example: JSONObject
Type* OutputPlugin::chooseType(const ExpressionType* exprType, Type* currType, materialization_mode mode)	{

	switch(exprType->getTypeID())
	{
	case BAG:
	case SET:
	case LIST:
		LOG(ERROR) << "[OUTPUT PG: ] DEALING WITH COLLECTION TYPE - NOT SUPPORTED YET";
		throw runtime_error(string("[OUTPUT PG: ] DEALING WITH COLLECTION TYPE - NOT SUPPORTED YET"));
		break;
	case RECORD: //I am not sure if this is case can occur
		LOG(ERROR) << "[OUTPUT PG: ] DEALING WITH RECORD TYPE - NOT SUPPORTED YET";
		throw runtime_error(string("[OUTPUT PG: ] DEALING WITH RECORD TYPE - NOT SUPPORTED YET"));
	case BOOL:
	case STRING:
	case FLOAT:
	case INT:
		switch(mode)	{
		case EAGER:
			return currType;
		case LAZY:
		case INTERMEDIATE:
			LOG(ERROR) << "[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - ONLY 'EAGER' CONVERSION AVAILABLE";
			throw runtime_error(string("[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - ONLY 'EAGER' CONVERSION AVAILABLE"));
		case BINARY:
			LOG(WARNING) <<"[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - 'BINARY' MODE ALREADY ACTIVE";
			break;
		case CSV:
		case JSON:
			LOG(ERROR) << "[OUTPUT PG: ] CONVERSION TO 'RAW' FORM NOT SUPPORTED (YET)";
			throw runtime_error(string("[OUTPUT PG: ] CONVERSION TO 'RAW' FORM NOT SUPPORTED (YET)"));
		}
		break;
	}

	return currType;
}

//TODO Must add functionality - currently only thin shell exists
Value* OutputPlugin::convert(Type* currType, Type* materializedType, Value* val)	{
	if(currType == materializedType)	{
		return val;
	}
	//TODO Many more cases to cater for here
	return val;
}
