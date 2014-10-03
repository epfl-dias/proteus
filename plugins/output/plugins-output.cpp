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

Materializer::Materializer(const vector<RecordAttribute*>& wantedFields, const vector<materialization_mode>& outputMode)
	: wantedFields(wantedFields), outputMode(outputMode) {}

OutputPlugin::OutputPlugin(RawContext* const context, const Materializer& materializer, const std::map<string, AllocaInst*>& bindings )
	: context(context), materializer(materializer), currentBindings(bindings)	{

	const vector<RecordAttribute*>& wantedFields = materializer.getWantedFields();
	// Declare our result (value) type
	materializedTypes = new std::vector<Type*>();
	int payload_type_size = 0;

	int attrNo=0;
	for(vector<RecordAttribute*>::const_iterator it = wantedFields.begin(); it != wantedFields.end(); it++)	{
		string bindingName = (*it)->getName();
		std::map<string, AllocaInst*>::const_iterator itSearch = currentBindings.find(bindingName);
		//Field needed
		if(itSearch != currentBindings.end())
		{
			LOG(INFO)<<"[MATERIALIZER: ] PART OF PAYLOAD: "<<bindingName;
			materialization_mode mode = (materializer.getOutputMode()).at(attrNo++);
			//gather datatypes
			Type* currType = itSearch->second->getAllocatedType();
			Type* requestedType = chooseType((*it)->getOriginalType(),currType, mode);
			materializedTypes->push_back(requestedType);
			payload_type_size += (requestedType->getPrimitiveSizeInBits() / 8);
		} else {
			LOG(ERROR) << "[OUTPUT PG: ] INPUT ERROR AT OPERATOR - UNKNOWN WANTED FIELD "<<bindingName;
			throw runtime_error(string("[OUTPUT PG: ] INPUT ERROR AT OPERATOR - UNKNOWN WANTED FIELD "+bindingName));
		}
	}
	LOG(INFO)<<"[OUTPUT PLUGIN: ] Size of tuple (payload): "<<payload_type_size;

	//Result type specified
	payloadType = llvm::StructType::get(context->getLLVMContext(),*materializedTypes);
	payloadTypeSize = payload_type_size;
};

//TODO Many more cases to cater for
//Current most relevant example: JSONObject
Type* OutputPlugin::chooseType(ExpressionType* exprType, Type* currType, materialization_mode mode)	{

	switch(exprType->getTypeID())
	{
	case COLLECTION:
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
