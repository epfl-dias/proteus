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

#include "plugins/json-plugin.hpp"

namespace semi_index	{

JSONPlugin::JSONPlugin(RawContext* const context, string& fname,
		vector<RecordAttribute*>* fieldsToSelect,
		vector<RecordAttribute*>* fieldsToProject) :
		attsToProject(fieldsToSelect), attsToSelect(fieldsToProject), context(context), fname(fname) {

	helper = new JSONHelper(fname, fieldsToSelect, fieldsToProject);
	RawCatalog& catalog = RawCatalog::getInstance();
	catalog.setJSONHelper(fname,helper);
}

void JSONPlugin::init() {}

void JSONPlugin::scanJSON(const RawOperator& producer, Function* debug) {

	//Prepare
	LLVMContext& llvmContext = context->getLLVMContext();
	Type* charPtrType = Type::getInt8PtrTy(llvmContext);
	Type* int64Type = Type::getInt64Ty(llvmContext);
	IRBuilder<>* Builder = context->getBuilder();

	//Get the entry block
	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	//Container for the variable bindings
	map<RecordAttribute, AllocaInst*>* variableBindings = new map<RecordAttribute, AllocaInst*>();

	//Loop through results (if any)
	BasicBlock *jsonScanCond, *jsonScanBody, *jsonScanInc, *jsonScanEnd;
	context->CreateForLoop("jsonScanCond", "jsonScanBody", "jsonScanInc","jsonScanEnd",
							&jsonScanCond, &jsonScanBody, &jsonScanInc,	&jsonScanEnd);

	JSONHelper* helper = this->helper;
	int attsNo = helper->getAttsNo();
	int nameSize = (helper->getFileNameStr()).size() + 1;
	char* filename_noconst = (char *) alloca(nameSize);
	memcpy(filename_noconst, helper->getFileName(), nameSize);
	Value* filenameLLVM = context->CreateGlobalString(filename_noconst);
	std::vector<Value*> ArgsV;

	//Flushing code in Entry Block
	Builder->CreateBr(jsonScanCond);

	//Condition! --> while (!(cursor == json_semi_index::cursor()))
	//Prepare eof function arguments
	Builder->SetInsertPoint(jsonScanCond);
	ArgsV.clear();
	ArgsV.push_back(filenameLLVM);
	Function* endCond = context->getFunction("eofJSON");
	Value* endCheck = Builder->CreateCall(endCond, ArgsV/*,"eofJSON"*/);
	Value* falseLLVM = context->createFalse();
	ICmpInst* eof_cmp = new ICmpInst(*jsonScanCond, ICmpInst::ICMP_EQ, endCheck,
			falseLLVM, "cmpJSONEnd");
	BranchInst::Create(jsonScanBody, jsonScanEnd, eof_cmp, jsonScanCond);

	//BODY
	Builder->SetInsertPoint(jsonScanBody);
	std::set<RecordAttribute *, bool (*)(RecordAttribute *, RecordAttribute *)>* allAtts =
			helper->getAllAtts();

	int i = 0;
	for (std::set<RecordAttribute*>::iterator it = allAtts->begin(); it != allAtts->end(); it++) {
		ArgsV.clear();
		ExpressionType* type = (*it)->getOriginalType();
		Value* currAttrNo = context->createInt32(i++);
		ArgsV.push_back(filenameLLVM);
		ArgsV.push_back(currAttrNo);
		string currAttrName = (*it)->getName();

		if ((*it)->isProjected() && !(type->isPrimitive())) {
			Function* jsonObjectScan = context->getFunction("getJSONPositions");
			LOG(INFO) << "[Scan - JSON: ] Non-primitive datatype " << currAttrName
					  << " requested from " << helper->getFileNameStr();

			//Preparing JSON struct
			Value* positionStructs = Builder->CreateCall(jsonObjectScan, ArgsV);
			Type* jsonStructType = context->CreateJSONPosStruct();
			AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,
					"currJSONResult", jsonStructType);
			Builder->CreateStore(positionStructs, Alloca);
			(*variableBindings)[*(*it)] = Alloca;
		} else {
			Value* result;
			Type* resultType;
			Function* jsonPrimitiveScan;
			LOG(INFO) << "[Scan - JSON: ] Primitive datatype " << currAttrName
					  << " requested from " << helper->getFileNameStr();
			switch (type->getTypeID()) {
			case BOOL:
				throw runtime_error(string("Not Supported Yet: ") + (*it)->getType());
			case STRING:
				jsonPrimitiveScan = context->getFunction("getJSONPositions");
				resultType = context->CreateJSONPosStruct();
				break;
			case FLOAT:
				jsonPrimitiveScan = context->getFunction("getJSONDouble");
				resultType = Type::getDoubleTy(llvmContext);
				break;
			case INT:
				jsonPrimitiveScan = context->getFunction("getJSONInt");
				resultType = Type::getInt32Ty(llvmContext);
				break;
			case BAG:
			case LIST:
			case SET:
				throw runtime_error(string("Only primitive types should qualify here: ")
								           + (*it)->getType());
			case RECORD:
				throw runtime_error(string("Only primitive types should qualify here: ")
								           + (*it)->getType());
			default:
				throw runtime_error(string("Unknown type: ") + (*it)->getType());
			}

			result = Builder->CreateCall(jsonPrimitiveScan, ArgsV);
			AllocaInst *Alloca = context->CreateEntryBlockAlloca(TheFunction,"currJSONResult", resultType);
			Builder->CreateStore(result, Alloca);
			(*variableBindings)[*(*it)] = Alloca;
		}
	}

	Builder->CreateBr(jsonScanInc);
	Builder->SetInsertPoint(jsonScanInc);


	//Triggering parent
	OperatorState* state = new OperatorState(producer, *variableBindings);
	RawOperator* const opParent = producer.getParent();
	opParent->consume(context,*state);

	// Block for.inc (scan_inc)
	Builder->CreateBr(jsonScanCond);

	// Block for.end (label_for_end)
	Builder->SetInsertPoint(jsonScanEnd);
	LOG(INFO) << "[Scan - JSON: ] End of scan code @ plugin";
}

//TODO Fill stub method
void JSONPlugin::finish() {
}

void JSONPlugin::generate(const RawOperator& producer) {
	scanJSON(producer, context->getGlobalFunction());
}

JSONPlugin::~JSONPlugin() {}

}
